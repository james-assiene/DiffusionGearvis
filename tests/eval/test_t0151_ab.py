"""T-0151 B2-B5 and B7: the locked 2-arm overview-frame A/B on S1-S5, re-derived from the run's records and compared with
``report.json`` (``gpu``, ``eval``; report-only plus the B5 decision rule, H-T151-1 approved 2026-10-24 with T-0150's
margins plus the S5 guard ``S5(B) >= S5(A) - 3/20``, ``overview not shown`` below 18/20 on S4 or S5, the P8 census and
the P9 click probe as voids).

Run through ``/launch`` on the 5090 box (the eval step of ``scripts/locked/t0151_overview_frame_ab.sh``)::

    T0151_RUN_DIR=logs/t0151-overview-frame-ab-<sha7> uv run pytest tests/eval/test_t0151_ab.py -q -m gpu -rA

Every run row skips when ``T0151_RUN_DIR`` is unset or the directory has no ``report.json``. They spend no GPU time but
are ``gpu`` because the data exists only after a ``/launch``-reviewed run.

**The oracle is this module, not the harness** (B7: ported from ``tests/eval/test_t0150_ab.py``, never imported from
it nor from any harness). It reuses the other *eval oracles* (T-0111g's ``succeeded``/``accepted``/``voiced_text``/
``dist``/``cheapest_deal``, T-0131's ``derive_arm``/``probe_findings``, T-0132's ``wilson``/``newcombe``/
``first_failing``, T-0134's ``score_v2``/``over_click``), never a script under ``scripts/`` and never T-0150's or
T-0140's eval (R-b: those files are the records of completed runs). ``tests/unit/test_t0151_ab_harness.py`` runs the
pure rows against this module too, so a wrong oracle is caught on CPU.

**Record fields read** (the harness surface is pinned in ``tests/unit/test_t0151_ab_harness.py``'s docstring):

- every trial, step and bodies line carries ``arm`` (``"A"``/``"B"``), ``task`` and ``repeat``;
- a trial also carries T-0111g's/T-0131's/T-0134's fields (``state``, ``voiced``, ``scanned_requests``,
  ``budgets_sent``, ``tool_loop_end``, ``timed_out``, ``session_id``, ``engine_up_after``, ...), its arm's settings as
  the harness passed them (``visual_tokens``, ``reground``, ``live_feedback``, ``echo``, ``recover_thought_calls``,
  ``degeneracy_guard``, ``page_overview`` and ``auto_overview`` (False in both arms), ``overview_frame``), the kit's
  ``tools_version`` (full sha256; v1 in both arms), and the turn line's keys as the session logged them (each ``None``
  when absent): ``overview_frame_logged`` (T-0151 A2-8's ``extras["overview_frame"]``, ``{"shown", "fits",
  "too_tall", "failed", "refused"}``), ``auto_overview_logged`` (T-0150 A5's key) and ``page_overview_logged``
  (T-0140 A10's; neither ever present here), ``degeneracy_guard_logged``, ``live_feedback_logged``,
  ``recover_thought_calls_logged`` and ``visual_tokens_logged`` (as T-0150's eval);
- a step also carries T-0131's fields (``http_status``, ``retries``, ``status``, ``reason``, ``call_name``,
  ``executed_name``, ``first_commit_ms``, ``done_ms``, ``voiced_sources``, ...) and four T-0151 fields:
  ``overview_shown`` (its request's ``llm.request`` record carried A2-8's ``overview_frame`` key), ``frame_kind`` (the
  ``frame_kind`` of its call's last ``tool.call`` record, A2-8: ``"overview"`` or ``None``), ``scrolled_px`` (an
  executed call's ``resolved["scrolled_px"]``, A1-4; ``None`` when absent) and ``cart_after`` (the agent page's
  ``window.__state.cart`` read right after the step's call executed; ``None`` when no call executed or the read
  failed);
- ``bodies.jsonl``, one line per turn: ``{"arm", "task", "repeat", "requests": [...], "overviews": [...]}``. One
  request per body the meters forwarded, in order: ``{"label", "phase", "step", "tools", "messages"}`` as T-0150's
  (``phase`` the session's ``llm.request`` phase; ``step`` 1 for the turn's generate and its over-long retry, the
  pass-1 step of the call a ``tool_repair`` repairs, ``None`` for a prefill; ``messages`` each ``{"role", "parts"}``
  with every part ``{"text": str}`` or ``{"image": sha16}``). One overview per overview-frame capture (the executor
  call whose id ends ``-overview-frame``) that ``overview_frame_of`` registered: ``{"step": 1, "sha"`` (sha16 of the
  frame's JPEG), ``"screens", "width", "height"}``;
- ``manifest.json``: ``live_default`` (``{"visual_tokens", "reground", "live_feedback", "echo",
  "recover_thought_calls", "page_overview", "auto_overview", "overview_frame"}`` as the harness read it from ``cli`` at
  run time), ``regime.arms`` (``arm_rows`` of it), ``arm_table_sha256``, ``task_table_sha256``, ``order``;
- ``probes.json``: T-0131's ``p1``-``p3``, ``p6`` (T-0150's shape), ``p7``, ``p8`` and ``p9`` (shapes in ``p7_ok``/
  ``p8_ok``/``p9_ok``), each with ``passed``;
- ``report.json``: ``{"void", "stopped", "probes", "arms": {"A", "B"}, "differences", "configuration_findings",
  "decision": {"label", "reason", "inputs"}}``.

**Readings** (test-author, reported; the unit module's docstring has the full list):

- *Engine error* and *forced failure* exactly as T-0150's eval (B-1, O-B-8 kept by B1).
- B4's placement: an overview-frame image is in place only in the final user message of the turn's step-1
  ``generate`` request (its over-long retry is a ``generate`` request of step 1 too) or of the ``tool_repair`` of the
  step-1 call, as **the only image of that message**, immediately followed by the exact ``OVERVIEW_FRAME_LABEL`` of its
  screens; that message carries no T-0134 label (the stem ``This screenshot is the browser screen now``). History
  images in earlier messages are allowed (H-T151-3). Overviews are matched on (sha, step), never on the sha alone. An
  overview-frame label text in any other *user* message part is a finding; so is T-0140/T-0150's overview label stem
  in any user message part; assistant text is never judged (O-17).
- O-B-2, tightened by A2-4 ("the over-long retry and the repair of step 1's call carry the same overview and
  label"): on an arm-B trial with a turn line, ``shown`` 0 while a step-1 body carries the overview frame, and ``shown``
  1 while **any** step-1 body (the generate, its retry, the step-1 repair) does not, are findings.
- B4's unlogged absence as T-0150's: an arm-B S4/S5 trial with a turn line whose five ``overview_frame`` counts are all
  0 while one of its ``generate`` bodies carries an image in its final user message and no overview.
- A well-formed ``overview_frame`` turn-line key has exactly A2-8's five keys, non-negative int (not bool) values and a
  sum of at most 1 ("the counts sum to 1 on a turn that sent step 1 and to 0 otherwise").
- B3's mechanism counts (S4 and S5): a call is "on the overview frame" when its step line has ``step`` 1 and
  ``frame_kind`` ``"overview"``; it "scrolled" when it executed with a non-zero ``scrolled_px``; "B9 refusals" are
  those refused with ``FRESHNESS_REASON``; "the tea towel right after step 1" is an S5 step-1 line whose
  ``cart_after`` holds the cheapest deal; a "step-1 ``scroll`` call" is a step-1 line whose ``call_name`` is
  ``scroll``.
- P9 judges the cart and the status only (B2: "must each add exactly the cotton tea towel"); the recorded action
  string and ``scroll_y_after`` are report-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any

import pytest

from tests.eval import test_live_smoke as g0111
from tests.eval import test_t0131_ab as t0131
from tests.eval import test_t0132_ab as t0132
from tests.eval import test_t0134_ab as t0134

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0151_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]

LABELS = ("A", "B")
TASK_IDS = ("S1", "S2", "S3", "S4", "S5")
#: P8 / docs/tasks/T-0151.md "Measured scope": the tasks whose start page fits (no overview) and those on deals.html
FIT_TASKS = ("S1", "S2", "S3")
SHOWN_TASKS = ("S4", "S5")
REPEATS = 20
TURNS_PER_ARM = 100
S4_TURNS = 20
S5_TURNS = 20
TOTAL_TURNS = 200
#: the budget every body carries (the live default at HEAD; B4: a non-1120 body voids the run)
MAIN = 1120
EPS = 1e-9
TASK_TABLE_SHA256 = t0131.TASK_TABLE_SHA256
#: ``tools_v1.json``, the tool set of both arms (B1; a full literal, never derived)
V1_SHA256 = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
#: B1: the live-default settings both arms carry (read from ``cli`` by the harness at run time)
DEFAULT_KEYS = ("visual_tokens", "reground", "live_feedback", "echo", "recover_thought_calls")
#: B1: the three overview settings, off in HEAD's live default
OVERVIEW_KEYS = ("page_overview", "auto_overview", "overview_frame")
#: B1: what each arm sets ``overview_frame`` to (the only difference between the arms)
OVERVIEW_FRAME_ON = {"A": False, "B": True}
#: B5 (H-T151-1, approved 2026-10-24): T-0150's margins plus the S5 guard
S4_MARGIN = 5
SUCCESS_MARGIN = 3
FALSE_CLAIM_MARGIN = 2
OVER_CLICK_MARGIN = 2
S5_MARGIN = 3
ENGINE_ERRORS_MAX = 2
#: B5: ``overview not shown`` when arm B's S4 or S5 turns with ``shown`` = 1 are fewer than this
NOT_SHOWN_BELOW = 18
#: B5's labels
OVERVIEW_FRAME = "overview frame"
NOT_SHOWN = "overview not shown"
NO_CHANGE = "no change"
INCONCLUSIVE = "inconclusive"
VOID = "void"
INCOMPLETE = "incomplete"
CLAUSES = ("acceptance", "safety", "engine")
#: B4 (``--limit-mm-per-prompt image=4``)
MAX_IMAGES = 4
#: B2 P7: T-0132 P4's limit, the serving window less one canvas (16384 - 256)
WINDOW_LIMIT = 16_128
#: B2 P7: the step-1 shape with the overview frame -- one image, the overview (it replaces the live frame)
P7_IMAGES = 1
#: B2 P6/P8/P9
VIEWPORT = (1280, 800)
P6_MIN_SCREENS = 1.3
#: T-0134's hostile fixture (2800 CSS px) and T-0134 P5's tolerance of 1 CSS px, by hand
HOSTILE_HEIGHT_PX = 2800
HOSTILE_TOLERANCE_PX = 1.0
#: T-0140 A3's strings and constants, typed from docs/tasks/T-0140.md
MAX_OVERVIEW_SCREENS = 4
OVERVIEW_FITS = "the whole page fits on the browser screen; no overview was taken"
OVERVIEW_FITS_RATIO = 1.05
#: T-0140 A7's label stem (T-0150's auto overview reuses it): no body of this A/B may carry it
OVERVIEW_STEM = "Whole page, scaled to fit one screen"
#: T-0151 A2-3 (H-T151-4) and A2-4 (H-T151-5), typed from docs/tasks/T-0151.md, never imported
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
#: T-0111g B9 and T-0151 A1-4's reasons, typed from the tasks
FRESHNESS_REASON = "the screen changed since this frame; take a new frame"
OVERVIEW_NOT_REACHED = "the page did not scroll to the point; nothing was clicked"
#: B2 P9 (A1-7): the two deals.html clicks in overview units, each on a fresh page, and the item each must add
P9_POINTS = ((160, 933), (160, 935))
P9_ITEM = "cotton tea towel"
#: B3 (report-only; T-0140's reading): a voiced text that mentions the overview
NARRATION = re.compile(r"\boverview\b|\bwhole page\b", re.IGNORECASE)
#: B3: the S4 distractor T-0134/T-0140 thoughts pick (the cheapest of the six deals visible at 1280x800)
MUG = "enamel mug"

wilson = t0132.wilson
newcombe = t0132.newcombe
score_v2 = t0134.score_v2
over_click = t0134.over_click


def overview_action(screens: float, clipped: bool) -> str:
    """T-0140 A3's executed action string."""
    c = f", top {MAX_OVERVIEW_SCREENS:.1f} screens shown" if clipped else ""
    return f"showed the whole page (about {screens:.1f} screens tall{c}) as a picture before the screen"


def overview_frame_label(screens: float) -> str:
    """T-0151 A2-4's ``OVERVIEW_FRAME_LABEL`` of ``screens``."""
    return OVERVIEW_FRAME_LABEL.format(screens=screens)


# --------------------------------------------------------------------------------------------- B1


def live_default_findings(default: Mapping[str, Any]) -> list[str]:
    """B1: the recorded live default is well formed, with the page overview, the auto overview and the overview frame
    all off (a ``cli`` default with any of them on is refused)."""
    keys = (*DEFAULT_KEYS, *OVERVIEW_KEYS)
    out = [f"live_default lacks {k}" for k in keys if k not in default]
    if out:
        return out
    if default["visual_tokens"] not in (280, 560, 1120):
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
    if default["echo"] not in ("pixels", "model"):
        out.append(f"echo {default['echo']!r}")
    if not isinstance(default["recover_thought_calls"], bool):
        out.append(f"recover_thought_calls {default['recover_thought_calls']!r}")
    for k in OVERVIEW_KEYS:
        if default[k] is not False:
            out.append(f"{k} {default[k]!r}: on by default (B1 runs HEAD's default with it off)")
    return out


def arm_rows(default: Mapping[str, Any]) -> list[dict[str, Any]]:
    """B1: both arms at the live default with the guard on, the page and auto overviews off and tools v1; A without the
    overview frame, B with it."""
    shared = {k: default[k] for k in DEFAULT_KEYS}
    return [
        {
            "label": label,
            **shared,
            "degeneracy_guard": True,
            "page_overview": False,
            "auto_overview": False,
            "overview_frame": OVERVIEW_FRAME_ON[label],
            "tools_sha256": V1_SHA256,
        }
        for label in LABELS
    ]


def arm_table_sha(arms: Sequence[Mapping[str, Any]]) -> str:
    """sha256 of ``json.dumps(list(arms), sort_keys=True, separators=(",", ":"))``."""
    return hashlib.sha256(
        json.dumps([dict(a) for a in arms], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def order(seed: int) -> list[tuple[str, str, int]]:
    """The 200 ``(arm, task, repeat)`` turns, arm-major, then one shuffle of the whole list (jointly)."""
    turns = [(a, t, r) for a in LABELS for t in TASK_IDS for r in range(REPEATS)]
    random.Random(seed).shuffle(turns)
    return turns


# --------------------------------------------------------------------------------------------- B2


def p6_ok(p6: Mapping[str, Any]) -> bool:
    """B2 P6 (as T-0150, T-0140 P6's rule): at 1280x800, ``deals`` showed an overview of at least 1.3 screens within the
    viewport, with A3's action string and ``scroll_y`` unchanged; ``fits`` gave the fits string; ``hostile`` reported
    T-0134's fixture's true height (2800 CSS px, within 1 px) and A3's string."""
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
            and abs(hs * vh - HOSTILE_HEIGHT_PX) <= HOSTILE_TOLERANCE_PX + EPS
            and h["clipped"] is False
            and h["action"] == overview_action(hs, False)
        )
        return bool(deals and fits and hostile)
    except (KeyError, TypeError, ValueError):
        return False


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def p7_ok(p7: Mapping[str, Any]) -> bool:
    """B2 P7 from its record, by T-0132 P4's rule: a ``/tokenize`` count of an S4-like *step-1* request with the overview
    frame (exactly 1 image, the overview: it replaces the live frame) within ``WINDOW_LIMIT`` (16128), the generate on it
    answering 200 and the engine healthy after. ``overview_tokens`` is report-only."""
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
    """B2 P8 (the start-page census through the real executor, browser only): at 1280x800, exactly S4 and S5 ``shown``
    (more than 1.05 and at most ``OVERVIEW_FRAME_MAX_SCREENS`` screens), S1, S2 and S3 ``fits`` -- every task present,
    none other, none ``too_tall``. A missing part fails, never raises."""
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
    """B2 P9 (browser only, no inference; A1-7): at 1280x800, the two deals.html clicks on the registered overview frame,
    ``click_at(160, 933)`` and ``click_at(160, 935)``, each on a fresh page through the real executor, are each
    ``executed`` and leave the cart holding exactly ``["cotton tea towel"]``. A missing part fails, never raises."""
    try:
        if [int(x) for x in p9["viewport"]] != list(VIEWPORT):
            return False
        clicks = list(p9["clicks"])
        if [tuple(int(v) for v in c["point"]) for c in clicks] != [tuple(p) for p in P9_POINTS]:
            return False
        return all(c["status"] == "executed" and list(c["cart"]) == [P9_ITEM] for c in clicks)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def probe_verdicts(probes: Mapping[str, Any]) -> dict[str, bool]:
    """B2, re-derived from ``probes.json`` (never from a recorded ``passed``): P1 by T-0131's ``probe_findings`` and the
    recorded deltas; P2 an HTTP 200 with the engine healthy after; P3 an HTTP 200; P6 ``p6_ok``; P7 ``p7_ok``; P8
    ``p8_ok``; P9 ``p9_ok``; ``void`` iff any failed."""
    p1 = t0131.probe_findings(probes) == [] and bool((probes.get("p1") or {}).get("ok"))
    if p1:
        p1 = all(
            t0131._p1_delta_ok(int(b), row.get(which))
            for b, row in probes["p1"]["budgets"].items()
            for which in ("frame_delta", "crop_delta")
        )
    p2 = (probes.get("p2") or {}).get("http_status") == 200 and (probes.get("p2") or {}).get(
        "healthy_after"
    ) is True
    p3 = (probes.get("p3") or {}).get("http_status") == 200
    p6 = p6_ok(probes.get("p6") or {})
    p7 = p7_ok(probes.get("p7") or {})
    p8 = p8_ok(probes.get("p8") or {})
    p9 = p9_ok(probes.get("p9") or {})
    return {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p6": p6,
        "p7": p7,
        "p8": p8,
        "p9": p9,
        "void": not (p1 and p2 and p3 and p6 and p7 and p8 and p9),
    }


# --------------------------------------------------------------------------------------------- B4


def _key(r: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(r["arm"]), str(r["task"]), int(r["repeat"]))


def _tid(r: Mapping[str, Any]) -> str:
    return f"{r['arm']} {r['task']}#{r['repeat']}"


def _is_step1_body(request: Mapping[str, Any]) -> bool:
    """A2-4: the step-1 request as sent -- the ``generate`` (and its over-long retry) or the repair of step 1's
    call."""
    return request.get("phase") in ("generate", "tool_repair") and request.get("step") == 1


def _placed(
    request: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    last_user: int | None,
    i: int,
    j: int,
    overview: Mapping[str, Any],
) -> bool:
    """B4/A2-4: the overview-frame image at ``messages[i].parts[j]`` is where ``overview`` (one record, the turn's step-1
    capture) puts it: in the final user message of a step-1 body, as that message's only image, immediately followed by
    the exact label of its screens."""
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


def _body_findings(t: Mapping[str, Any], rec: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    tid, arm, task = _tid(t), str(t["arm"]), str(t["task"])
    overviews: dict[str, list[Mapping[str, Any]]] = {}
    for o in rec.get("overviews") or []:
        overviews.setdefault(str(o["sha"]), []).append(o)
    if arm == "A" and overviews:
        out.append(f"{tid}: an arm-A turn registered an overview frame")
    step1_carries: list[bool] = []  # per step-1 body: carries the overview frame, in place
    step1_any: list[bool] = []  # per step-1 body: carries an overview-frame image anywhere
    generate_frames: list[bool] = []  # per generate body: a current frame and no overview image
    for r in rec.get("requests") or []:
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
            elif any(_placed(r, messages, last_user, i, j, o) for o in overviews[sha]):
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
    logged = t.get("overview_frame_logged")
    if arm == "B" and not wrote_no_turn_line(t) and _a2_well_formed(logged):
        # O-B-2 tightened by A2-4: the turn line's ``shown`` and every step-1 body agree
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
        and not wrote_no_turn_line(t)
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
    """B4, one string per finding, each naming the trial (``"<arm> <task>#<repeat>"``) or the setting. The arm rows are
    judged as B1 writes them; the trials only when they forwarded a body (non-empty ``scanned_requests``; O-11); every
    reground line is a finding (ReGround is off in both arms)."""
    del steps  # B4 reads the trials, the bodies and the reground lines
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
        bad = [b for b in t.get("budgets_sent") or [] if b != MAIN]
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


def _a2_well_formed(logged: Any) -> bool:
    return (
        isinstance(logged, Mapping)
        and set(logged) == set(A2_KEYS)
        and all(_int(v) and v >= 0 for v in logged.values())
        and sum(logged.values()) <= 1
    )


def logged_setting_findings(t: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    """B4 "the arms' feedback, echo, recovery or guard settings off their rows", "an arm-A ``overview_frame`` key, or an
    arm-B line without one" and "any ``auto_overview`` or ``page_overview`` key", judged on what the session logged on
    the turn line (T-0140 review F3). A trial that wrote no turn line (T-0135b A7) has nothing to judge."""
    if wrote_no_turn_line(t):
        return []
    tid = _tid(t)
    lf = row.get("live_feedback")
    want = {
        "live_feedback_logged": None if lf is None else {**dict(lf), "echo": row.get("echo")},
        "recover_thought_calls_logged": row.get("recover_thought_calls"),
        "visual_tokens_logged": row.get("visual_tokens"),
    }
    out = [f"{tid}: {k} {t.get(k)!r}, its row says {v!r}" for k, v in want.items() if t.get(k) != v]
    guard = t.get("degeneracy_guard_logged")
    if not (isinstance(guard, Mapping) and guard.get("on") is True):
        out.append(f"{tid}: degeneracy_guard_logged {guard!r}")
    of = t.get("overview_frame_logged")
    on = OVERVIEW_FRAME_ON.get(str(t["arm"]))
    if on is False and of is not None:
        out.append(f"{tid}: overview_frame_logged {of!r} on an arm-A turn line")
    if on is True and not _a2_well_formed(of):
        out.append(f"{tid}: overview_frame_logged {of!r} on an arm-B turn line")
    for k in ("auto_overview_logged", "page_overview_logged"):
        if t.get(k) is not None:
            out.append(f"{tid}: {k} {t.get(k)!r} on a turn line")
    return out


def wrote_no_turn_line(t: Mapping[str, Any]) -> bool:
    """T-0135b A7: a trial that hit the turn timeout (``timed_out`` true) or never had a session (``session_id`` ``None``)
    wrote no turn line, so it has no turn-line key to log."""
    return t.get("timed_out") is True or t.get("session_id") is None


def arm_setting_findings(trials: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """B1/A2-8 evidence: ``(findings, exempted)``. Every trial that scanned requests carries ``page_overview`` and
    ``auto_overview`` False, its arm's ``overview_frame``, ``degeneracy_guard`` True and tools v1; its turn line logged
    A2-8's five counts in B and no ``overview_frame`` key in A, no ``page_overview`` or ``auto_overview`` key in either,
    and the guard on -- except on a trial that wrote no turn line, listed in ``exempted`` (T-0135b A7)."""
    findings: list[str] = []
    exempted: list[str] = []
    for t in trials:
        if not t.get("scanned_requests"):
            continue
        arm, tid = str(t["arm"]), _tid(t)
        for k in ("page_overview", "auto_overview"):
            if t.get(k) is not False:
                findings.append(f"{tid}: {k} {t.get(k)!r}")
        if t.get("overview_frame") is not OVERVIEW_FRAME_ON[arm]:
            findings.append(f"{tid}: overview_frame {t.get('overview_frame')!r}")
        if t.get("degeneracy_guard") is not True:
            findings.append(f"{tid}: degeneracy_guard {t.get('degeneracy_guard')!r}")
        if t.get("tools_version") != V1_SHA256:
            findings.append(f"{tid}: tools_version {t.get('tools_version')!r}")
        if wrote_no_turn_line(t):
            exempted.append(tid)
            continue
        for k in ("page_overview_logged", "auto_overview_logged"):
            if t.get(k) is not None:
                findings.append(f"{tid}: {k} {t.get(k)!r}")
        logged = t.get("overview_frame_logged")
        if arm == "A" and logged is not None:
            findings.append(f"{tid}: overview_frame_logged {logged!r} on an arm-A turn line")
        if arm == "B" and not _a2_well_formed(logged):
            findings.append(f"{tid}: overview_frame_logged {logged!r}")
        guard = t.get("degeneracy_guard_logged")
        if not (isinstance(guard, Mapping) and guard.get("on") is True):
            findings.append(f"{tid}: degeneracy_guard_logged {guard!r}")
    return findings, exempted


# --------------------------------------------------------------------------------------------- B3


def is_degenerate(t: Mapping[str, Any]) -> bool:
    return t.get("tool_loop_end") == "degenerate"


#: O-B-8: the over-long statuses the session's retry recovers from (T-0020)
OVER_LONG = (400, 413)


def failed_attempts(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """A step's attempts (retries, then its own) that did not answer 200, except an over-long 400/413 that a later
    attempt of the same step recovered with a 200 (O-B-8: an over-long prompt, not an engine error)."""
    atts = g0111.attempts(step)
    return [
        a
        for k, a in enumerate(atts)
        if a.get("http_status") != 200
        and not (
            a.get("http_status") in OVER_LONG
            and any(b.get("http_status") == 200 for b in atts[k + 1 :])
        )
    ]


def engine_error(t: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """B5 (B-1 kept, O-B-8): a timeout, the engine down after, or any non-200 attempt (5xx, a connection failure, HTTP 0
    from an abort other than the guard's) -- never a ``degenerate`` end, never a recovered over-long retry."""
    if is_degenerate(t):
        return False
    if t.get("timed_out") is True or t.get("engine_up_after") is False:
        return True
    return any(failed_attempts(s) for s in steps)


def accepted(t: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """T-0111g's per-trial acceptance (``g0111.accepted``) with O-B-8's reading: a recovered over-long attempt is not a
    failed attempt (test-author's consequence of O-B-8, as T-0150's)."""
    if t.get("exception") or t.get("timed_out") or t.get("log_complete") is False:
        return False
    return bool(steps) and not any(failed_attempts(s) for s in steps)


def forced_failure(t: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """B5: an engine-error or ``degenerate`` trial counts as a failed trial in every metric."""
    return is_degenerate(t) or engine_error(t, steps)


def _turn(r: Mapping[str, Any]) -> tuple[str, int]:
    return (str(r["task"]), int(r["repeat"]))


def _false_claim(t: Mapping[str, Any], steps: Sequence[Mapping[str, Any]], ok: bool) -> bool:
    """T-0131's proxy with ``ok`` (``score_v2``, False on a forced failure) as the success test."""
    if t.get("exception") or t.get("timed_out") or not steps:
        return False
    if ok:
        return False
    last = max(steps, key=lambda s: int(s["step"]))
    if last.get("status") is not None or last.get("call_name") is not None:
        return False
    said = " ".join(str(v) for v in last.get("voiced_sources") or [])
    return bool(said.strip()) and "?" not in said


def _said(t: Mapping[str, Any]) -> str:
    return " ".join(g0111.voiced_text(t).split()).casefold()


def _ci(k: int, n: int) -> list[float] | None:
    return list(wilson(k, n)) if n else None


def _on_overview(s: Mapping[str, Any]) -> bool:
    """B3: a step-1 call read off the overview frame (A2-8's ``frame_kind``)."""
    return (
        int(s["step"]) == 1 and s.get("frame_kind") == "overview" and s.get("call_name") is not None
    )


def mechanism_counts(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """B3's new mechanism counts on S4 and S5 (the readings in this module's docstring): step-1 calls on the overview
    frame by name, their statuses, those that scrolled, the S5 turns whose cart holds the tea towel right after step 1,
    step-1 ``scroll`` calls, B9 refusals of overview calls and ``OVERVIEW_NOT_REACHED`` failures."""
    cheapest = g0111.cheapest_deal()[0].casefold()
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


def derive_arm(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """B3's counts for one arm's records (a turn counts once in every turn count)."""
    by = {_turn(t): [s for s in steps if _turn(s) == _turn(t)] for t in trials}
    forced = {_turn(t): forced_failure(t, by[_turn(t)]) for t in trials}
    ok = {
        _turn(t): (not forced[_turn(t)])
        and score_v2(str(t["task"]), t.get("state"), g0111.voiced_text(t))
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
            "first_commit_ms": g0111.dist(
                [s["first_commit_ms"] for s in mine if s.get("first_commit_ms") is not None]
            ),
            "done_ms": g0111.dist([s["done_ms"] for s in mine if s.get("done_ms") is not None]),
        }

    cheapest = g0111.cheapest_deal()[0].casefold()
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
        "s4_named_cheapest": sum(1 for t in s4 if cheapest in _said(t)),
        "s4_named_mug": sum(1 for t in s4 if MUG in _said(t)),
        "narration": sum(1 for t in trials if NARRATION.search(g0111.voiced_text(t))),
        "degenerate": sum(1 for t in trials if is_degenerate(t)),
        "engine_errors": sum(1 for t in trials if engine_error(t, by[_turn(t)])),
        "latency": {"shown": split(True), "not_shown": split(False)},
        **mechanism_counts(trials, steps),
    }


#: T-0131's per-arm sections the harness carries over (``success``/``false_claims`` renamed, as T-0134's)
T0131_SECTIONS = t0134.T0131_SECTIONS


def t0131_section(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """One arm's T-0131 section (steps, latency, query tokens and the prefix-cache hit, wall time, VRAM, the engine
    flag, every safety count) by T-0131's oracle at 1120 with ReGround off, with T-0134's two renames and T-0131's step
    latency as ``latency_v1``: ``latency`` is B3's step-1 shown/not-shown split."""
    sec = dict(t0131.derive_arm(MAIN, False, trials, steps, regrounds))
    sec["success_v1"] = sec.pop("success")
    sec["false_claims_v1"] = sec.pop("false_claims")
    sec["latency_v1"] = sec.pop("latency")
    return sec


def decision_inputs(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """B5's inputs for one arm, re-derived: acceptance (T-0111g's per trial, over the trials that are neither an engine
    error nor ``degenerate``), the safety sum (T-0131's), the engine errors, B3's counts, the S4 and S5 turns with
    ``shown`` = 1, and per clause the first failing trial ``"<task>#<repeat>"``."""
    by = {_turn(t): [s for s in steps if _turn(s) == _turn(t)] for t in trials}
    m = derive_arm(trials, steps)
    plain = [t for t in trials if not forced_failure(t, by[_turn(t)])]
    failing = [t for t in plain if not accepted(t, by[_turn(t)])]
    errors = [t for t in trials if engine_error(t, by[_turn(t)])]
    sec = t0131.derive_arm(MAIN, False, trials, steps, regrounds)

    def tid(t: Mapping[str, Any]) -> str:
        return f"{t['task']}#{int(t['repeat'])}"

    return {
        "accepted": not failing,
        "safety": sum(int(v) for v in sec["safety"].values()),
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
            "safety": t0132.first_failing(trials, steps, regrounds)["safety"],
            "engine": tid(errors[0]) if errors else None,
        },
    }


# --------------------------------------------------------------------------------------------- B5


def _ge(a: float, b: float) -> bool:
    return float(a) >= float(b) - EPS


def _le(a: float, b: float) -> bool:
    return float(a) <= float(b) + EPS


def _first_ineligible(inputs: Mapping[str, Mapping[str, Any]]) -> tuple[str, str] | None:
    for arm in LABELS:
        i = inputs[arm]
        ok = {
            "acceptance": bool(i["accepted"]),
            "safety": int(i["safety"]) == 0,
            "engine": int(i["engine_errors"]) <= ENGINE_ERRORS_MAX,
        }
        for clause in CLAUSES:
            if not ok[clause]:
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
    """B5: P1 failed -> inconclusive; a configuration finding -> void (``configuration: <first>``); P2/P3/P6/P7/P8/P9
    failed (``void``) -> void; stopped -> incomplete; an arm ineligible (acceptance, safety, more than 2 engine errors) ->
    inconclusive naming the arm, the clause and the first failing trial; then ``overview frame`` iff every adoption
    clause holds (the S5 guard among them); else ``overview not shown`` when arm B's S4 or S5 turns with ``shown`` = 1
    are under 18, else ``no change``."""
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


def decide(inputs: Mapping[str, Mapping[str, Any]], **kw: Any) -> str:
    return decide_with_reason(inputs, **kw)[0]


def differences(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
    """B3: Newcombe 95 % of B-A on S4 (of 20), on success (of 100) and on S5 (of 20), at B3's fixed denominators."""
    a, b = arms["A"], arms["B"]
    return {
        "s4 B-A": newcombe(b["s4"], S4_TURNS, a["s4"], S4_TURNS),
        "success B-A": newcombe(b["success"], TURNS_PER_ARM, a["success"], TURNS_PER_ARM),
        "s5 B-A": newcombe(b["success_s5"], S5_TURNS, a["success_s5"], S5_TURNS),
    }


# --------------------------------------------------------------------------------------------- B5, stated numbers


def s5_guard_false_reject(p: Fraction, margin: int, n: int = S5_TURNS) -> Fraction:
    """B5 "stated in the report": the S5 guard's false-reject rate under a true null, exactly -- the probability that
    ``S5(B) < S5(A) - margin`` for two independent Bin(n, p)."""
    pmf = [comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
    return sum(
        (pmf[a] * pmf[b] for a in range(n + 1) for b in range(n + 1) if b < a - margin),
        Fraction(0),
    )


def fisher_two_sided(a: int, n1: int, b: int, n2: int) -> Fraction:
    """Fisher's exact test, two-sided (the sum of the tables no more likely than the observed one), exactly."""
    k, n = a + b, n1 + n2

    def h(x: int) -> Fraction:
        return Fraction(comb(n1, x) * comb(n2, k - x), comb(n, k))

    obs = h(a)
    return sum((h(x) for x in range(max(0, k - n2), min(k, n1) + 1) if h(x) <= obs), Fraction(0))


# --------------------------------------------------------------------------------------------- the run (gpu)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _of(rows: Sequence[Mapping[str, Any]], arm: str) -> list[Mapping[str, Any]]:
    return [r for r in rows if r.get("arm") == arm]


def _manifest_arms(run: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(((run.get("manifest") or {}).get("regime") or {}).get("arms") or [])


@pytest.fixture(scope="module")
def run_dir() -> Path:
    where = os.environ.get(ENV)
    if not where:
        pytest.skip(f"{ENV} is unset: the T-0151 A/B has not run (docs/tasks/T-0151.md B6)")
    d = Path(where)
    if not (d / "report.json").is_file():
        pytest.skip(f"{d}/report.json does not exist")
    return d


@pytest.fixture(scope="module")
def run(run_dir: Path) -> dict[str, Any]:
    return {
        "report": json.loads((run_dir / "report.json").read_text()),
        "manifest": json.loads((run_dir / "manifest.json").read_text()),
        "probes": json.loads((run_dir / "probes.json").read_text()),
        "trials": _jsonl(run_dir / "trials.jsonl"),
        "steps": _jsonl(run_dir / "steps.jsonl"),
        "regrounds": _jsonl(run_dir / "regrounds.jsonl"),
        "bodies": _jsonl(run_dir / "bodies.jsonl"),
    }


def _inputs(run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        arm: decision_inputs(
            _of(run["trials"], arm), _of(run["steps"], arm), _of(run["regrounds"], arm)
        )
        for arm in LABELS
    }


def _findings(run: Mapping[str, Any]) -> list[str]:
    return configuration_findings(
        run["trials"], run["steps"], run["regrounds"], run["bodies"], _manifest_arms(run)
    )


def test_the_run_is_complete(run: dict[str, Any]) -> None:
    """B7 completeness: the manifest's arm rows are B1's two rows over its recorded live default (tools v1 in both), its
    arm-table sha is theirs, the task-table sha matches its pin, the manifest's order is the 200-turn seed-0 joint order
    and the trials ran in it."""
    manifest = run["manifest"]
    want_rows = arm_rows(manifest.get("live_default") or {})
    assert _manifest_arms(run) == want_rows, (_manifest_arms(run), want_rows)
    assert manifest.get("arm_table_sha256") == arm_table_sha(want_rows), manifest.get(
        "arm_table_sha256"
    )
    assert manifest.get("task_table_sha256") == TASK_TABLE_SHA256
    assert [tuple(x) for x in manifest.get("order") or []] == order(0), "B1: the manifest's order"
    got = [(t["arm"], t["task"], int(t["repeat"])) for t in run["trials"]]
    if run["report"].get("stopped"):
        assert got == order(0)[: len(got)], "B1: a stopped run is a prefix of the seed-0 order"
        pytest.skip(f"the run stopped early: {run['report']['stopped']}")
    assert got == order(0), "B1: the trials ran in the seed-0 joint order"


def test_stopped_re_derived(run: dict[str, Any]) -> None:
    """B7: ``stopped`` is set exactly when fewer than 200 trials were written."""
    assert bool(run["report"].get("stopped")) == (len(run["trials"]) < TOTAL_TURNS)


def test_live_default_in_both_arms(run: dict[str, Any]) -> None:
    """B1: the manifest records a well-formed live default with all three overviews off, and every trial that ran a
    session ran at it."""
    recorded = run["manifest"].get("live_default") or {}
    assert live_default_findings(recorded) == [], (recorded, live_default_findings(recorded))
    for t in run["trials"]:
        if t.get("scanned_requests"):
            got = {k: t.get(k) for k in DEFAULT_KEYS}
            assert got == {k: recorded[k] for k in DEFAULT_KEYS}, (_tid(t), got, recorded)


def test_arm_setting_reached_every_turn(run: dict[str, Any]) -> None:
    """B1/A2-8 evidence (``arm_setting_findings``): each arm's overview frame, guard and tool set reached every turn that
    sent a body; a turn that wrote no turn line is exempted from the turn-line keys only (T-0135b A7)."""
    findings, exempted = arm_setting_findings(run["trials"])
    assert findings == [], findings
    if exempted:
        print(f"exempted (no turn line): {exempted}")


@pytest.mark.parametrize("arm", LABELS)
def test_arm_metrics_re_derived(run: dict[str, Any], arm: str) -> None:
    """B3: every per-arm count re-derived from the jsonl files (the mechanism counts included) equals report.json's."""
    want = derive_arm(_of(run["trials"], arm), _of(run["steps"], arm))
    got = run["report"]["arms"][arm]
    for key, value in want.items():
        if key.endswith("_ci") and (value is None or got[key] is None):
            assert got[key] is value is None, (key, got[key], value)
        elif key.endswith("_ci"):
            assert all(abs(x - y) < 1e-6 for x, y in zip(got[key], value, strict=True)), (
                key,
                got[key],
                value,
            )
        else:
            assert got[key] == value, (arm, key, got[key], value)


@pytest.mark.parametrize("arm", LABELS)
def test_t0131_sections_re_derived(run: dict[str, Any], arm: str) -> None:
    """B3 (carried from T-0150 B3): steps per turn, step latency, query tokens, the prefix-cache hit, turn wall time,
    the minimum free VRAM, acceptance, the engine flag and every safety count, re-derived per arm by T-0131's oracle."""
    want = t0131_section(
        _of(run["trials"], arm), _of(run["steps"], arm), _of(run["regrounds"], arm)
    )
    got = run["report"]["arms"][arm]
    keys = [k for k in T0131_SECTIONS if k != "latency"]  # B3 owns ``latency``
    for key in (*keys, "success_v1", "false_claims_v1", "latency_v1"):
        assert got.get(key) == want[key], (arm, key, got.get(key), want[key])


@pytest.mark.parametrize("arm", LABELS)
def test_safety_counts_zero(run: dict[str, Any], arm: str) -> None:
    """B3: every T-0131/T-0132 safety count, re-derived from the records, is 0."""
    want = t0131_section(
        _of(run["trials"], arm), _of(run["steps"], arm), _of(run["regrounds"], arm)
    )
    assert all(int(v) == 0 for v in want["safety"].values()), (arm, want["safety"])


def test_configuration_findings_re_derived(run: dict[str, Any]) -> None:
    """B4: the findings re-derived from the records and the manifest's arm rows equal report.json's."""
    want = _findings(run)
    assert sorted(run["report"]["configuration_findings"]) == sorted(want), (
        run["report"]["configuration_findings"],
        want,
    )


def test_probes_re_derived(run: dict[str, Any]) -> None:
    """B2: P1-P3, P6, P7, P8 and P9 re-derived from probes.json; each recorded ``passed`` and the report's ``void``
    agree, and a failed P1 is labelled ``inconclusive``, a failed P2/P3/P6/P7/P8/P9 ``void``."""
    probes = run["probes"]
    v = probe_verdicts(probes)
    for p in ("p1", "p2", "p3", "p6", "p7", "p8", "p9"):
        assert bool(probes[p].get("passed")) is v[p], (p, probes[p], v[p])
    assert bool(run["report"].get("void")) is v["void"], (run["report"].get("void"), v)
    if not v["p1"]:
        assert run["report"]["decision"]["label"] == INCONCLUSIVE, run["report"]["decision"]
    elif v["void"]:
        assert run["report"]["decision"]["label"] == VOID, run["report"]["decision"]


def test_p8_census(run: dict[str, Any]) -> None:
    """B2 P8: the census pins which tasks receive the overview frame -- S4 and S5 ``shown``, S1-S3 ``fits``, none
    ``too_tall`` -- so the guards mean what B5 says; a run whose census differs is void (checked by the row above), and
    this row says which task moved."""
    p8 = run["probes"].get("p8") or {}
    got = {t: (p8.get("tasks") or {}).get(t, {}).get("outcome") for t in TASK_IDS}
    want = {**dict.fromkeys(FIT_TASKS, "fits"), **dict.fromkeys(SHOWN_TASKS, "shown")}
    assert got == want, (got, want)


def test_p9_clicks(run: dict[str, Any]) -> None:
    """B2 P9: each of the two deals.html clicks on the overview frame added exactly the cotton tea towel; this row says
    which click did not (the void itself is checked by ``test_probes_re_derived``)."""
    p9 = run["probes"].get("p9") or {}
    got = [
        (tuple(c.get("point") or ()), c.get("status"), c.get("cart"))
        for c in p9.get("clicks") or []
    ]
    assert got == [(p, "executed", [P9_ITEM]) for p in P9_POINTS], got


@pytest.mark.parametrize("arm", LABELS)
def test_decision_inputs_re_derived(run: dict[str, Any], arm: str) -> None:
    """B5: each arm's decision inputs re-derived from the records, never read from the report's sections."""
    want = _inputs(run)[arm]
    assert run["report"]["decision"]["inputs"][arm] == want, (
        arm,
        run["report"]["decision"]["inputs"][arm],
        want,
    )


def test_decision_re_derived(run: dict[str, Any]) -> None:
    """B5: the label and reason from re-derived inputs, probes, findings and the trial count."""
    v = probe_verdicts(run["probes"])
    label, reason = decide_with_reason(
        _inputs(run),
        p1_passed=v["p1"],
        void=v["void"],
        stopped=run["report"].get("stopped") if len(run["trials"]) < TOTAL_TURNS else None,
        configuration_findings=_findings(run),
    )
    d = run["report"]["decision"]
    assert (d["label"], d.get("reason")) == (label, reason)


def test_differences_re_derived(run: dict[str, Any]) -> None:
    """B3: the Newcombe intervals for B-A on S4, on success and on S5, from counts re-derived from the records."""
    arms = {arm: derive_arm(_of(run["trials"], arm), _of(run["steps"], arm)) for arm in LABELS}
    got = run["report"]["differences"]
    assert set(got) == set(differences(arms)), got
    for key, (lo, hi) in differences(arms).items():
        assert abs(got[key][0] - lo) < 1e-6 and abs(got[key][1] - hi) < 1e-6, (
            key,
            got[key],
            (lo, hi),
        )
