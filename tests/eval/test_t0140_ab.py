"""T-0140 B2-B5 and B7: the locked 2-arm page-overview A/B on S4, re-derived from the run's records and compared with
``report.json`` (``gpu``, ``eval``; report-only plus the H-T140-1 decision rule, approved as proposed 2026-10-23).

Run through ``/launch`` on the 5090 box (the eval step of ``scripts/locked/t0140_overview_ab.sh``)::

    T0140_RUN_DIR=logs/t0140-overview-ab-<sha7> uv run pytest tests/eval/test_t0140_ab.py -q -m gpu -rA

Every run row skips when ``T0140_RUN_DIR`` is unset or the directory has no ``report.json``. They spend no GPU time but
are ``gpu`` because the data exists only after a ``/launch``-reviewed run.

**The oracle is this module, not the harness** (B7: ported, never imported from the harness). It reuses the other
*eval oracles* (T-0111g's ``succeeded``/``accepted``/``voiced_text``/``dist``, T-0131's ``derive_arm``/
``probe_findings``, T-0132's ``wilson``/``newcombe``/``first_failing``, T-0134's ``score_v2``/``over_click``), never a
script under ``scripts/``. ``tests/unit/test_t0140_ab_harness.py`` runs the pure rows against this module too, so a
wrong oracle is caught on CPU.

**Record fields read** (the harness surface is pinned in ``tests/unit/test_t0140_ab_harness.py``'s docstring):

- every trial, step and bodies line carries ``arm`` (``"A"``/``"B"``), ``task`` and ``repeat``;
- a trial also carries T-0111g's/T-0131's/T-0134's fields (``state``, ``voiced``, ``scanned_requests``,
  ``budgets_sent``, ``tool_loop_end``, ``timed_out``, ``session_id``, ``engine_up_after``, ...), its arm's settings as
  the harness passed them (``visual_tokens``, ``reground``, ``live_feedback``, ``echo``, ``recover_thought_calls``,
  ``degeneracy_guard``, ``page_overview``), the kit's ``tools_version`` (full sha256), and the turn line's keys as the
  session logged them (each ``None`` when absent): ``page_overview_logged`` (A10's ``extras["page_overview"]``),
  ``degeneracy_guard_logged`` (T-0142's ``extras["degeneracy_guard"]``), ``live_feedback_logged`` (T-0134 A5's
  ``extras["live_feedback"]``, ``{"screen_label", "page_extent", "echo"}``, the echo read off the browser),
  ``recover_thought_calls_logged`` (T-0135 A8's ``extras["recover_thought_calls"]``) and ``visual_tokens_logged``
  (T-0131 A3's ``extras["visual_tokens"]``; review F3, 2026-10-24: these are what B4's "settings off their rows" is
  judged on, since the trial's own setting fields are copied from the row);
- a step also carries T-0131's fields (``http_status``, ``retries``, ``status``, ``call_name``, ``executed_name``,
  ``done_ms``, ``voiced_sources``, ...) and ``overview_shown`` (its request's ``llm.request`` record carried A10's
  ``overview`` key);
- ``bodies.jsonl``, one line per turn: ``{"arm", "task", "repeat", "requests": [...], "overviews": [...]}``. One
  request per body the meters forwarded, in order: ``{"label", "phase", "step", "tools", "messages"}``: ``phase`` the
  session's ``llm.request`` phase (``prefill``, ``generate``, ``tool_step``, ``tool_repair``, ``reground``, ...),
  ``step`` the step line's number the request belongs to (1 = the turn's generate; ``None`` for a prefill; for a
  ``tool_repair`` body, the step line of the call it repairs, i.e. of the pass-1 request it continues, review F2),
  ``tools``
  the declared names of the body's ``tools`` field (``None`` when the body has none), ``messages`` each
  ``{"role", "parts"}`` with every content part ``{"text": str}`` or ``{"image": sha16}`` (sha16 = the first 16 hex of
  sha256 of the image's JPEG bytes). One overview per executed ``page_overview`` that kept an image: ``{"step"`` (the
  step line of the call), ``"sha"`` (sha16 of the overview frame's JPEG bytes), ``"screens", "clipped", "width",
  "height"}``;
- ``manifest.json``: ``live_default`` (``{"visual_tokens", "reground", "live_feedback", "echo",
  "recover_thought_calls", "page_overview"}`` as the harness read it from ``cli`` at run time), ``regime.arms``
  (``arm_rows`` of it), ``arm_table_sha256``, ``task_table_sha256``, ``order``;
- ``probes.json``: T-0131's ``p1``-``p3`` and ``p6``/``p7`` (shapes in ``p6_ok``/``p7_ok``), each with ``passed``;
- ``report.json``: ``{"void", "stopped", "probes", "arms": {"A", "B"}, "differences", "configuration_findings",
  "decision": {"label", "reason", "inputs"}}``.

**Readings** (test-author, reported; the unit module's docstring has the full list):

- *Engine error* (B5): a trial that timed out, whose engine was down after it (``engine_up_after`` False), or one of
  whose step attempts (retries included) did not answer HTTP 200 -- unless the trial ended ``degenerate`` (the T-0142
  guard's own abort is not an engine error). B5's relaxation is read through the acceptance clause too: engine-error and
  ``degenerate`` trials are left out of T-0111g's per-trial acceptance (otherwise one timeout would still make the arm
  ineligible on ``acceptance``, which is what the change is for); the engine clause fails only above 2 per arm, naming
  the first engine-error trial.
- *Counts as a failed trial in every metric*: an engine-error or ``degenerate`` trial is never a success (success, per
  task, S4), and T-0131's false-claim proxy judges it as not succeeded; over-clicks are read from the cart as always.
- B4's "right before the current frame": in the final user message of the ``tool_step`` request numbered one past the
  call's step (or of a ``tool_repair`` of that step's call, which continues its pass-1 prompt: review F2), the parts ``[overview image, OVERVIEW_LABEL text of its screens/clipped, the current frame]`` where the
  current frame is that message's last image. Overviews are matched on (sha, step), never on the sha alone: repeated
  calls on an unchanged page give the same sha (review F1). An overview label text in any other *user* message part is a finding;
  assistant text is never judged for it (O-17: the model's own spoken words reach later bodies legitimately).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.eval import test_live_smoke as g0111
from tests.eval import test_t0131_ab as t0131
from tests.eval import test_t0132_ab as t0132
from tests.eval import test_t0134_ab as t0134

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0140_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]

LABELS = ("A", "B")
TASK_IDS = ("S1", "S2", "S3", "S4", "S5")
REPEATS = 20
TURNS_PER_ARM = 100
S4_TURNS = 20
TOTAL_TURNS = 200
#: the budget every body carries (the live default at HEAD; B4: a non-1120 body voids the run)
MAIN = 1120
EPS = 1e-9
TASK_TABLE_SHA256 = t0131.TASK_TABLE_SHA256
#: ``tools_v1.json`` (arm A) and ``tools_v1_overview.json`` (arm B), full literals (A1; never derived)
V1_SHA256 = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
OVERVIEW_SHA256 = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"
#: B1: the live-default settings both arms carry (read from ``cli`` by the harness at run time)
DEFAULT_KEYS = ("visual_tokens", "reground", "live_feedback", "echo", "recover_thought_calls")
#: B5 (H-T140-1, approved as proposed 2026-10-23)
S4_MARGIN = 5
SUCCESS_MARGIN = 3
FALSE_CLAIM_MARGIN = 2
OVER_CLICK_MARGIN = 2
ENGINE_ERRORS_MAX = 2
UNUSED_BELOW = 5
#: B5's labels
OVERVIEW = "overview"
UNUSED = "overview unused"
NO_CHANGE = "no change"
INCONCLUSIVE = "inconclusive"
VOID = "void"
INCOMPLETE = "incomplete"
CLAUSES = ("acceptance", "safety", "engine")
#: B4 (``--limit-mm-per-prompt image=4``)
MAX_IMAGES = 4
#: B2 P7: T-0132 P4's limit, the serving window less one canvas (16384 - 256)
WINDOW_LIMIT = 16_128
#: B2 P6
VIEWPORT = (1280, 800)
P6_MIN_SCREENS = 1.3
#: T-0134's hostile fixture (``HOSTILE_PAGE`` of ``scripts/gpu/t0134_feedback_ab.py``, 2800 CSS px), by hand; T-0134
#: P5's tolerance of 1 CSS px
HOSTILE_HEIGHT_PX = 2800
HOSTILE_TOLERANCE_PX = 1.0
#: A3's strings and constants, typed from docs/tasks/T-0140.md
MAX_OVERVIEW_SCREENS = 4
OVERVIEW_FITS = "the whole page fits on the browser screen; no overview was taken"
#: A7's label stem, and the label's whole shape
OVERVIEW_STEM = "Whole page, scaled to fit one screen"
#: A10's counts on the turn line
A10_KEYS = ("calls", "shown", "fits", "refused", "failed")
#: B3 (report-only; the test-author's reading): a voiced text that mentions the overview
NARRATION = re.compile(r"\boverview\b|\bwhole page\b", re.IGNORECASE)

wilson = t0132.wilson
newcombe = t0132.newcombe
score_v2 = t0134.score_v2
over_click = t0134.over_click


def overview_action(screens: float, clipped: bool) -> str:
    """A3's executed action string."""
    c = f", top {MAX_OVERVIEW_SCREENS:.1f} screens shown" if clipped else ""
    return f"showed the whole page (about {screens:.1f} screens tall{c}) as a picture before the screen"


def overview_label(screens: float, clipped: bool) -> str:
    """A7's ``OVERVIEW_LABEL``."""
    c = f", top {MAX_OVERVIEW_SCREENS:.1f} screens shown" if clipped else ""
    return (
        f"[Whole page, scaled to fit one screen: about {screens:.1f} screens tall{c}. "
        "The next image is the browser screen.]"
    )


# --------------------------------------------------------------------------------------------- B1


def live_default_findings(default: Mapping[str, Any]) -> list[str]:
    """B1: the recorded live default is well formed, and its overview is off (a ``cli`` default with the overview
    already on is refused)."""
    keys = (*DEFAULT_KEYS, "page_overview")
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
    if default["page_overview"] is not False:
        out.append(
            f"page_overview {default['page_overview']!r}: the overview is already the default"
        )
    return out


def arm_rows(default: Mapping[str, Any]) -> list[dict[str, Any]]:
    """B1: both arms at the live default with the guard on; A without the overview on tools v1, B with it on the
    overview set."""
    shared = {k: default[k] for k in DEFAULT_KEYS}
    return [
        {
            "label": "A",
            **shared,
            "degeneracy_guard": True,
            "page_overview": False,
            "tools_sha256": V1_SHA256,
        },
        {
            "label": "B",
            **shared,
            "degeneracy_guard": True,
            "page_overview": True,
            "tools_sha256": OVERVIEW_SHA256,
        },
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
    """B2 P6 from its record: at 1280x800, ``deals`` showed an overview of at least 1.3 screens within the viewport,
    with A3's action string and ``scroll_y`` unchanged; ``fits`` gave the fits string; ``hostile`` reported
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


def p7_ok(p7: Mapping[str, Any]) -> bool:
    """B2 P7 from its record, by T-0132 P4's rule: a ``/tokenize`` count of the step-2 request carrying one
    overview, at most 4 images, within ``WINDOW_LIMIT`` (16128), the generate on it answering 200 and the engine
    healthy after."""
    tokens = p7.get("tokens")
    return (
        isinstance(tokens, int)
        and not isinstance(tokens, bool)
        and p7.get("limit") == WINDOW_LIMIT
        and tokens <= WINDOW_LIMIT
        and p7.get("http_status") == 200
        and p7.get("healthy_after") is True
        and p7.get("overview_images") == 1
        and isinstance(p7.get("images"), int)
        and int(p7["images"]) <= MAX_IMAGES
    )


def probe_verdicts(probes: Mapping[str, Any]) -> dict[str, bool]:
    """B2, re-derived from ``probes.json`` (never from a recorded ``passed``): P1 by T-0131's ``probe_findings`` and
    the recorded deltas; P2 an HTTP 200 with the engine healthy after; P3 an HTTP 200; P6 ``p6_ok``; P7 ``p7_ok``;
    ``void`` iff any failed."""
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
    return {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p6": p6,
        "p7": p7,
        "void": not (p1 and p2 and p3 and p6 and p7),
    }


# --------------------------------------------------------------------------------------------- B4


def _key(r: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(r["arm"]), str(r["task"]), int(r["repeat"]))


def _tid(r: Mapping[str, Any]) -> str:
    return f"{r['arm']} {r['task']}#{r['repeat']}"


def _placed(
    request: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    last_user: int | None,
    i: int,
    j: int,
    overview: Mapping[str, Any],
) -> bool:
    """B4/A7: the overview image at ``messages[i].parts[j]`` is where ``overview`` (one record, so one step) puts it:
    the step request one past the call's step, or a repair of that step's call (it continues the pass-1 prompt,
    review F2)."""
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


def _body_findings(t: Mapping[str, Any], rec: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    tid, arm = _tid(t), str(t["arm"])
    # review F1: several calls on an unchanged page share a sha, so a sha maps to every record that carries it
    overviews: dict[str, list[Mapping[str, Any]]] = {}
    for o in rec.get("overviews") or []:
        overviews.setdefault(str(o["sha"]), []).append(o)
    for r in rec.get("requests") or []:
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
            if any(_placed(r, messages, last_user, i, j, o) for o in overviews[sha]):
                placed.add((i, j + 1))
            else:
                calls = sorted(int(o["step"]) for o in overviews[sha])
                out.append(f"{where}: an overview image out of place (the step-{calls} call)")
        for i, m in enumerate(messages):
            if m.get("role") != "user":
                continue  # O-17: spoken words reach later bodies as assistant text
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
    """B4, one string per finding, each naming the trial (``"<arm> <task>#<repeat>"``) or the setting. The arm rows
    are judged as B1 writes them; the bodies only of trials that forwarded a body (non-empty ``scanned_requests``;
    O-11, as T-0134); every reground line is a finding (ReGround is off in both arms)."""
    del steps  # B4 reads the trials, the bodies and the reground lines
    out: list[str] = []
    rows = [dict(r) for r in arms]
    for k in sorted({k for r in rows for k in r} - {"label", "page_overview", "tools_sha256"}):
        values = [r.get(k) for r in rows]
        if any(v != values[0] for v in values):
            out.append(f"the arms differ in {k}: {values}")
    want = {"A": (False, V1_SHA256), "B": (True, OVERVIEW_SHA256)}
    for r in rows:
        label = str(r.get("label"))
        if label in want and (r.get("page_overview"), r.get("tools_sha256")) != want[label]:
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
        bad = [b for b in t.get("budgets_sent") or [] if b != MAIN]
        if bad or not t.get("budgets_sent"):
            out.append(f"{tid}: body budget {bad or 'none'}")
        for k in ("live_feedback", "echo", "degeneracy_guard"):
            if t.get(k) != row.get(k):
                out.append(f"{tid}: {k} {t.get(k)!r}, its row says {row.get(k)!r}")
        out += logged_setting_findings(t, row)
        out += _body_findings(t, by_bodies.get(_key(t)) or {})
    return out


def logged_setting_findings(t: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    """B4 "the arms' feedback, echo or guard settings off their rows", judged on what the session logged on the turn
    line (review F3: the trial's own setting fields are copied from the row, so comparing them is a tautology).
    ``live_feedback_logged`` is the row's feedback plus its echo (``None`` when the row has no feedback: the session
    logs the key only when given), ``recover_thought_calls_logged`` and ``visual_tokens_logged`` the row's values, and
    the guard logged on. A trial that wrote no turn line (T-0135b A7) has nothing to judge."""
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
    return out


def wrote_no_turn_line(t: Mapping[str, Any]) -> bool:
    """T-0135b A7: a trial that hit the turn timeout (``timed_out`` true) or never had a session (``session_id``
    ``None``) wrote no turn line, so it has no turn-line key to log."""
    return t.get("timed_out") is True or t.get("session_id") is None


def arm_setting_findings(trials: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """B1/A10 evidence: ``(findings, exempted)``. Every trial that scanned requests carries its arm's
    ``page_overview`` (A False, B True), ``degeneracy_guard`` True and its arm's ``tools_version``; its turn line
    logged A10's five counts in B and no ``page_overview`` key in A, and the guard on -- except on a trial that wrote
    no turn line (``wrote_no_turn_line``), listed in ``exempted`` (T-0135b A7)."""
    want = {"A": (False, V1_SHA256), "B": (True, OVERVIEW_SHA256)}
    findings: list[str] = []
    exempted: list[str] = []
    for t in trials:
        if not t.get("scanned_requests"):
            continue
        arm, tid = str(t["arm"]), _tid(t)
        overview, sha = want[arm]
        if t.get("page_overview") is not overview:
            findings.append(f"{tid}: page_overview {t.get('page_overview')!r}")
        if t.get("degeneracy_guard") is not True:
            findings.append(f"{tid}: degeneracy_guard {t.get('degeneracy_guard')!r}")
        if t.get("tools_version") != sha:
            findings.append(f"{tid}: tools_version {t.get('tools_version')!r}")
        if wrote_no_turn_line(t):
            exempted.append(tid)
            continue
        logged = t.get("page_overview_logged")
        if arm == "A" and logged is not None:
            findings.append(f"{tid}: page_overview_logged {logged!r} on an arm-A turn line")
        if arm == "B" and not (
            isinstance(logged, Mapping)
            and set(logged) == set(A10_KEYS)
            and all(isinstance(v, int) and not isinstance(v, bool) for v in logged.values())
        ):
            findings.append(f"{tid}: page_overview_logged {logged!r}")
        guard = t.get("degeneracy_guard_logged")
        if not (isinstance(guard, Mapping) and guard.get("on") is True):
            findings.append(f"{tid}: degeneracy_guard_logged {guard!r}")
    return findings, exempted


# --------------------------------------------------------------------------------------------- B3


def is_degenerate(t: Mapping[str, Any]) -> bool:
    return t.get("tool_loop_end") == "degenerate"


def engine_error(t: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """B5 (reading in the module docstring): transport, HTTP or timeout -- never a ``degenerate`` end."""
    if is_degenerate(t):
        return False
    if t.get("timed_out") is True or t.get("engine_up_after") is False:
        return True
    return any(a.get("http_status") != 200 for s in steps for a in g0111.attempts(s))


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
        # O-11: an arm with no trials (a probe-void run) has no interval
        "success_ci": list(wilson(success, len(trials))) if trials else None,
        "per_task": per_task,
        "s4": per_task["S4"],
        "s4_ci": list(wilson(per_task["S4"], len(s4))) if s4 else None,
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
        "narration": sum(1 for t in trials if NARRATION.search(g0111.voiced_text(t))),
        "degenerate": sum(1 for t in trials if is_degenerate(t)),
        "engine_errors": sum(1 for t in trials if engine_error(t, by[_turn(t)])),
        "latency": {"overview": g0111.dist(shown), "other": g0111.dist(other)},
    }


#: T-0131's per-arm sections the harness carries over (``success``/``false_claims`` renamed, as T-0134's)
T0131_SECTIONS = t0134.T0131_SECTIONS


def t0131_section(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """One arm's T-0131 section (steps, latency, query tokens and the prefix-cache hit, wall time, VRAM, the engine
    flag, every safety count) by T-0131's oracle at 1120 with ReGround off, with T-0134's two renames and (O-B-10(a))
    T-0131's step latency as ``latency_v1``: ``latency`` is B3's overview/other split."""
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
    """B5's inputs for one arm, re-derived: acceptance (T-0111g's per trial, over the trials that are neither an
    engine error nor ``degenerate``), the safety sum (T-0131's), the engine errors, B3's counts, and per clause the
    first failing trial ``"<task>#<repeat>"``."""
    by = {_turn(t): [s for s in steps if _turn(s) == _turn(t)] for t in trials}
    m = derive_arm(trials, steps)
    plain = [t for t in trials if not forced_failure(t, by[_turn(t)])]
    failing = [t for t in plain if not g0111.accepted(t, by[_turn(t)])]
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
        "false_claims_non_s4": m["false_claims_non_s4"],
        "over_clicks": m["over_clicks"],
        "s4_overview_call_turns": m["s4_overview_call_turns"],
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
    """B5: P1 failed -> inconclusive; a configuration finding -> void (``configuration: <first>``); P2/P3/P6/P7
    failed (``void``) -> void; stopped -> incomplete; an arm ineligible (acceptance, safety, more than 2 engine
    errors) -> inconclusive naming the arm, the clause and the first failing trial; then ``overview`` iff every
    adoption clause holds; else ``overview unused`` when arm B's S4 overview-call turns are under 5, else ``no
    change``."""
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


def decide(inputs: Mapping[str, Mapping[str, Any]], **kw: Any) -> str:
    return decide_with_reason(inputs, **kw)[0]


def differences(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
    """B3: Newcombe 95 % of B-A on S4 (of 20) and on success (of 100), at B3's fixed denominators."""
    a, b = arms["A"], arms["B"]
    return {
        "s4 B-A": newcombe(b["s4"], S4_TURNS, a["s4"], S4_TURNS),
        "success B-A": newcombe(b["success"], TURNS_PER_ARM, a["success"], TURNS_PER_ARM),
    }


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
        pytest.skip(f"{ENV} is unset: the T-0140 A/B has not run (docs/tasks/T-0140.md B6)")
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
    """B7 completeness: the manifest's arm rows are B1's two rows over its recorded live default (tools v1 and the
    overview set at their pins), its arm-table sha is theirs, the task-table sha matches its pin, the manifest's order
    is the 200-turn seed-0 joint order and the trials ran in it."""
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
    """B1: the manifest records a well-formed live default with the overview off, and every trial that ran a session
    ran at it."""
    recorded = run["manifest"].get("live_default") or {}
    assert live_default_findings(recorded) == [], (recorded, live_default_findings(recorded))
    for t in run["trials"]:
        if t.get("scanned_requests"):
            got = {k: t.get(k) for k in DEFAULT_KEYS}
            assert got == {k: recorded[k] for k in DEFAULT_KEYS}, (_tid(t), got, recorded)


def test_arm_setting_reached_every_turn(run: dict[str, Any]) -> None:
    """B1/A10 evidence (``arm_setting_findings``): each arm's overview, guard and tool set reached every turn that
    sent a body; a turn that wrote no turn line is exempted from the turn-line keys only (T-0135b A7)."""
    findings, exempted = arm_setting_findings(run["trials"])
    assert findings == [], findings
    if exempted:
        print(f"exempted (no turn line): {exempted}")


@pytest.mark.parametrize("arm", LABELS)
def test_arm_metrics_re_derived(run: dict[str, Any], arm: str) -> None:
    """B3: every per-arm count re-derived from the jsonl files equals report.json's."""
    want = derive_arm(_of(run["trials"], arm), _of(run["steps"], arm))
    got = run["report"]["arms"][arm]
    for key, value in want.items():
        if key in ("success_ci", "s4_ci") and (value is None or got[key] is None):
            assert got[key] is value is None, (key, got[key], value)
        elif key in ("success_ci", "s4_ci"):
            assert all(abs(x - y) < 1e-6 for x, y in zip(got[key], value, strict=True)), (
                key,
                got[key],
                value,
            )
        else:
            assert got[key] == value, (arm, key, got[key], value)


@pytest.mark.parametrize("arm", LABELS)
def test_t0131_sections_re_derived(run: dict[str, Any], arm: str) -> None:
    """B3 (the rest): steps per turn, step latency, query tokens, the prefix-cache hit, turn wall time, the minimum
    free VRAM, acceptance, the engine flag and every safety count, re-derived per arm by T-0131's oracle."""
    want = t0131_section(
        _of(run["trials"], arm), _of(run["steps"], arm), _of(run["regrounds"], arm)
    )
    got = run["report"]["arms"][arm]
    keys = [k for k in T0131_SECTIONS if k != "latency"]  # O-B-10(a): B3 owns ``latency``
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
    """B2: P1-P3, P6 and P7 re-derived from probes.json; each recorded ``passed`` and the report's ``void`` agree,
    and a failed P1 is labelled ``inconclusive``, a failed P2/P3/P6/P7 ``void``."""
    probes = run["probes"]
    v = probe_verdicts(probes)
    for p in ("p1", "p2", "p3", "p6", "p7"):
        assert bool(probes[p].get("passed")) is v[p], (p, probes[p], v[p])
    assert bool(run["report"].get("void")) is v["void"], (run["report"].get("void"), v)
    if not v["p1"]:
        assert run["report"]["decision"]["label"] == INCONCLUSIVE, run["report"]["decision"]
    elif v["void"]:
        assert run["report"]["decision"]["label"] == VOID, run["report"]["decision"]


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
    """B3: the Newcombe intervals for B-A on S4 and on success, from counts re-derived from the records."""
    arms = {arm: derive_arm(_of(run["trials"], arm), _of(run["steps"], arm)) for arm in LABELS}
    got = run["report"]["differences"]
    for key, (lo, hi) in differences(arms).items():
        assert abs(got[key][0] - lo) < 1e-6 and abs(got[key][1] - hi) < 1e-6, (
            key,
            got[key],
            (lo, hi),
        )
