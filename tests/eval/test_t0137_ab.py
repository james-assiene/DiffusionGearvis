"""T-0137 group C: the locked 2-arm GPU A/B of ``--repeat-guard`` on T-0138's suite S6-S15 (arm A HEAD's live default,
arm B the same plus ``repeat_guard=True``), re-derived from the run's records and compared with ``report.json``
(``gpu``, ``eval``; C-7).

Run through ``/launch`` on the 5090 box (the eval step of ``scripts/locked/t0137_repeat_guard_ab.sh``)::

    T0137_RUN_DIR=logs/t0137-repeat-guard-ab-<sha7> uv run pytest tests/eval/test_t0137_ab.py -q -m gpu -rA

Every run row skips when ``T0137_RUN_DIR`` is unset or the directory has no ``report.json``. They spend no GPU time but
are ``gpu`` because the data exists only after a ``/launch``-reviewed run.

**Re-targeted onto T-0138's suite** (orchestrator, 2026-10-25: H-T137-6's default foresaw the fork once T-0138's suite
existed; ``docs/cd/suite-baseline.md`` shows S9's silent scrolls, where a repeat stop can be exercised). The task file's
C-1...C-5 name T-0151's S1-S5 harness; this module reads them on the suite: 2 arms x 10 tasks x 10 = 200 turns, joint
seed-0 order, T-0138's probes (P1-P3, P-S1, P-S2, P-S3), paired exact McNemar by ``(task, repeat)``, C-3's W1 and
mechanism counts, C-4's voids, and C-5 re-expressed (below).

**The oracle is this module, not the harness** (C-7). It never imports the harness, nor any script under
``scripts/``, nor ``local_astra.conductor.progress``: **the T-0137 rule (A-3...A-6 and O-A2) is ported here from
docs/tasks/T-0137.md**, and ``tests/unit/test_t0137_ab_harness.py`` checks the port against ``progress.judge`` (group
A's reviewed module) on random histories, on CPU. It reuses T-0138's eval oracle (``tests/eval/test_t0138_ab.py``,
the record of a completed run, content-pinned by the unit module's C-8 rows) for what the guard does not change: the
judge-based success and false claim, the probe rules, the safety and latency sections, McNemar's exact p, the
step-1 placement rule of the overview frame and the image mask; and T-0111g's ``voiced_text``/``dist``.

**Record fields read** (the harness surface is pinned in ``tests/unit/test_t0137_ab_harness.py``'s docstring): T-0138's
(that module's docstring), and

- a trial carries its arm's settings as passed: T-0138's plus ``action_outcome`` (T-0136, HEAD's default) and
  ``repeat_guard`` (``False`` in A, ``True`` in B); the turn line's keys (``None`` when absent) T-0138's plus
  ``action_outcome_logged`` (T-0136 B's counts) and ``repeat_guard_logged`` (B-8's five ints); and ``history`` (the
  turn's stored history text, as T-0138's ``turn_records`` writes it, or ``None``);
- a step line carries T-0138's fields plus (C-1) ``call_arguments`` (the call's arguments as the model wrote them,
  ``call.arguments``; ``None`` without a call), ``screen`` (the executor phash of the live frame the step's request
  showed, B-2; ``None`` on a blind step or without a call), ``repeat_of_step`` (the ``step`` of the
  ``repeat_of_step`` mapping in the step's settled tool result, B-5/B-6; else ``None``), ``no_progress``
  (``"repeat"``/``"recurring"`` on a stopped call, from its ``tool.call`` record, B-5; else ``None``), ``no_change``
  (``True``/``False``/``None``, B-3) and ``no_change_source`` (``"t0136"``/``"screenshot"``/``None``); and, pinned
  to group B's merged output (d0f1f02, O-C), ``guard_outcome`` (the step's B-3 outcome mapping as the guard holds it:
  ``{"screenshot"}``, or ``{"outcome", "screenshot"}`` when T-0136 measured the step; ``None`` without a call) and
  ``repeat_fact`` (the ``repeat_of_step`` mapping of the step's settled result, ``{"step", "action", **<step j's
  guard_outcome>}``, on an executed repeat and on a ``"repeat"`` stop alike, S-R1; else ``None``);
- ``manifest.json``: ``live_default`` (with ``action_outcome`` and ``repeat_guard``), ``regime.arms`` (``arm_rows``
  of it), ``arm_table_sha256``, ``suite_table_sha256``, ``order``, ``guidance_sha256`` (``LIVE_TOOL_GUIDANCE``'s
  sha256 at run time);
- ``report.json``: ``{"void", "stopped", "probes", "arms": {"A", "B"}, "mcnemar", "configuration_findings",
  "decision": {"label", "reason", "inputs"}}``; each arm ``derive_arm``'s keys plus ``safety`` and ``latency``.

**Readings** (test-author, reported to the orchestrator as ambiguities):

1. *Success*, *false claim*, *engine error*, *acceptance* are T-0138's (its eval readings 1 and 5). *Over-actions*
   are the judge's ``over_action`` flags summed over T-0138's six over-action tasks (out of 60), the suite's
   equivalent of T-0151's over-clicks out of 60. T-0151's S4/S5 rows have no suite equivalent and are dropped.
2. *The guard's history* (A-2) is the turn's step lines that carry a call and a status, in step order; answer-only
   steps are not settled steps. A step line's ``overview`` is ``frame_kind == "overview"``.
3. *Which calls the guard judged* (B-4): a call at loop step >= 2 whose name is in ``ACTIONS`` and that was not
   refused with ``FRESHNESS_REASON`` (B9 first). For a judged call the rule's ``stop`` must equal the logged
   ``no_progress``; a stopped call is ``refused`` with that stop's reason, and its ``repeat_of_step`` is the rule's
   ``repeat_of`` for a ``"repeat"`` stop and ``None`` for ``"recurring"``. An unstopped *executed* judged call's
   ``repeat_of_step`` is the rule's ``repeat_of``; every other step's is ``None`` (B-6: failed, timed-out and
   non-guard refusals get no fact).
4. *W1* (C-3) counts, over every arm, the call step lines whose name is in ``ACTIONS``, whose status is ``executed``
   or ``refused`` (any refusal, B9 included), and whose rule verdict over the earlier steps has ``repeat_of`` set.
5. *Mechanism counts* (C-3, both arms, from the logged fields): ``repeat_facts`` by the referenced step's
   ``no_change`` (``"no change"``/``"changed"``/``"unknown"``), counted on executed, unstopped steps with a
   ``repeat_of_step``; ``stops`` by kind; ``after_stop`` -- the step line after a stop: a call (``called_again``),
   no call with voiced text (``answered_voiced``), none and a ``budget`` end (``budget``), anything else (``other``);
   ``stops_in`` the stops in turns that succeeded or failed.
6. *C-5 re-expressed on the suite* (an ambiguity; the orchestrator's defaults): the order is T-0151's
   (``inconclusive``, ``void``, ``incomplete``, an ineligible arm); then ``not exercised`` when W1(A) < 6; else
   ``adopt`` when W1(B) <= W1(A) / 2, pooled success rate(B) >= rate(A) - 3/100, false claims(B) <= false
   claims(A) + 2 and over-actions(B) <= over-actions(A) + 2, each with a 1e-9 tolerance; else ``no change``. T-0151's
   S4 clause has no suite equivalent and is dropped. McNemar is reported, never decided on.
7. *C-4 on the suite*: T-0138's configuration findings that apply when both arms run HEAD's default (the overview
   frame on or off in **both** arms, as the default says; the step-1 pair check over **every** task, image bytes
   masked), plus the guard's: a ``repeat_guard`` key, a ``repeat_of_step``, a ``no_progress`` or a guard reason in
   arm A (its step lines or its bodies' text); an arm-B turn line without a well-formed key, or whose stop counts
   disagree with its step lines; a stop or fact the ported rule does not justify, or the reverse; a guard reason in
   a trial's voiced sources, a step's voiced sources or its history; a ``no_change`` off its source; the
   ``LIVE_TOOL_GUIDANCE`` sha off its pin; a tools sha off v1.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.eval import test_live_smoke as g0111
from tests.eval import test_t0138_ab as t38

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0137_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]

LABELS = ("A", "B")
TASK_IDS = t38.TASK_IDS
CENSUS = t38.CENSUS
SHOWN_TASKS = t38.SHOWN_TASKS
OVER_ACTION_TASKS = t38.OVER_ACTION_TASKS
REPEATS = 10
TURNS_PER_ARM = 100
TOTAL_TURNS = 200
SUITE_TABLE_SHA256 = t38.SUITE_TABLE_SHA256
V1_SHA256 = t38.V1_SHA256
#: B-OFF's frozen ``LIVE_TOOL_GUIDANCE`` (768a973b...), a full literal
LIVE_TOOL_GUIDANCE_SHA256 = "768a973bb546979536d2964f1c9767b764346843ef35921769b8cd75638af2ec"
DEFAULT_KEYS = t38.DEFAULT_KEYS
#: the settings that follow HEAD's default and are recorded in both arms (C-1)
FOLLOWED_KEYS = ("overview_frame", "action_outcome")
#: the settings HEAD's default must have off (tools v1 in both arms, no ReGround: the T-0138 B2 pattern)
OFF_KEYS = ("reground", "page_overview", "auto_overview")
#: C-1: arm A HEAD's default (the guard off), arm B plus the guard
REPEAT_GUARD_ON = {"A": False, "B": True}
GUARD_ARM = "B"

# ------------------------------------------------------------------ the T-0137 rule, ported (A-1, typed)
REPEAT_TOLERANCE = 10
MAX_NO_CHANGE_REPEATS = 2
MAX_SCREEN_VISITS = 3
ACTIONS = frozenset({"click_at", "type_at", "scroll", "press_key"})
NO_PROGRESS_REPEAT_REASON = (
    "the same action did not change this screen the last two times; it was not done again"
)
NO_PROGRESS_RECURRING_REASON = (
    "this screen has come back for the third time in this request; the action was not done"
)
STOP_REASONS = {"repeat": NO_PROGRESS_REPEAT_REASON, "recurring": NO_PROGRESS_RECURRING_REASON}
GUARD_REASONS = (NO_PROGRESS_REPEAT_REASON, NO_PROGRESS_RECURRING_REASON)
#: T-0111g B9's stale-frame reason (``session.FRESHNESS_REASON``, typed): such a call never reaches the guard (B-4)
FRESHNESS_REASON = "the screen changed since this frame; take a new frame"
#: B-8's turn-line key. As merged (d0f1f02, ``Session._guard_counts``): ``repeats`` the executed calls that carried a
#: fact; ``no_change_repeats`` those of them whose own step then showed no change; the stops by kind;
#: ``max_screen_visits`` the largest ``screen_visits`` of any call the guard judged (0 when none)
REPEAT_GUARD_KEYS = (
    "repeats",
    "no_change_repeats",
    "stops_repeat",
    "stops_recurring",
    "max_screen_visits",
)
NO_CHANGE_SOURCES = ("t0136", "screenshot")
#: T-0136's no-change literal (``action_outcome.NO_VISIBLE_CHANGE``, typed)
NO_VISIBLE_CHANGE = "no visible change"
REPEAT_FACT_KEYS = ("no change", "changed", "unknown")
AFTER_STOP_KEYS = ("answered_voiced", "called_again", "budget", "other")

# ------------------------------------------------------------------ C-5 (reading 6)
W1_EXERCISED_MIN = 6
W1_RATIO = 0.5
SUCCESS_MARGIN = 0.03
FALSE_CLAIM_MARGIN = 2
OVER_ACTION_MARGIN = 2
EPS = 1e-9
ENGINE_ERRORS_MAX = 2
INCONCLUSIVE = "inconclusive"
VOID = "void"
INCOMPLETE = "incomplete"
NOT_EXERCISED = "not exercised"
ADOPT = "adopt"
NO_CHANGE = "no change"
CLAUSES = ("acceptance", "safety", "engine")
STEP_CAP_END = t38.STEP_CAP_END

wilson = t38.wilson


# --------------------------------------------------------------------------------------------- C-1: arms, order


def live_default_findings(default: Mapping[str, Any]) -> list[str]:
    """C-1: the recorded live default is well formed; the guard is **off** (else the arms would be identical arms);
    ReGround, the page overview and the auto overview are off (tools v1 in both arms); the overview frame and T-0136's
    outcome label are booleans, whatever HEAD's default says."""
    keys = (*DEFAULT_KEYS, *FOLLOWED_KEYS, "page_overview", "auto_overview", "repeat_guard")
    out = [f"live_default lacks {k}" for k in keys if k not in default]
    if out:
        return out
    if default["visual_tokens"] not in (280, 560, 1120):
        out.append(f"visual_tokens {default['visual_tokens']!r}")
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
    for k in OFF_KEYS:
        if default[k] is not False:
            out.append(f"{k} {default[k]!r}: on by default")
    for k in FOLLOWED_KEYS:
        if not isinstance(default[k], bool):
            out.append(f"{k} {default[k]!r}: not a boolean")
    if default["repeat_guard"] is not False:
        out.append(
            f"repeat_guard {default['repeat_guard']!r}: the guard is on by default, so the arms would be "
            "identical arms"
        )
    return out


def arm_rows(default: Mapping[str, Any]) -> list[dict[str, Any]]:
    """C-1: both arms at the live default (its five settings, its overview frame and its T-0136 label), the T-0142
    guard on, the page and auto overviews off, tools v1; A without the repeat guard, B with it."""
    shared = {k: default[k] for k in DEFAULT_KEYS}
    return [
        {
            "label": label,
            **shared,
            "degeneracy_guard": True,
            "page_overview": False,
            "auto_overview": False,
            "overview_frame": default["overview_frame"],
            "action_outcome": default["action_outcome"],
            "repeat_guard": REPEAT_GUARD_ON[label],
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
    """The 200 ``(arm, task, repeat)`` turns, arm-major over ``TASK_IDS`` and ``range(REPEATS)``, then one
    ``random.Random(seed).shuffle`` of the whole list (jointly; T-0138's construction)."""
    turns = [(a, t, r) for a in LABELS for t in TASK_IDS for r in range(REPEATS)]
    random.Random(seed).shuffle(turns)
    return turns


# --------------------------------------------------------------------------------------------- C-2: probes

ps1_ok = t38.ps1_ok
ps2_ok = t38.ps2_ok
ps3_ok = t38.ps3_ok
probe_verdicts = t38.probe_verdicts


# --------------------------------------------------------------------------------------------- the ported rule


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _close(a: Any, b: Any) -> bool:
    """One coordinate: within ``REPEAT_TOLERANCE`` when both are numbers, else equal."""
    if _number(a) and _number(b):
        return float(abs(a - b)) <= REPEAT_TOLERANCE
    return bool(a == b)


def same_action(p: Mapping[str, Any], q: Mapping[str, Any]) -> bool:
    """A-3 over two calls ``{"name", "arguments", "screen", "overview"}``."""
    name = p.get("name")
    if name != q.get("name") or name not in ACTIONS:
        return False
    if p.get("screen") is None or q.get("screen") is None or p["screen"] != q["screen"]:
        return False
    if p.get("overview") or q.get("overview"):
        return False
    a = dict(p.get("arguments") or {})
    b = dict(q.get("arguments") or {})
    if name == "scroll":
        return bool(a.get("direction") == b.get("direction"))
    if name == "press_key":
        return bool(a.get("combo") == b.get("combo"))
    # click_at and type_at: Chebyshev, both axes within the tolerance
    if not (_close(a.get("x"), b.get("x")) and _close(a.get("y"), b.get("y"))):
        return False
    if name == "type_at":
        # a missing ``press_enter`` is ``False``
        same_enter = a.get("press_enter", False) == b.get("press_enter", False)
        return bool(a.get("text") == b.get("text") and same_enter)
    return True


def rule_verdict(history: Sequence[Mapping[str, Any]], call: Mapping[str, Any]) -> dict[str, Any]:
    """A-4...A-6 with O-A2 over ``history`` (calls ``{"step", "name", "arguments", "screen", "status", "overview",
    "no_change"}`` in order) and the proposed ``call``: ``{"repeat_of", "stop", "no_change_repeats",
    "screen_visits"}``."""
    instances = [h for h in history if h.get("status") == "executed" and same_action(h, call)]
    repeat_of = int(instances[-1]["step"]) if instances else None
    # O-A2: the trailing run of no-change repeats (the original never counts)
    trailing = 0
    for h in instances[1:]:
        trailing = trailing + 1 if h.get("no_change") is True else 0
    screen = call.get("screen")
    visits = 0
    if screen is not None:
        seq = [h.get("screen") for h in history if h.get("screen") is not None] + [screen]
        runs = [x for k, x in enumerate(seq) if k == 0 or seq[k - 1] != x]
        visits = sum(1 for x in runs if x == screen)
    stop: str | None = None
    if call.get("name") in ACTIONS and screen is not None:
        if repeat_of is not None and trailing >= MAX_NO_CHANGE_REPEATS:
            stop = "repeat"
        elif visits >= MAX_SCREEN_VISITS:
            stop = "recurring"
    return {
        "repeat_of": repeat_of,
        "stop": stop,
        "no_change_repeats": trailing,
        "screen_visits": visits,
    }


def as_call(step: Mapping[str, Any]) -> dict[str, Any]:
    """Reading 2: a step line as the rule's call."""
    return {
        "step": int(step["step"]),
        "name": step.get("call_name"),
        "arguments": dict(step.get("call_arguments") or {}),
        "screen": step.get("screen"),
        "status": step.get("status"),
        "overview": step.get("frame_kind") == "overview",
        "no_change": step.get("no_change"),
    }


def _settled(steps: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        (s for s in steps if s.get("call_name") is not None and s.get("status") is not None),
        key=lambda s: int(s["step"]),
    )


def judged_by_guard(step: Mapping[str, Any]) -> bool:
    """Reading 3 (B-4): loop step >= 2, a browser action, not refused by B9 first."""
    if int(step["step"]) < 2 or step.get("call_name") not in ACTIONS:
        return False
    return not (step.get("status") == "refused" and step.get("reason") == FRESHNESS_REASON)


def guard_expectations(steps: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Reading 3, per settled call step of one turn: the rule's verdict, and the ``no_progress`` and
    ``repeat_of_step`` the guard must have logged."""
    calls = _settled(steps)
    out: dict[int, dict[str, Any]] = {}
    for k, s in enumerate(calls):
        v = rule_verdict([as_call(h) for h in calls[:k]], as_call(s))
        judged = judged_by_guard(s)
        stop = v["stop"] if judged else None
        if stop == "repeat" or (stop is None and judged and s.get("status") == "executed"):
            fact = v["repeat_of"]
        else:
            fact = None
        out[int(s["step"])] = {"verdict": v, "no_progress": stop, "repeat_of_step": fact}
    return out


def w1_steps(steps: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Reading 4: one turn's repeat steps (executed or refused)."""
    calls = _settled(steps)
    out = []
    for k, s in enumerate(calls):
        if s.get("call_name") not in ACTIONS or s.get("status") not in ("executed", "refused"):
            continue
        if rule_verdict([as_call(h) for h in calls[:k]], as_call(s))["repeat_of"] is not None:
            out.append(s)
    return out


# --------------------------------------------------------------------------------------------- C-4: findings


def _key(r: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(r["arm"]), str(r["task"]), int(r["repeat"]))


def _tid(r: Mapping[str, Any]) -> str:
    return f"{r['arm']} {r['task']}#{r['repeat']}"


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


wrote_no_turn_line = t38.wrote_no_turn_line


def _rg_well_formed(logged: Any) -> bool:
    return (
        isinstance(logged, Mapping)
        and set(logged) == set(REPEAT_GUARD_KEYS)
        and all(_int(v) and v >= 0 for v in logged.values())
    )


def _body_findings(t: Mapping[str, Any], rec: Mapping[str, Any], frame_on: bool) -> list[str]:
    """T-0138 B7's body rules (its eval's ``_body_findings``) with the frame taken from the arm's row, not from the arm
    label: at HEAD's default both arms show the overview frame on S7/S11 (reading 7)."""
    out: list[str] = []
    tid, task = _tid(t), str(t["task"])
    overviews: dict[str, list[Mapping[str, Any]]] = {}
    for o in rec.get("overviews") or []:
        overviews.setdefault(str(o["sha"]), []).append(o)
    if not frame_on and overviews:
        out.append(f"{tid}: a frame-off turn registered an overview frame")
    step1_carries: list[bool] = []
    step1_any: list[bool] = []
    generate_frames: list[bool] = []
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
        if len(images) > t38.MAX_IMAGES:
            out.append(f"{where}: {len(images)} images")
        mine = [x for x in images if x[2] in overviews]
        if len(mine) > 1:
            out.append(f"{where}: {len(mine)} overview-frame images")
        users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        last_user = users[-1] if users else None
        placed: set[tuple[int, int]] = set()
        for i, j, sha in mine:
            if not frame_on:
                out.append(f"{where}: a frame-off body carries an overview-frame image")
            elif any(t38._placed(r, messages, last_user, i, j, o) for o in overviews[sha]):
                placed.add((i, j + 1))
            else:
                out.append(f"{where}: an overview-frame image out of place")
        for i, m in enumerate(messages):
            if m.get("role") != "user":
                continue
            for j, p in enumerate(m.get("parts") or []):
                text = p.get("text")
                if not isinstance(text, str):
                    continue
                if t38.OVERVIEW_FRAME_STEM in text and (i, j) not in placed:
                    out.append(f"{where}: an overview-frame label out of place")
                if t38.OVERVIEW_STEM in text:
                    out.append(f"{where}: an auto/page overview label in a body")
                if t38.SCREEN_LABEL_STEM in text and placed and i == last_user:
                    out.append(f"{where}: a T-0134 label on the overview-frame body")
        if t38._is_step1_body(r):
            step1_carries.append(bool(placed))
            step1_any.append(bool(mine))
        if r.get("phase") == "generate" and last_user is not None:
            frame = any("image" in p for p in messages[last_user].get("parts") or [])
            generate_frames.append(frame and not mine)
    logged = t.get("overview_frame_logged")
    if frame_on and not wrote_no_turn_line(t) and t38._a2_well_formed(logged):
        if logged["shown"] == 0 and any(step1_any):
            out.append(
                f"{tid}: a step-1 body carries the overview frame while the turn line's shown is 0"
            )
        if logged["shown"] >= 1 and not (step1_carries and all(step1_carries)):
            out.append(
                f"{tid}: the turn line's shown is {logged['shown']} but a step-1 body is not exactly the overview "
                "frame and its label"
            )
    if (
        frame_on
        and task in SHOWN_TASKS
        and not wrote_no_turn_line(t)
        and t38._counts_all_zero(logged)
        and any(generate_frames)
    ):
        out.append(
            f"{tid}: an unlogged absence (step 1 carried a frame but no overview, counts all 0)"
        )
    return out


def logged_setting_findings(t: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    """The turn line: the arm's settings as logged; the overview-frame key exactly when the row's frame is on; no
    page/auto overview key; T-0136's key exactly when its label is on; the repeat-guard key, well formed, exactly when
    the row's guard is on (C-4). A trial with no turn line has nothing to judge."""
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
    if row.get("overview_frame") is True and not t38._a2_well_formed(of):
        out.append(f"{tid}: overview_frame_logged {of!r} on a frame-on turn line")
    if row.get("overview_frame") is not True and of is not None:
        out.append(f"{tid}: overview_frame_logged {of!r} on a frame-off turn line")
    for k in ("auto_overview_logged", "page_overview_logged"):
        if t.get(k) is not None:
            out.append(f"{tid}: {k} {t.get(k)!r} on a turn line")
    ao = t.get("action_outcome_logged")
    if row.get("action_outcome") is True and not isinstance(ao, Mapping):
        out.append(f"{tid}: action_outcome_logged {ao!r} on an outcome-label turn line")
    if row.get("action_outcome") is not True and ao is not None:
        out.append(f"{tid}: action_outcome_logged {ao!r} with the outcome label off")
    rg = t.get("repeat_guard_logged")
    if row.get("repeat_guard") is True and not _rg_well_formed(rg):
        out.append(f"{tid}: repeat_guard_logged {rg!r} on a guard-on (arm B) turn line")
    if row.get("repeat_guard") is not True and rg is not None:
        out.append(f"{tid}: repeat_guard_logged {rg!r} on a guard-off (arm A) turn line")
    return out


def _texts_of(line: Mapping[str, Any]) -> list[tuple[str, str]]:
    out = []
    for r in line.get("requests") or []:
        for m in r.get("messages") or []:
            for p in m.get("parts") or []:
                if isinstance(p.get("text"), str):
                    out.append((str(r.get("label")), p["text"]))
    return out


def guard_findings(
    t: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    line: Mapping[str, Any],
    row: Mapping[str, Any],
) -> list[str]:
    """C-4's guard findings for one trial (reading 7): guard-off arm clean; guard-on arm consistent with the ported
    rule both ways and with its turn line; no guard reason voiced or stored; ``no_change`` on its source."""
    tid = _tid(t)
    out: list[str] = []
    mine = sorted(steps, key=lambda s: int(s["step"]))
    spoken = [("voiced", str(v)) for v in t.get("voiced") or []]
    spoken += [
        (f"step {s['step']} voiced_sources", str(v))
        for s in mine
        for v in s.get("voiced_sources") or []
    ]
    if t.get("history"):
        spoken.append(("history", str(t["history"])))
    for where, text in spoken:
        if any(reason in text for reason in GUARD_REASONS):
            out.append(f"{tid}: a guard reason in {where}")
    for s in mine:
        k, src, nc = int(s["step"]), s.get("no_change_source"), s.get("no_change")
        if src is not None and src not in NO_CHANGE_SOURCES:
            out.append(f"{tid} step {k}: no_change_source {src!r}")
        if src is None and nc is not None:
            out.append(f"{tid} step {k}: no_change {nc!r} without a source")
        shot = s.get("screenshot")
        if (
            src == "screenshot"
            and shot in ("updated", "unchanged")
            and nc is not (shot == "unchanged")
        ):
            out.append(f"{tid} step {k}: no_change {nc!r} off its screenshot {shot!r}")
        if src == "t0136" and row.get("action_outcome") is not True:
            out.append(f"{tid} step {k}: a t0136 no_change with the outcome label off")
        out += _outcome_findings(tid, s)
    out += _fact_findings(tid, mine)
    if row.get("repeat_guard") is not True:
        for s in mine:
            k = int(s["step"])
            if s.get("repeat_of_step") is not None:
                out.append(
                    f"{tid} step {k}: repeat_of_step {s['repeat_of_step']!r} in a guard-off arm"
                )
            if s.get("no_progress") is not None:
                out.append(f"{tid} step {k}: no_progress {s['no_progress']!r} in a guard-off arm")
            if s.get("reason") in GUARD_REASONS:
                out.append(f"{tid} step {k}: a guard reason in a guard-off arm")
        for label, text in _texts_of(line):
            if "repeat_of_step" in text or any(reason in text for reason in GUARD_REASONS):
                out.append(
                    f"{tid} {label}: a guard-off body carries a repeat fact or a guard reason"
                )
        return out
    expected = guard_expectations(mine)
    for s in mine:
        k = int(s["step"])
        e = expected.get(k, {"no_progress": None, "repeat_of_step": None})
        got_stop, got_fact = s.get("no_progress"), s.get("repeat_of_step")
        if got_stop != e["no_progress"]:
            out.append(
                f"{tid} step {k}: no_progress {got_stop!r}, the rule gives {e['no_progress']!r}"
            )
        if got_fact != e["repeat_of_step"]:
            out.append(
                f"{tid} step {k}: repeat_of_step {got_fact!r}, the rule gives {e['repeat_of_step']!r}"
            )
        if got_stop is not None:
            if s.get("status") != "refused" or s.get("reason") != STOP_REASONS.get(str(got_stop)):
                out.append(f"{tid} step {k}: a {got_stop!r} stop not refused with its reason")
        elif s.get("reason") in GUARD_REASONS:
            out.append(f"{tid} step {k}: a guard reason without a stop")
    logged = t.get("repeat_guard_logged")
    if not wrote_no_turn_line(t) and _rg_well_formed(logged):
        want_counts = turn_line_counts(mine)
        for name in REPEAT_GUARD_KEYS:
            if logged[name] != want_counts[name]:
                out.append(
                    f"{tid}: the turn line's {name} is {logged[name]}, its step lines give {want_counts[name]}"
                )
    return out


def turn_line_counts(steps: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """B-8 as merged (``REPEAT_GUARD_KEYS``' comment), from one turn's step lines and the ported rule."""
    mine = sorted(steps, key=lambda s: int(s["step"]))
    facts = [
        s
        for s in mine
        if s.get("status") == "executed"
        and s.get("repeat_of_step") is not None
        and s.get("no_progress") is None
    ]
    expected = guard_expectations(mine)
    at = {int(s["step"]): s for s in mine}
    visits = [e["verdict"]["screen_visits"] for k, e in expected.items() if judged_by_guard(at[k])]
    return {
        "repeats": len(facts),
        "no_change_repeats": sum(1 for s in facts if s.get("no_change") is True),
        "stops_repeat": sum(1 for s in mine if s.get("no_progress") == "repeat"),
        "stops_recurring": sum(1 for s in mine if s.get("no_progress") == "recurring"),
        "max_screen_visits": max(visits, default=0),
    }


def _outcome_findings(tid: str, s: Mapping[str, Any]) -> list[str]:
    """B-3 as merged: a call's ``guard_outcome`` is ``{"screenshot": <B8>}`` on the fallback, and ``{"outcome": <T-0136
    text>, "screenshot": <B8>}`` when T-0136 measured it, whose ``no_change`` is exactly the no-change literal."""
    if s.get("call_name") is None or s.get("status") is None:
        return []
    k, src, got = int(s["step"]), s.get("no_change_source"), s.get("guard_outcome")
    shot = s.get("screenshot")
    if src == "screenshot":
        if got != {"screenshot": shot}:
            return [f"{tid} step {k}: guard_outcome {got!r} off its screenshot {shot!r}"]
    elif src == "t0136":
        if not (
            isinstance(got, Mapping)
            and set(got) == {"outcome", "screenshot"}
            and got["screenshot"] == shot
        ):
            return [
                f"{tid} step {k}: guard_outcome {got!r} is not T-0136's outcome and the screenshot"
            ]
        if s.get("no_change") is not (got["outcome"] == NO_VISIBLE_CHANGE):
            return [
                f"{tid} step {k}: no_change {s.get('no_change')!r} off T-0136's {got['outcome']!r}"
            ]
    return []


def _fact_findings(tid: str, steps: Sequence[Mapping[str, Any]]) -> list[str]:
    """B-6/S-R1 as merged: a step with a ``repeat_of_step`` carries the mapping ``{"step": j, "action": <str>,
    **<step j's guard_outcome>}`` (``outcome`` among its keys exactly when T-0136 measured step j); a step without one
    carries none."""
    at = {int(s["step"]): s for s in steps}
    out: list[str] = []
    for s in steps:
        k, j, fact = int(s["step"]), s.get("repeat_of_step"), s.get("repeat_fact")
        if j is None:
            if fact is not None:
                out.append(f"{tid} step {k}: repeat_fact {fact!r} without a repeat_of_step")
            continue
        ref = at.get(int(j)) or {}
        want = {"step": int(j), **dict(ref.get("guard_outcome") or {})}
        if not (
            isinstance(fact, Mapping)
            and isinstance(fact.get("action"), str)
            and {key: v for key, v in fact.items() if key != "action"} == want
        ):
            out.append(
                f"{tid} step {k}: repeat_fact {fact!r}, step {j}'s fact is {want!r} plus its action"
            )
    return out


def pair_findings(
    trials: Sequence[Mapping[str, Any]], bodies: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Reading 7: per ``(task, repeat)`` of **every** task with both trials sent, the first step-1 ``generate`` bodies
    agree in ``tools`` and in ``messages`` with image bytes masked (T-0138 O-B2): the guard acts from step 2 on."""
    sent = {_key(t) for t in trials if t.get("scanned_requests")}
    by = {_key(b): b for b in bodies}
    out: list[str] = []
    for task, repeat in sorted(
        {(k[1], k[2]) for k in sent}, key=lambda x: (TASK_IDS.index(x[0]), x[1])
    ):
        if ("A", task, repeat) not in sent or ("B", task, repeat) not in sent:
            continue
        a = t38._first_step1_generate(by.get(("A", task, repeat)) or {})
        b = t38._first_step1_generate(by.get(("B", task, repeat)) or {})
        if a is None or b is None:
            continue
        if a.get("tools") != b.get("tools") or t38._masked(a) != t38._masked(b):
            out.append(
                f"{task}#{repeat}: the step-1 bodies of arms A and B differ in more than the request ids "
                f"({a.get('label')} vs {b.get('label')})"
            )
    return out


def configuration_findings(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    bodies: Sequence[Mapping[str, Any]],
    arms: Sequence[Mapping[str, Any]],
    *,
    guidance_sha256: str | None,
) -> list[str]:
    """C-4, one string per finding, naming the trial (``"<arm> <task>#<repeat>"``, with ``step <k>`` for a step), the
    pair (``"<task>#<repeat>"``) or the setting. The arm rows are judged as C-1 writes them; the trials only when they
    forwarded a body; every reground line is a finding."""
    out: list[str] = []
    rows = [dict(r) for r in arms]
    if [r.get("label") for r in rows] != list(LABELS):
        out.append(f"the arm rows are {[r.get('label') for r in rows]}")
    for k in sorted({k for r in rows for k in r} - {"label", "repeat_guard"}):
        values = [r.get(k) for r in rows]
        if any(v != values[0] for v in values):
            out.append(f"the arms differ in {k}: {values}")
    for r in rows:
        label = str(r.get("label"))
        if label in REPEAT_GUARD_ON and r.get("repeat_guard") is not REPEAT_GUARD_ON[label]:
            out.append(f"arm {label} row: repeat_guard {r.get('repeat_guard')!r}")
        if r.get("tools_sha256") != V1_SHA256:
            out.append(f"arm {label} row: tools {r.get('tools_sha256')!r}")
        for k in OFF_KEYS:
            if r.get(k) is not False:
                out.append(f"arm {label} row: {k} {r.get(k)!r}")
        if r.get("degeneracy_guard") is not True:
            out.append(f"arm {label} row: degeneracy_guard {r.get('degeneracy_guard')!r}")
        for k in FOLLOWED_KEYS:
            if not isinstance(r.get(k), bool):
                out.append(f"arm {label} row: {k} {r.get(k)!r}")
    if guidance_sha256 != LIVE_TOOL_GUIDANCE_SHA256:
        out.append(
            f"LIVE_TOOL_GUIDANCE sha256 {guidance_sha256!r}, not {LIVE_TOOL_GUIDANCE_SHA256}"
        )
    by_row = {str(r.get("label")): r for r in rows}
    for g in regrounds:
        out.append(f"{_tid(g)}: a reground line")
    by_bodies = {_key(b): b for b in bodies}
    by_steps: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {_key(t): [] for t in trials}
    for s in steps:
        if _key(s) in by_steps:
            by_steps[_key(s)].append(s)
    for t in trials:
        if not t.get("scanned_requests"):
            continue
        tid = _tid(t)
        row = by_row.get(str(t["arm"])) or {}
        bad = [b for b in t.get("budgets_sent") or [] if b != row.get("visual_tokens")]
        if bad or not t.get("budgets_sent"):
            out.append(f"{tid}: body budget {bad or 'none'}")
        if t.get("tools_version") != V1_SHA256:
            out.append(f"{tid}: tools_version {t.get('tools_version')!r}")
        for k in (
            "live_feedback",
            "echo",
            "recover_thought_calls",
            "degeneracy_guard",
            "overview_frame",
            "auto_overview",
            "page_overview",
            "action_outcome",
            "repeat_guard",
        ):
            if t.get(k) != row.get(k):
                out.append(f"{tid}: {k} {t.get(k)!r}, its row says {row.get(k)!r}")
        line = by_bodies.get(_key(t)) or {}
        logged = t.get("overview_frame_logged")
        shown = int(logged.get("shown") or 0) if isinstance(logged, Mapping) else 0
        if str(t["task"]) not in SHOWN_TASKS and (line.get("overviews") or shown > 0):
            out.append(f"{tid}: an overview frame on a {CENSUS[str(t['task'])]} task")
        out += logged_setting_findings(t, row)
        out += _body_findings(t, line, row.get("overview_frame") is True)
        out += guard_findings(t, by_steps[_key(t)], line, row)
    out += pair_findings(trials, bodies)
    return out


# --------------------------------------------------------------------------------------------- C-3: metrics


def _turn(r: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(r.get("arm")), str(r["task"]), int(r["repeat"]))


def _steps_of(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str, int], list[Mapping[str, Any]]]:
    by: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {_turn(t): [] for t in trials}
    for s in steps:
        if _turn(s) in by:
            by[_turn(s)].append(s)
    return by


def _ci(k: int, n: int) -> list[float] | None:
    return list(wilson(k, n)) if n else None


def mechanism_counts(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Reading 5: repeat facts by the referenced step's outcome, stops by kind, what followed each stop, and the
    stops in successful and failing turns."""
    by = _steps_of(trials, steps)
    facts = dict.fromkeys(REPEAT_FACT_KEYS, 0)
    stops = {"repeat": 0, "recurring": 0}
    after = dict.fromkeys(AFTER_STOP_KEYS, 0)
    stops_in = {"success": 0, "failure": 0}
    for t in trials:
        mine = sorted(by[_turn(t)], key=lambda s: int(s["step"]))
        at = {int(s["step"]): s for s in mine}
        ok = t38.succeeded(t, mine)
        for s in mine:
            k = int(s["step"])
            j = s.get("repeat_of_step")
            if j is not None and s.get("no_progress") is None and s.get("status") == "executed":
                nc = (at.get(int(j)) or {}).get("no_change")
                facts["no change" if nc is True else "changed" if nc is False else "unknown"] += 1
            stop = s.get("no_progress")
            if stop not in stops:
                continue
            stops[str(stop)] += 1
            stops_in["success" if ok else "failure"] += 1
            nxt = at.get(k + 1)
            if nxt is None:
                after["budget" if t.get("tool_loop_end") == STEP_CAP_END else "other"] += 1
            elif nxt.get("call_name") is not None:
                after["called_again"] += 1
            elif any(str(v).strip() for v in nxt.get("voiced_sources") or []):
                after["answered_voiced"] += 1
            else:
                after["other"] += 1
    return {"repeat_facts": facts, "stops": stops, "after_stop": after, "stops_in": stops_in}


def derive_arm(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """C-3's counts for one arm's records (readings 1, 4 and 5)."""
    by = _steps_of(trials, steps)
    ok = {_turn(t): t38.succeeded(t, by[_turn(t)]) for t in trials}
    v = {_turn(t): t38.verdict(t) for t in trials}
    per_task = {
        task: sum(1 for t in trials if t["task"] == task and ok[_turn(t)]) for task in TASK_IDS
    }
    per_n = {task: sum(1 for t in trials if t["task"] == task) for task in TASK_IDS}
    success = sum(1 for x in ok.values() if x)
    over = {
        task: sum(1 for t in trials if t["task"] == task and v[_turn(t)].over_action)
        for task in OVER_ACTION_TASKS
    }
    w1 = [s for t in trials for s in w1_steps(by[_turn(t)])]
    n_steps = [len(by[_turn(t)]) for t in trials]
    return {
        "n": len(trials),
        "success": success,
        "success_ci": _ci(success, len(trials)),
        "per_task": per_task,
        "per_task_n": per_n,
        "per_task_ci": {task: _ci(per_task[task], per_n[task]) for task in TASK_IDS},
        "over_action": over,
        "over_actions": sum(over.values()),
        "false_claims": sum(1 for t in trials if t38.false_claim(t, by[_turn(t)], ok[_turn(t)])),
        "steps_per_turn": g0111.dist(n_steps),
        "mean_steps": (sum(n_steps) / len(n_steps)) if n_steps else None,
        "step_cap_ends": sum(1 for t in trials if t.get("tool_loop_end") == STEP_CAP_END),
        "timeouts": sum(1 for t in trials if t.get("timed_out") is True),
        "engine_errors": sum(1 for t in trials if t38.engine_error(t, by[_turn(t)])),
        "degenerate": sum(1 for t in trials if t38.is_degenerate(t)),
        "w1": len(w1),
        "w1_by_status": {
            st: sum(1 for s in w1 if s.get("status") == st) for st in ("executed", "refused")
        },
        **mechanism_counts(trials, steps),
    }


def decision_inputs(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """C-5's per-arm inputs: T-0138's eligibility inputs (acceptance over the trials that are neither an engine error
    nor ``degenerate``, the safety sum with reground lines, the engine errors, the first failing trial per clause) and
    the rule's counts."""
    by = _steps_of(trials, steps)
    plain = [t for t in trials if not t38.forced_failure(t, by[_turn(t)])]
    failing = [t for t in plain if not t38.accepted(t, by[_turn(t)])]
    errors = [t for t in trials if t38.engine_error(t, by[_turn(t)])]
    unsafe = [t for t in trials if sum(t38.safety_counts([t], by[_turn(t)]).values()) != 0]
    m = derive_arm(trials, steps)

    def tid(t: Mapping[str, Any]) -> str:
        return f"{t['task']}#{int(t['repeat'])}"

    return {
        "accepted": not failing,
        "safety": sum(t38.safety_counts(trials, steps).values()) + len(regrounds),
        "engine_errors": len(errors),
        "first_failing": {
            "acceptance": tid(failing[0]) if failing else None,
            "safety": tid(unsafe[0]) if unsafe else None,
            "engine": tid(errors[0]) if errors else None,
        },
        "n": m["n"],
        "success": m["success"],
        "false_claims": m["false_claims"],
        "over_actions": m["over_actions"],
        "w1": m["w1"],
    }


# --------------------------------------------------------------------------------------------- McNemar (report-only)

mcnemar_p = t38.mcnemar_p


def _section(pairs: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    b = sum(1 for a, x in pairs if a and not x)
    c = sum(1 for a, x in pairs if x and not a)
    return {"n": len(pairs), "b": b, "c": c, "p": float(mcnemar_p(b, c))}


def mcnemar_sections(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Pairs by ``(task, repeat)``: ``pooled`` over every task and ``per_task``; ``b`` A succeeded and B failed, ``c``
    the reverse; ``unpaired`` the trials without a partner (sorted, ``"<arm> <task>#<repeat>"``)."""
    by = _steps_of(trials, steps)
    ok = {_turn(t): t38.succeeded(t, by[_turn(t)]) for t in trials}
    pairs: dict[tuple[str, int], tuple[bool, bool]] = {}
    unpaired: list[str] = []
    for task, r in {(task, r) for (_, task, r) in ok}:
        a, b = ok.get(("A", task, r)), ok.get(("B", task, r))
        if a is None or b is None:
            unpaired.append(f"{'A' if a is not None else 'B'} {task}#{r}")
            continue
        pairs[(task, r)] = (a, b)

    def of(tasks: Sequence[str]) -> list[tuple[bool, bool]]:
        return [v for (task, _), v in sorted(pairs.items()) if task in tasks]

    return {
        "pooled": _section(of(TASK_IDS)),
        "per_task": {task: _section(of((task,))) for task in TASK_IDS},
        "unpaired": sorted(unpaired),
    }


# --------------------------------------------------------------------------------------------- C-5: the decision


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


def adoption(inputs: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    """Reading 6 past eligibility: ``not exercised``, ``adopt`` or ``no change``, with the first clause that failed."""
    a, b = inputs["A"], inputs["B"]
    w1_a, w1_b = int(a["w1"]), int(b["w1"])
    if w1_a < W1_EXERCISED_MIN:
        return NOT_EXERCISED, f"W1(A) {w1_a} < {W1_EXERCISED_MIN}"
    rate_a = int(a["success"]) / int(a["n"]) if int(a["n"]) else 0.0
    rate_b = int(b["success"]) / int(b["n"]) if int(b["n"]) else 0.0
    clauses = (
        ("W1", w1_b <= W1_RATIO * w1_a + EPS),
        ("success", rate_b >= rate_a - SUCCESS_MARGIN - EPS),
        (
            "false claims",
            int(b["false_claims"]) <= int(a["false_claims"]) + FALSE_CLAIM_MARGIN + EPS,
        ),
        (
            "over-actions",
            int(b["over_actions"]) <= int(a["over_actions"]) + OVER_ACTION_MARGIN + EPS,
        ),
    )
    for name, held in clauses:
        if not held:
            return NO_CHANGE, name
    return ADOPT, "every clause held"


def decide_with_reason(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    p1_passed: bool,
    void: bool = False,
    stopped: str | None = None,
    configuration_findings: Sequence[str] = (),
) -> tuple[str, str]:
    """C-5 (reading 6): P1 failed -> inconclusive; a configuration finding -> void (``configuration: <first>``); a probe
    void -> void; stopped -> incomplete; an ineligible arm -> inconclusive (arm, clause, first failing trial); then
    ``adoption``."""
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
    return adoption(inputs)


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
        pytest.skip(f"{ENV} is unset: the T-0137 A/B has not run (docs/tasks/T-0137.md C-6)")
    d = Path(where)
    if not (d / "report.json").is_file():
        pytest.skip(f"{d}/report.json does not exist")
    return d


@pytest.fixture(scope="module")
def run(run_dir: Path) -> dict[str, Any]:
    return {
        "report": json.loads((run_dir / "report.json").read_text()),
        "report_md": (run_dir / "report.md").read_text()
        if (run_dir / "report.md").is_file()
        else "",
        "manifest": json.loads((run_dir / "manifest.json").read_text()),
        "probes": json.loads((run_dir / "probes.json").read_text()),
        "trials": _jsonl(run_dir / "trials.jsonl"),
        "steps": _jsonl(run_dir / "steps.jsonl"),
        "regrounds": _jsonl(run_dir / "regrounds.jsonl"),
        "bodies": _jsonl(run_dir / "bodies.jsonl"),
    }


def _findings(run: Mapping[str, Any]) -> list[str]:
    return configuration_findings(
        run["trials"],
        run["steps"],
        run["regrounds"],
        run["bodies"],
        _manifest_arms(run),
        guidance_sha256=(run.get("manifest") or {}).get("guidance_sha256"),
    )


def _inputs(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        arm: decision_inputs(
            _of(run["trials"], arm), _of(run["steps"], arm), _of(run["regrounds"], arm)
        )
        for arm in LABELS
    }


def test_the_run_is_complete(run: dict[str, Any]) -> None:
    """C-1 completeness: the manifest's arm rows are C-1's two rows over its recorded live default, its arm-table sha
    is theirs, the suite-table sha is the pin, the order is the 200-turn seed-0 joint order and the trials ran in it."""
    manifest = run["manifest"]
    want_rows = arm_rows(manifest.get("live_default") or {})
    assert _manifest_arms(run) == want_rows, (_manifest_arms(run), want_rows)
    assert manifest.get("arm_table_sha256") == arm_table_sha(want_rows)
    assert manifest.get("suite_table_sha256") == SUITE_TABLE_SHA256
    assert [tuple(x) for x in manifest.get("order") or []] == order(0), "C-1: the manifest's order"
    got = [(t["arm"], t["task"], int(t["repeat"])) for t in run["trials"]]
    if run["report"].get("stopped"):
        assert got == order(0)[: len(got)], "C-1: a stopped run is a prefix of the seed-0 order"
        pytest.skip(f"the run stopped early: {run['report']['stopped']}")
    assert got == order(0), "C-1: the trials ran in the seed-0 joint order"


def test_stopped_re_derived(run: dict[str, Any]) -> None:
    assert bool(run["report"].get("stopped")) == (len(run["trials"]) < TOTAL_TURNS)


def test_live_default_recorded(run: dict[str, Any]) -> None:
    """C-1: the manifest records a well-formed live default with the guard off, and every trial that ran a session ran
    at it (its five settings, its overview frame and its T-0136 label in both arms)."""
    recorded = run["manifest"].get("live_default") or {}
    assert live_default_findings(recorded) == [], (recorded, live_default_findings(recorded))
    for t in run["trials"]:
        if t.get("scanned_requests"):
            keys = (*DEFAULT_KEYS, *FOLLOWED_KEYS)
            got = {k: t.get(k) for k in keys}
            assert got == {k: recorded[k] for k in keys}, (_tid(t), got, recorded)


@pytest.mark.parametrize("arm", LABELS)
def test_arm_metrics_re_derived(run: dict[str, Any], arm: str) -> None:
    """C-3: every per-arm count, W1 and the mechanism counts included, re-derived from the jsonl files equals
    report.json's."""
    want = derive_arm(_of(run["trials"], arm), _of(run["steps"], arm))
    got = run["report"]["arms"][arm]
    for key, value in want.items():
        if key == "success_ci":
            if value is None or got[key] is None:
                assert got[key] is value is None, (key, got[key], value)
            else:
                assert all(abs(x - y) < 1e-6 for x, y in zip(got[key], value, strict=True)), key
        elif key == "per_task_ci":
            for task, ci in value.items():
                g = got[key][task]
                assert (g is None) == (ci is None), (task, g, ci)
                if ci is not None:
                    assert all(abs(x - y) < 1e-6 for x, y in zip(g, ci, strict=True)), (task, g)
        elif key == "mean_steps":
            assert (got[key] is None) == (value is None) and (
                value is None or abs(float(got[key]) - value) < 1e-9
            ), (key, got[key], value)
        else:
            assert got[key] == value, (arm, key, got[key], value)


@pytest.mark.parametrize("arm", LABELS)
def test_safety_and_latency_re_derived(run: dict[str, Any], arm: str) -> None:
    """C-3: T-0151's safety counts (all 0) and the step latency, re-derived per arm."""
    trials, steps = _of(run["trials"], arm), _of(run["steps"], arm)
    want = t38.safety_counts(trials, steps)
    assert run["report"]["arms"][arm]["safety"] == want, (arm, run["report"]["arms"][arm]["safety"])
    assert all(v == 0 for v in want.values()), (arm, want)
    assert run["report"]["arms"][arm]["latency"] == t38.latency(trials, steps)


def test_configuration_findings_re_derived(run: dict[str, Any]) -> None:
    """C-4: the findings re-derived from the records, the manifest's arm rows and its guidance sha equal
    report.json's (the guard's stops and facts recomputed by the ported rule among them)."""
    want = _findings(run)
    assert sorted(run["report"]["configuration_findings"]) == sorted(want), (
        run["report"]["configuration_findings"],
        want,
    )


def test_probes_re_derived(run: dict[str, Any]) -> None:
    """C-2: each probe re-derived from probes.json; each recorded ``passed`` and the report's ``void`` agree; a failed
    P1 is ``inconclusive``, any other failed probe ``void``."""
    probes = run["probes"]
    v = probe_verdicts(probes)
    for p in ("p1", "p2", "p3", "ps1", "ps2", "ps3"):
        assert bool(probes[p].get("passed")) is v[p], (p, probes[p], v[p])
    assert bool(run["report"].get("void")) is v["void"]
    if not v["p1"]:
        assert run["report"]["decision"]["label"] == INCONCLUSIVE
    elif v["void"]:
        assert run["report"]["decision"]["label"] == VOID


def test_mcnemar_re_derived(run: dict[str, Any]) -> None:
    want = mcnemar_sections(run["trials"], run["steps"])
    got = run["report"]["mcnemar"]
    for k in ("n", "b", "c"):
        assert got["pooled"][k] == want["pooled"][k], (k, got["pooled"], want["pooled"])
    assert abs(float(got["pooled"]["p"]) - want["pooled"]["p"]) < 1e-12
    for task in TASK_IDS:
        for k in ("n", "b", "c"):
            assert got["per_task"][task][k] == want["per_task"][task][k], (task, k)
        assert abs(float(got["per_task"][task]["p"]) - want["per_task"][task]["p"]) < 1e-12
    assert sorted(got["unpaired"]) == want["unpaired"], got["unpaired"]


@pytest.mark.parametrize("arm", LABELS)
def test_decision_inputs_re_derived(run: dict[str, Any], arm: str) -> None:
    assert run["report"]["decision"]["inputs"][arm] == _inputs(run)[arm], (
        arm,
        run["report"]["decision"],
    )


def test_decision_re_derived(run: dict[str, Any]) -> None:
    """C-5: the label and its reason from re-derived inputs, probes, findings and the trial count."""
    v = probe_verdicts(run["probes"])
    want = decide_with_reason(
        _inputs(run),
        p1_passed=v["p1"],
        void=v["void"],
        stopped=run["report"].get("stopped") if len(run["trials"]) < TOTAL_TURNS else None,
        configuration_findings=_findings(run),
    )
    d = run["report"]["decision"]
    assert (d["label"], d["reason"]) == want, (d, want)


def test_report_states_w1_and_not_exercised(run: dict[str, Any]) -> None:
    """C-5 "stated in the report": both arms' W1 and that ``not exercised`` is a legitimate result."""
    md = run["report_md"]
    assert "not exercised" in md and "W1" in md, md[:2000]
