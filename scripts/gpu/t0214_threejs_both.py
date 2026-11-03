"""T-0214 -- one locked, report-only GPU arm on the three.js editor tasks M1-M5: "both", HEAD's live defaults plus
T-0136's action-outcome clause and T-0131/T-0133's ReGround pass 2 (crop 280 from "default" at the 1120 main budget),
turned on together; M1-M5 x n = 3 = 15 typed turns through real ``Session`` objects on the vendored loopback copy,
web mode with a deny-all resolver and connector, search off. ``/launch`` ONLY.

    uv run python scripts/gpu/t0214_threejs_both.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing.**

A1: T-0212's harness (``scripts/gpu/t0212_threejs_arms.py``) is imported read-only by path as ``R`` under a fixed
``sys.modules`` name; ``M``, ``D``, ``G`` are its module objects. Every turn runs through ``R.M._one_turn(run,
task_id, repeat)`` unchanged, the probes are T-0212's (``M.run_probes``, ``M.probe_pm1``, ``M.probe_pm2``,
``M.probe_verdicts``), and the per-turn metrics are ``R``'s. No attribute of ``R``, ``M``, ``D`` or ``G`` is assigned.
T-0212's harness bytes are pinned against the newest ``Merge T-0212:`` merge, T-0210/T-0205's against
``Merge T-0210-c:`` (``R.pinned_changes``); either off its pin is a refusal.

What cannot be reused as-is (it hard-codes T-0212's three arms or limits) has a small T-0214 equivalent here:
:func:`both_context` (``R.arm_context`` refuses "both"), :func:`both_keywords` / :func:`both_grounding` /
:func:`pa_check`, :func:`order`, :func:`budget_stop`, :func:`_run_all`, :func:`both_evidence` (A7: the reground and
outcome rows together, no negative rows), :func:`turn_row` / :func:`finish_turn` (``R``'s, with ``evidence`` from
:func:`both_evidence`), :func:`build_report`, :func:`render_markdown`, :func:`reading`, :func:`reading_mark`.

A10: T-0212's three arms are quoted beside "both" as constants (``ANCHOR``) from its run at 5b87b723, an
**unpaired** anchor; ``logs/`` is never read at run time. Regime drift is HEAD's base keyword dict against T-0212's
recorded one (``ANCHOR_BASE_KEYWORDS``, whose canonical-JSON sha256 is ``ANCHOR_BASE_KEYWORDS_SHA256``).

Output (``--out-dir``, default ``logs/t0214-threejs-both-<sha7>/``): ``order.json`` (before the probes),
``manifest.json``, ``probes.json``, ``trials.jsonl``, ``steps.jsonl``, ``thoughts.jsonl``, ``turns.jsonl``,
``sessions/*.jsonl``, ``report.json``, ``report.md``. Exit: 0 measured; 1 no engine or stopped early; 2 refused;
3 void.

Regime (state it beside every number): profile lean, canvas length 256, 48 denoising steps, thinking on; vLLM
0.29.0, NVFP4 rev from ``services/versions.lock``, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16, one boot;
HEAD's live web-mode defaults through cli (recorded), both levers rebinding cli's defaults; search off; headless
Chromium 1280x800 (WebGL2 through SwiftShader), deny-all resolver and connector; three.js r179; typed turns, fake
STT/TTS, no headset or speakers; 15 turns in three blocks, order seed 0, 300 s per turn, no engine seed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

from local_astra.pages import threejs_tasks

#: A6: every time is measured from process start (probes included)
PROCESS_START_S: float = time.monotonic()

REPO = Path(__file__).resolve().parents[2]
#: T-0212's harness, read-only (A1: imported, never edited)
T0212_REL = "scripts/gpu/t0212_threejs_arms.py"
T0212_PATH: Path = REPO / T0212_REL
#: the fixed ``sys.modules`` name T-0212's harness is loaded under
T0212_MODULE_NAME = "_t0212_threejs_arms_for_t0214"


def _load_t0212() -> ModuleType:
    """T-0212's harness, imported read-only by path under ``T0212_MODULE_NAME``. Importing it opens no socket."""
    found = sys.modules.get(T0212_MODULE_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0212_MODULE_NAME, T0212_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0212_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0212_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0212_MODULE_NAME, None)
        raise
    return module


R: ModuleType = _load_t0212()
#: T-0212's chain, the same module objects
M: ModuleType = R.M
D: ModuleType = R.D
G: ModuleType = R.G
T: ModuleType = R.T
P: ModuleType = R.P
#: the tree is read through this name; every function looks it up at call time
_git = G._git

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3
ARM = "both"
ARMS: tuple[str, ...] = (ARM,)
#: A2: the two cli defaults "both" rebinds to ``True`` (in this order)
CLI_DEFAULTS: tuple[str, ...] = tuple(R.CLI_DEFAULTS)
TASK_IDS: tuple[str, ...] = tuple(R.TASK_IDS)
#: A5: n = 3 per task, three blocks
REPEATS = 3
BLOCKS = 3
ORDER_SEED = 0
TOTAL_TURNS = len(TASK_IDS) * BLOCKS
TURN_TIMEOUT_S = int(M.TURN_TIMEOUT_S)
#: A6: the wrapper's ``timeout -k 60s 25m``
HARD_LIMIT_S = 25 * 60
RUN_BUDGET_S = 18 * 60
TURN_WORST_S = 480
REPORT_S = 60
TURN_WORST_NOTE = R.TURN_WORST_NOTE
#: A3: the pass-2 crop budget and source at the main budget (T-0212's reground arm)
CROP_BUDGET = int(R.CROP_BUDGET)
CROP_SOURCE = str(R.CROP_SOURCE)
MAIN_BUDGET = int(R.MAIN_BUDGET)
#: A4: the merge subject (exact prefix) read from ``git log --merges``
MERGE_T0212 = "Merge T-0212:"
VOID, INCOMPLETE, MEASURED = R.VOID, R.INCOMPLETE, R.MEASURED
POINTER = tuple(R.POINTER)
SAFETY_EVENTS: tuple[str, ...] = tuple(R.SAFETY_EVENTS)
SAFETY_NAMES: tuple[str, ...] = (
    *SAFETY_EVENTS,
    "over_action",
    "traps",
    "malformed_state",
    "state_errors",
)

#: A10: T-0212's run at 5b87b723 (``logs/t0212-threejs-arms-5b87b72/report.json``), quoted, never re-measured and
#: never read at run time
ANCHOR_RUN = "logs/t0212-threejs-arms-5b87b72"
ANCHOR_COMMIT = "5b87b723cdab25dca5a50ce400fa773202fd28e0"
ANCHOR_ARMS: tuple[str, ...] = ("base", "outcome", "reground")
ANCHOR: Mapping[str, Mapping[str, Any]] = {
    "base": {
        "success": (0, 15),
        "click_misses": (35, 96),
        "repeat_guard_refusals": 18,
        "wall_p50_s": 20.2,
    },
    "outcome": {
        "success": (0, 15),
        "click_misses": (23, 69),
        "repeat_guard_refusals": 18,
        "wall_p50_s": 12.13,
    },
    "reground": {
        "success": (1, 15),
        "click_misses": (10, 74),
        "repeat_guard_refusals": 5,
        "wall_p50_s": 13.36,
    },
}
#: A10: ``arm_keywords.base`` of that run's ``manifest.json`` (already ``_jsonable``: the guidance by sha256)
ANCHOR_BASE_KEYWORDS: Mapping[str, Any] = {
    "tool_guidance": {
        "sha256": "de3a3f9859bb8930d1c8779928994b2aa9bc3c1e8c5a9c311f2b787791b8cac3",
        "chars": 815,
    },
    "visual_tokens": 1120,
    "reground": False,
    "degeneracy_guard": True,
    "live_feedback": {"screen_label": True, "page_extent": False},
    "recover_thought_calls": True,
    "overview_frame": True,
    "repeat_guard": True,
    "screen_reminder": True,
    "completion_check": True,
    "reask_announced_calls": True,
    "page_fact": True,
}
#: A10: sha256 of :func:`canonical_json` of that dict (computed from the manifest file, 2026-10-31)
ANCHOR_BASE_KEYWORDS_SHA256 = "67d1c2779725b7613ca15f564bd5d60ebdb87fa9e2e4ef8ea98748832ad76968"
UNPAIRED_NOTE = (
    "unpaired anchor: T-0212 at 5b87b723, a different run and HEAD, the engine unseeded; T-0137 C measured about 10 "
    "points of run-to-run noise between identical arms; n = 15; a paired A/B decides"
)
WORTH = "worth a paired follow-up"
NO_SIGN = "no sign beyond ReGround alone"
MIXED = "mixed"
WORTH_AT = Fraction(4, 15)
NO_SIGN_AT = Fraction(1, 15)
DRIFT_MARK = "provisional (regime drift)"
CLICK_MISS_CAVEAT = R.CLICK_MISS_CAVEAT
NEAR_MISS_NOTE = R.NEAR_MISS_NOTE
GROUNDING_SOURCE_NOTE = R.GROUNDING_SOURCE_NOTE
REPEAT_GUARD_FOOTNOTE = (
    "both arm: the action-outcome clause is on, so the repeat guard's no-change input is the outcome detector "
    f"(T-0137), not B8's label; T-0212's footnote: {R.REPEAT_GUARD_FOOTNOTE}"
)
GATE_FILES = tuple(R.GATE_FILES)
SCOPE_NOTE = (
    "one arm (action outcome and ReGround together) on a pinned loopback copy of the three.js editor (r179), five "
    "tasks, n = 3 per task: wide intervals; report-only, nothing adopts on it."
)


def _say(msg: str) -> None:
    print(f"t0214_threejs_both: {msg}", file=sys.stderr, flush=True)


def _now() -> float:
    """The clock the budget reads (a seam for the unit module's fake clock)."""
    return time.monotonic()


# =====================================================================================================
# A5: the order (pure)
# =====================================================================================================


def order(seed: int = ORDER_SEED) -> list[tuple[str, str, int]]:
    """A5: blocks ``r = 0, 1, 2``; each holds M1-M5 once as ``(task, "both", r)``, shuffled in turn with one
    ``random.Random(seed)``."""
    rng = random.Random(seed)
    out: list[tuple[str, str, int]] = []
    for r in range(BLOCKS):
        block = [(t, ARM, r) for t in TASK_IDS]
        rng.shuffle(block)
        out += block
    return out


# =====================================================================================================
# A2/A3: both levers, read through cli
# =====================================================================================================

#: open while :func:`both_context` is (at most one: nesting is refused)
_ACTIVE: list[str] = []


@contextlib.contextmanager
def both_context() -> Iterator[None]:
    """A2: for one turn, ``cli.LIVE_DEFAULT_ACTION_OUTCOME`` and ``cli.LIVE_DEFAULT_REGROUND`` are ``True``; both
    are restored in a ``finally`` to the values they held, also when the body raises and when a rebinding itself
    fails. Refused (``RuntimeError``) inside itself and while T-0212's ``R._ACTIVE`` is non-empty (read only)."""
    from local_astra import cli

    if _ACTIVE:
        raise RuntimeError("both_context() inside both_context(): nesting is refused")
    if R._ACTIVE:
        raise RuntimeError(
            f"both_context() inside T-0212's arm_context({R._ACTIVE[-1]!r}): nesting is refused"
        )
    _ACTIVE.append(ARM)
    try:
        held = {name: getattr(cli, name) for name in CLI_DEFAULTS}
        try:
            for name in CLI_DEFAULTS:
                setattr(cli, name, True)
            yield
        finally:
            failed: BaseException | None = None
            for name, value in held.items():
                try:
                    setattr(cli, name, value)
                except (
                    BaseException
                ) as exc:  # restore every name before re-raising the first failure
                    failed = failed or exc
            if failed is not None:
                raise failed
    finally:
        _ACTIVE.pop()


def cli_defaults() -> dict[str, Any]:
    """N2: the two cli defaults now."""
    from local_astra import cli

    return {name: getattr(cli, name) for name in CLI_DEFAULTS}


def both_keywords(profile: Any) -> dict[str, Any]:
    """A3: ``D.session_keywords(profile)`` built under :func:`both_context`."""
    with both_context():
        return dict(D.session_keywords(profile))


def _grounding_dict(g: Any) -> dict[str, Any]:
    return {
        "visual_tokens": g.visual_tokens,
        "visual_tokens_from": g.visual_tokens_from,
        "reground": g.reground,
        "reground_from": g.reground_from,
        "reground_visual_tokens": g.reground_visual_tokens,
        "reground_visual_tokens_from": g.reground_visual_tokens_from,
    }


def base_grounding(profile: Any) -> dict[str, Any]:
    """``cli.grounding_of(profile, D._live_args(), "live")`` outside any context, as a dict."""
    from local_astra import cli

    return _grounding_dict(cli.grounding_of(profile, D._live_args(), "live"))


def both_grounding(profile: Any) -> dict[str, Any]:
    """A3: ``cli.grounding_of(profile, D._live_args(), "live")`` under :func:`both_context`, as a dict."""
    from local_astra import cli

    with both_context():
        g = cli.grounding_of(profile, D._live_args(), "live")
    return _grounding_dict(g)


#: A3: the one diff "both" may have against base
EXPECTED_DIFF: Mapping[str, Any] = {
    "added": {"action_outcome": True},
    "removed": ["overview_frame"],
    "changed": {"reground": [False, True]},
}


def pa_problems(
    base: Mapping[str, Any],
    both: Mapping[str, Any],
    grounding: Mapping[str, Any],
    reground_arm: Mapping[str, Any] | None = None,
) -> list[str]:
    """A3 P-A: every way "both" differs from base other than its two levers (empty iff P-A holds). ``grounding`` is
    the both-context grounding; ``reground_arm`` (when given) is ``R.arm_keywords(profile, "reground")`` for the
    cross-check."""
    out: list[str] = []
    diff = R.keyword_diff(base, both)
    if diff != EXPECTED_DIFF:
        out.append(f"both - base is {diff}, not exactly {dict(EXPECTED_DIFF)}")
    if "overview_frame" in both:
        out.append("the both arm carries overview_frame (cli's H-T152-1 yield is missing)")
    if not isinstance(base.get("tool_guidance"), str) or both.get("tool_guidance") != base.get(
        "tool_guidance"
    ):
        out.append("tool_guidance differs between base and both")
    if base.get("action_outcome") or base.get("reground"):
        out.append("the base keywords already have a lever on")
    if not (
        grounding.get("reground") is True
        and grounding.get("reground_visual_tokens") == CROP_BUDGET
        and grounding.get("reground_visual_tokens_from") == CROP_SOURCE
        and grounding.get("visual_tokens") == MAIN_BUDGET
    ):
        out.append(
            f"the both arm's grounding is {dict(grounding)}, not ReGround on at {CROP_BUDGET} from "
            f"{CROP_SOURCE!r} with the main budget {MAIN_BUDGET}"
        )
    if reground_arm is not None:
        want = dict(reground_arm) | {"action_outcome": True}
        if dict(both) != want:
            out.append(
                f"both is not T-0212's reground arm plus action_outcome: {R.keyword_diff(want, both)}"
            )
    return out


def _sha(text: Any) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def pa_check(profile: Any) -> dict[str, Any]:
    """A3 P-A (no socket): base's and both's keywords (``R._jsonable``: the guidance by sha256), the diff, both
    groundings, the guidance sha256s, the cross-check against T-0212's reground arm, and ``problems`` (empty iff
    P-A holds). This is also the manifest's arm record."""
    base = dict(D.session_keywords(profile))
    both = both_keywords(profile)
    reground_arm = dict(R.arm_keywords(profile, "reground"))
    grounding = {"base": base_grounding(profile), "both": both_grounding(profile)}
    cross = reground_arm | {"action_outcome": True}
    return {
        "keywords": {"base": R._jsonable(base), "both": R._jsonable(both)},
        "diffs": {ARM: R._jsonable(R.keyword_diff(base, both))},
        "grounding": grounding,
        "tool_guidance_sha256": {
            "base": _sha(base.get("tool_guidance")),
            "both": _sha(both.get("tool_guidance")),
        },
        "cross_check": {
            "rule": 'both == R.arm_keywords(profile, "reground") | {"action_outcome": True}',
            "holds": both == cross,
            "diff": R._jsonable(R.keyword_diff(cross, both)),
        },
        "problems": pa_problems(base, both, grounding["both"], reground_arm),
    }


# =====================================================================================================
# A10: regime drift against T-0212's recorded base keywords (pure)
# =====================================================================================================


def canonical_json(value: Any) -> str:
    """The canonical JSON of a keyword dict: ``R._jsonable`` then ``json.dumps(..., sort_keys=True)``."""
    return json.dumps(R._jsonable(value), sort_keys=True)


def keywords_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def regime_drift(base_keywords: Mapping[str, Any] | None) -> dict[str, Any]:
    """A10: HEAD's base keyword dict (as the manifest records it) against T-0212's. ``drift`` is ``None`` when no
    base dict is recorded (not checked)."""
    if base_keywords is None:
        return {"checked": False, "drift": None, "sha256": None, "diff": None}
    sha = keywords_sha256(base_keywords)
    return {
        "checked": True,
        "drift": sha != ANCHOR_BASE_KEYWORDS_SHA256,
        "sha256": sha,
        "anchor_sha256": ANCHOR_BASE_KEYWORDS_SHA256,
        "diff": R._jsonable(
            R.keyword_diff(R._jsonable(ANCHOR_BASE_KEYWORDS), R._jsonable(base_keywords))
        ),
    }


# =====================================================================================================
# A7: arm evidence per turn (pure)
# =====================================================================================================


def both_evidence(
    records: Sequence[Mapping[str, Any]], extras: Mapping[str, Any] | None
) -> list[str]:
    """A7 P-E: what T-0212 A7 asks of a reground turn and of an outcome turn, together (empty when it holds). The
    turn-line rows apply only where the turn wrote a line (``extras`` not ``None``; R-3: a not-run turn has none);
    the log row always applies. T-0212's negative rows (no reground records, no outcome stem) are not applied."""
    out: list[str] = []
    ordered = R._seq_sorted(records)
    regrounds = [r for r in ordered if r.get("kind") == "tool.reground"]
    acted = [
        r
        for r in ordered
        if r.get("kind") == "tool.call"
        and r["data"].get("status") == "executed"
        and r["data"].get("name") in POINTER
    ]
    if acted and not regrounds:
        out.append("both turn: an executed click_at/type_at and no tool.reground record")
    if extras is not None:
        if "reground" not in extras:
            out.append("both turn: the turn line has no extras.reground")
        g = extras.get("grounding")
        if not isinstance(g, Mapping) or g.get("reground_visual_tokens") != CROP_BUDGET:
            out.append(f"both turn: extras.grounding.reground_visual_tokens is not {CROP_BUDGET}")
        if "action_outcome" not in extras:
            out.append("both turn: the turn line has no extras.action_outcome")
        if R.overview_shown(extras) > 0:
            out.append(
                f"both turn: extras.overview_frame.shown is {R.overview_shown(extras)} (expected absent or 0)"
            )
    return out


# =====================================================================================================
# Per turn (pure)
# =====================================================================================================


def turn_row(
    trial: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    extras: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """``R.turn_row``'s counts, with ``evidence`` from :func:`both_evidence` (``R.arm_evidence`` knows no "both")."""
    row = dict(R.turn_row(trial, steps, records, extras))
    row["evidence"] = both_evidence(records, extras)
    return row


def reading(summary: Mapping[str, Any]) -> str:
    """A10, pre-registered (not a bar), exact Fractions over the arm's success rate k/n: ``WORTH`` iff success >=
    4/15, zero safety events and ``over_action`` 0; ``NO_SIGN`` iff success <= 1/15; else ``MIXED``."""
    n = int(summary["n"])
    rate = Fraction(int(summary["success"]), n) if n else Fraction(0)
    safety = R.safety_total(summary)
    over = int(summary["safety"].get("over_action", 0))
    if rate >= WORTH_AT and safety == 0 and over == 0:
        return WORTH
    if rate <= NO_SIGN_AT:
        return NO_SIGN
    return MIXED


def reading_mark(
    label: str, errored_turns: int, safety_events: int, drift: bool | None
) -> str | None:
    """A10 marks joined with ``; ``: ``R.reading_mark``'s (run label, errored N; safety supersedes every mark) plus
    ``provisional (regime drift)`` when HEAD's base keywords differ from T-0212's."""
    mark = R.reading_mark(label, errored_turns, safety_events)
    if safety_events:
        return str(mark)
    marks = [str(mark)] if mark else []
    if drift:
        marks.append(DRIFT_MARK)
    return "; ".join(marks) if marks else None


def reading_line(text: str, mark: str | None) -> str:
    line = f"{ARM}: {text}"
    if mark:
        line += f" [{mark}]"
    return line


def regime_of(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """A9: ``R.regime_of`` (T-0212's ``REGIME_FIELDS`` plus ``arm_keywords`` and ``arm_diffs``)."""
    return dict(R.regime_of(manifest))


def regime_missing(regime: Mapping[str, Any]) -> list[str]:
    """A9 / T-0212 R-5: the regime fields whose value is ``None``."""
    return sorted(k for k, v in regime.items() if v is None)


def anchor_table(both: Mapping[str, Any]) -> dict[str, Any]:
    """A10: T-0212's three arms (quoted constants) beside this run's "both"."""
    rows: dict[str, Any] = {
        arm: {
            "success": list(ANCHOR[arm]["success"]),
            "click_misses": list(ANCHOR[arm]["click_misses"]),
            "repeat_guard_refusals": ANCHOR[arm]["repeat_guard_refusals"],
            "wall_p50_s": ANCHOR[arm]["wall_p50_s"],
            "source": f"{ANCHOR_RUN} at {ANCHOR_COMMIT[:8]} (quoted)",
        }
        for arm in ANCHOR_ARMS
    }
    rows[ARM] = {
        "success": [both["success"], both["n"]],
        "click_misses": [both["click_misses"], both["labelled_clicks"]],
        "repeat_guard_refusals": both["repeat_guard_refusals"],
        "wall_p50_s": both["wall_s"]["p50"],
        "source": "this run",
    }
    return {"run": ANCHOR_RUN, "commit": ANCHOR_COMMIT, "arms": rows, "note": UNPAIRED_NOTE}


# =====================================================================================================
# A9: the report (pure over the run dir's files)
# =====================================================================================================


def build_report(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    records_by_key: Mapping[tuple[str, str, int], Sequence[Mapping[str, Any]]],
    probes: Mapping[str, Any],
    stopped: str | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A8-A10 over the run's trials, steps and each turn's session-log records (keyed ``(task, arm, repeat)``)."""
    by_turn: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for s in steps:
        by_turn.setdefault(R._key(s), []).append(s)
    rows = []
    for t in trials:
        recs = list(records_by_key.get(R._key(t), []))
        rows.append(turn_row(t, by_turn.get(R._key(t), []), recs, R.turn_extras(recs)))
    summary = R.arm_summary(rows)
    per_task = {tid: R.arm_summary([r for r in rows if r["task"] == tid]) for tid in TASK_IDS}
    violations = [
        f"{r['task']}/{r['arm']}/r{r['repeat']}: {v}" for r in rows for v in r["evidence"]
    ]
    probe_void = bool(M.probe_verdicts(probes)["void"])
    label = R.report_label(probe_void, len(violations), stopped)
    safety = R.safety_total(summary)
    mf = manifest or {}
    base_kw = (mf.get("arm_keywords") or {}).get("base")
    drift = regime_drift(base_kw if isinstance(base_kw, Mapping) else None)
    text = reading(summary)
    mark = reading_mark(label, int(summary["errored"]), safety, drift["drift"])
    audit = []
    for t in sorted(trials, key=R._key):
        mine = sorted(by_turn.get(R._key(t), []), key=lambda s: int(s.get("step") or 0))
        audit.append(
            {
                "task": str(t.get("task")),
                "arm": str(t.get("arm")),
                "repeat": int(t.get("repeat") or 0),
                "success": R.succeeded(t, mine),
                "errored": R.errored(t, mine),
                "tool_loop_end": t.get("tool_loop_end"),
                "exception": t.get("exception"),
                "voiced": [str(x) for x in t.get("voiced") or []],
                "shown": str(t.get("shown") or ""),
                "typed": list(M._typed_texts(mine)),
            }
        )
    regime = regime_of(manifest)
    return {
        "label": label,
        "stopped": stopped,
        "turns": len(trials),
        "regime": regime,
        "regime_none": regime_missing(regime),
        "probes": M.probe_verdicts(probes),
        "violations": violations,
        "per_arm": {ARM: summary},
        "per_task": {ARM: per_task},
        "reading": {
            "reading": text,
            "mark": mark,
            "line": reading_line(text, mark),
            "rule": (
                f"{WORTH!r} iff success >= 4/15, zero safety events and over_action 0; {NO_SIGN!r} iff success <= "
                f"1/15; else {MIXED!r} (exact Fractions of k/n)"
            ),
        },
        "regime_drift": drift,
        "anchor": anchor_table(summary),
        "notes": {
            "click_miss": CLICK_MISS_CAVEAT,
            "near_miss": NEAR_MISS_NOTE,
            "grounding_source": GROUNDING_SOURCE_NOTE,
            "repeat_guard": REPEAT_GUARD_FOOTNOTE,
            "scope": SCOPE_NOTE,
            "unpaired": UNPAIRED_NOTE,
        },
        "safety_events": list(SAFETY_EVENTS),
        "cli_after": dict(mf.get("cli_after") or {}),
        "audit": audit,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, Mapping | list):
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return str(value)


def _fence(text: str) -> list[str]:
    return ["```text", text, "```"]


def _json_block(value: Any) -> list[str]:
    return ["```json", json.dumps(value, indent=1, default=str, ensure_ascii=False), "```"]


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """``report.md``: the regime, the label, the reading line, the unpaired anchor table, the arm's and per-task
    tables, the safety table, and every voiced, shown and typed text verbatim."""
    reg = report.get("regime") or {}
    mf = manifest or {}
    a = report["per_arm"][ARM]
    cc = a["completion_check"]
    rg = a["reground"]
    lines = [
        "# T-0214: three.js M1-M5, one arm (action outcome and ReGround together), report-only",
        "",
        f"- label: **{report['label']}**; stopped: {_fmt(report.get('stopped'))}; turns: {report.get('turns')}",
        f"- probes: {json.dumps(report.get('probes'))}",
        f"- scope: {SCOPE_NOTE}",
        "",
        "## Regime",
        "",
    ]
    lines += [f"- {k}: {_fmt(v)}" for k, v in reg.items()]
    if report.get("regime_none"):
        lines.append(f"- regime fields with no value: {report['regime_none']}")
    if mf:
        lines += [
            "",
            "Arm keywords (``D.session_keywords`` outside any context and under ``both_context``; the guidance by "
            "sha256):",
            "",
            *_json_block(mf.get("arm_keywords")),
            "",
            "Diff against base:",
            "",
            *_json_block(mf.get("arm_diffs")),
            "",
            "Grounding (base, both):",
            "",
            *_json_block(mf.get("arm_grounding")),
            "",
            f"- cross-check: {_fmt(mf.get('cross_check'))}",
        ]
    lines += [
        "",
        f"- {GROUNDING_SOURCE_NOTE}",
        "",
        "## Reading (A10, pre-registered; not a bar)",
        "",
        f"- {report['reading']['line']}",
        f"- rule: {report['reading']['rule']}",
        "",
        "## Beside T-0212's three arms",
        "",
        "| arm | success | click misses / labelled | repeat-guard refusals | wall p50 s | source |",
        "|---|---|---|---|---|---|",
    ]
    for arm, row in report["anchor"]["arms"].items():
        lines.append(
            f"| {arm} | {row['success'][0]}/{row['success'][1]} | {row['click_misses'][0]}/"
            f"{row['click_misses'][1]} | {row['repeat_guard_refusals']} | {_fmt(row['wall_p50_s'])} | "
            f"{row['source']} |"
        )
    drift = report.get("regime_drift") or {}
    lines += [
        "",
        "---",
        "",
        UNPAIRED_NOTE,
        "",
        f"- regime drift (HEAD's base keywords vs T-0212's): checked {drift.get('checked')}, drift "
        f"{_fmt(drift.get('drift'))}, sha256 {_fmt(drift.get('sha256'))} vs {ANCHOR_BASE_KEYWORDS_SHA256}; diff "
        f"{_fmt(drift.get('diff'))}",
        "",
        "## The arm",
        "",
        "| arm | success (Wilson 95 %) | errored | steps p50/max | wall s p50/max | unchanged labels / labels | "
        "repeat-guard refusals | completion fired / outcomes / success after | body-focus | clicks | misses / "
        "labelled (rate) | unlabelled | re-aims |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| {ARM} | {a['success']}/{a['n']} {a['wilson95']} | {a['errored']} | {a['steps']['p50']}/"
        f"{a['steps']['max']} | {a['wall_s']['p50']}/{a['wall_s']['max']} | {a['unchanged_labels']}/"
        f"{a['labels']} | {a['repeat_guard_refusals']} | {cc['fired_turns']} / {cc['outcomes']} / "
        f"{cc['success_after_fire']} | {a['body_focus_refusals']} | {a['clicks']} | {a['click_misses']}/"
        f"{a['labelled_clicks']} ({_fmt(a['miss_rate'])}) | {a['unlabelled_clicks']} | {a['reaims']} |",
        "",
        f"- {REPEAT_GUARD_FOOTNOTE}",
        f"- outcome clause counts by kind ('none' is \"no visible change\"): {a['action_outcome']}",
        f"- tool.reground outcomes {rg['outcomes']}; requests {rg['requests']}; ok rate {_fmt(rg['ok_rate'])}; "
        f"pass-2 shift (model units, ok records) p50/max {rg['shift']['p50']}/{rg['shift']['max']}; pass-2 ms "
        f"p50/max {rg['ms']['p50']}/{rg['ms']['max']}; miss rate moved {_fmt(rg['miss_rate_moved'])} "
        f"{rg['split']['moved']}, not moved {_fmt(rg['miss_rate_not_moved'])} {rg['split']['not_moved']}",
        f"- budgets sent: {a['budgets_sent']}",
        f"- turns with the overview frame shown: {a['overview_shown_turns']}",
        f"- {CLICK_MISS_CAVEAT}",
        f"- {NEAR_MISS_NOTE}",
        "",
        "## Per task (success k/3)",
        "",
        f"| task | {ARM} |",
        "|---|---|",
    ]
    for tid in TASK_IDS:
        t = report["per_task"][ARM][tid]
        lines.append(f"| {tid} | {t['success']}/{t['n']} |")
    lines += [
        "",
        "## SAFETY (each expected 0; safety events: " + ", ".join(SAFETY_EVENTS) + ")",
        "",
        "| arm | " + " | ".join(SAFETY_NAMES) + " |",
        "|---|" + "---|" * len(SAFETY_NAMES),
        f"| {ARM} | " + " | ".join(_fmt(a["safety"].get(n)) for n in SAFETY_NAMES) + " |",
        "",
        "## Arm evidence (A7)",
        "",
    ]
    lines += [f"- {v}" for v in report.get("violations") or []] or ["- no violation"]
    lines += ["", f"- cli defaults after the run (N2): {report.get('cli_after')}", ""]
    lines += ["## Every voiced, shown and typed text, verbatim", ""]
    for x in report.get("audit") or []:
        lines.append(
            f"#### {x['task']}/{x['arm']}/r{x['repeat']}: success {x['success']}, errored {x['errored']}, end "
            f"{_fmt(x['tool_loop_end'])}, exception {_fmt(x['exception'])}"
        )
        for v in x["voiced"]:
            lines += ["voiced:", *_fence(v)]
        if x["shown"]:
            lines += ["shown:", *_fence(x["shown"])]
        for v in x["typed"]:
            lines += ["typed:", *_fence(v)]
        lines.append("")
    return "\n".join(lines) + "\n"


# =====================================================================================================
# The run dir's writers
# =====================================================================================================


def write_report(
    out: Path, probes: Mapping[str, Any], manifest: Mapping[str, Any], stopped: str | None = None
) -> dict[str, Any]:
    trials = G._read_jsonl(out / "trials.jsonl")
    report = build_report(
        trials,
        G._read_jsonl(out / "steps.jsonl"),
        R.records_by_key(out, trials),
        probes,
        stopped,
        manifest,
    )
    G._write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report, manifest), encoding="utf-8")
    return report


def finish_turn(
    out: Path, task: str, arm: str, repeat: int, trial: dict[str, Any], steps: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``R.finish_turn`` (``task``/``arm``/``repeat`` on every line, the turn line's lever extras, the cli defaults
    after the turn) for arm "both" only, with ``evidence`` from :func:`both_evidence` over the turn's session log."""
    if arm != ARM:
        raise ValueError(f"unknown arm {arm!r} (T-0214 runs only {ARM!r})")
    trial, steps = R.finish_turn(out, task, arm, repeat, trial, steps)
    name = trial.get("session_log")
    path = out / "sessions" / str(name)
    recs = R._read_records(path) if isinstance(name, str) and name and path.is_file() else []
    trial["evidence"] = both_evidence(recs, R.turn_extras(recs))
    trial["cli_after"] = cli_defaults()
    return trial, steps


def budget_stop(elapsed: float, errored_run: int, index: int, last: str = "") -> str | None:
    """A6, checked before each turn: the reason the run stops, or ``None`` when the turn may start."""
    if errored_run >= G.ERRORED_STOP:
        return f"{errored_run} consecutive errored turns (last: {last}; at turn {index})"
    if elapsed + TURN_WORST_S + REPORT_S > HARD_LIMIT_S:
        return (
            f"a worst-case turn ({TURN_WORST_S} s) and the report ({REPORT_S} s) no longer fit inside {HARD_LIMIT_S} s "
            f"from process start ({elapsed:.0f} s spent; at turn {index})"
        )
    if elapsed > RUN_BUDGET_S:
        return f"the run budget of {RUN_BUDGET_S} s is spent ({elapsed:.0f} s; at turn {index})"
    return None


async def _run_all(run: Any, turns: Sequence[tuple[str, str, int]]) -> None:
    """Every turn of ``turns`` under :func:`both_context`, through ``M._one_turn``; a stop (:func:`budget_stop`)
    sends nothing more and the caller still writes the report. A turn that raises is ``M._not_run``'s line."""
    errored_run = 0
    last = ""
    for index, (task, arm, repeat) in enumerate(turns):
        if arm != ARM:
            raise ValueError(f"unknown arm {arm!r} (T-0214 runs only {ARM!r})")
        started = _now()
        stop = budget_stop(started - PROCESS_START_S, errored_run, index, last)
        if stop is not None:
            run.stopped = stop
            run.manifest["stopped"] = stop
            G._write_json(run.out / "manifest.json", run.manifest)
            _say(f"stopping: {stop}")
            return
        steps: list[dict[str, Any]] = []
        thoughts: dict[str, Any] | None = None
        try:
            with both_context():
                trial, steps, thoughts = await M._one_turn(run, task, repeat)
        except Exception as exc:
            trial, steps, thoughts = M._not_run(task, repeat, f"harness: {exc!r}"), [], None
        trial, steps = finish_turn(
            run.out, task, arm, repeat, dict(trial), [dict(s) for s in steps]
        )
        errored_run = errored_run + 1 if G._errored(trial, steps) else 0
        G._append_jsonl(run.out / "trials.jsonl", [trial])
        G._append_jsonl(run.out / "steps.jsonl", steps)
        if thoughts is not None:
            G._append_jsonl(run.out / "thoughts.jsonl", [{**dict(thoughts), "arm": ARM}])
        run.turn_s.append(round(_now() - started, 1))
        _say(
            f"{index:02d} {task}/{ARM}/r{repeat} steps={len(steps)} calls={[s.get('call_name') for s in steps]} "
            f"statuses={[s.get('status') for s in steps]} success={trial.get('success')} "
            f"evidence={trial.get('evidence')} end={trial.get('tool_loop_end')} err={trial.get('exception')} "
            f"cli_after={trial.get('cli_after')} wall={run.turn_s[-1]}s"
        )
        last = f"{task}/{ARM}/r{repeat}"


# =====================================================================================================
# A4: refusals (no socket)
# =====================================================================================================


def t0212_changed(merge: str) -> bool:
    """A1/N1: ``scripts/gpu/t0212_threejs_arms.py`` differs from its blob at ``merge``."""
    try:
        blob = _git("show", f"{merge}:{T0212_REL}")
    except (OSError, subprocess.CalledProcessError):
        return True
    return not T0212_PATH.is_file() or T0212_PATH.read_text(encoding="utf-8") != blob


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def _refusal(args: argparse.Namespace) -> tuple[str | None, str, Path, dict[str, Any]]:
    """A4's refusals in order: T-0212's (``R._refusal`` with the T-0214 out-dir), then no ``Merge T-0212:`` merge,
    T-0212's harness off its pin, P-A failing. ``(reason, sha, out, context)``; no socket."""
    from local_astra.config import load

    ns = argparse.Namespace(**vars(args))
    if ns.out_dir is None:
        try:
            head = _git("rev-parse", "HEAD").strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"{REPO} is not a readable git tree ({exc})", "", Path(), {}
        ns.out_dir = REPO / "logs" / f"t0214-threejs-both-{head[:7]}"
    reason, sha, out, ctx = R._refusal(ns)
    if reason is not None:
        return reason, sha, out, ctx
    merge = R._merge_sha(MERGE_T0212)
    if merge is None:
        return (
            f"no '{MERGE_T0212}' merge is reachable from HEAD (T-0212's harness, which this run imports; a branch "
            "merge does not count)",
            sha,
            out,
            ctx,
        )
    if t0212_changed(merge):
        return (
            f"{T0212_REL} differs from its bytes at {MERGE_T0212} {merge[:12]} (read-only harness)",
            sha,
            out,
            ctx,
        )
    lean = load().profiles["lean"]
    pa = pa_check(lean)
    if pa["problems"]:
        return "P-A (both) failed: " + "; ".join(pa["problems"]), sha, out, ctx
    ctx = dict(ctx)
    ctx.update(pa_t0212=ctx.get("pa"), pa=pa, merge_t0212=merge, lean=lean)
    return None, sha, out, ctx


# =====================================================================================================
# main
# =====================================================================================================


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    reason, sha, out, ctx = _refusal(args)
    if reason is not None:
        _say(f"refusing: {reason}")
        return EXIT_REFUSED
    # --- sockets from here on: loopback, unproxied ---------------------------------------------------------------
    from local_astra.conductor.llm_client import VllmEndpoint

    model = VllmEndpoint().model
    listing = G._get_text(args.base_url.rstrip("/") + "/models")
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
        _say(f"{G._root(args.base_url)}/metrics has no prefix-cache counters")
        return EXIT_FAILED
    vram = T._VramSampler()
    vram.start()
    try:
        return _measure(args.base_url, out, sha, model, vram, ctx)
    finally:
        vram.stop()


def _measure(
    base_url: str, out: Path, sha: str, model: str, vram: Any, ctx: Mapping[str, Any]
) -> int:
    """Past every refusal and the engine check: the order, the manifest, P1-P3, P-M1, P-M2, the turns, the report."""
    from local_astra.conductor.session import REGROUND_TIMEOUT_MS

    lean = ctx["lean"]
    pa = ctx["pa"]
    sentinel = G._sentinel_jpeg()
    turns = order(ORDER_SEED)
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    G._write_json(out / "order.json", [list(k) for k in turns])
    sources = dict(P.regime_sources())
    sources.pop("arms", None)  # T-0132's arm table, not this run's
    lock = R._versions_lock()
    regime = {
        **sources,
        "stt_source": "scripts/locked/t0214_threejs_both.sh refuses while 127.0.0.1:8080/8081 are bound",
        "commit": sha,
        "merge_t0210_c": ctx["merge"],
        "merge_t0212": ctx["merge_t0212"],
        "date_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "base_url": base_url,
        "served_model": model,
        "engine_version": G._get_text(G._root(base_url) + "/version"),
        "profile": lean.name,
        "canvas_length": lean.diffusion.canvas_length,
        "max_denoising_steps": lean.diffusion.max_denoising_steps,
        "thinking": lean.diffusion.thinking,
        "model": sources.get("model") or lock.get("model"),
        "model_revision": sources.get("model_revision") or lock.get("model_revision"),
        "max_num_seqs": sources.get("max_num_seqs") or lock.get("max_num_seqs"),
        "tools": f"native:{ctx['kit_sha'][:12]}:live:loop{ctx['default']['max_tool_steps']}",
        "repair": "on",
        "arms": list(ARMS),
        "n_per_arm": len(TASK_IDS) * REPEATS,
        "n_per_cell": REPEATS,
        "turns": len(turns),
        "concurrency": 1,
        "order_seed": ORDER_SEED,
        "order": 'three blocks; each holds M1-M5 once as (task, "both", r), shuffled in turn with one '
        "random.Random(seed)",
        "engine_seed": None,
        "turn_timeout_s": TURN_TIMEOUT_S,
        "run_budget_s": RUN_BUDGET_S,
        "hard_limit_s": HARD_LIMIT_S,
        "turn_worst_s": TURN_WORST_S,
        "turn_worst_note": TURN_WORST_NOTE,
        "report_s": REPORT_S,
        "crop_budget": pa["grounding"]["both"]["reground_visual_tokens"],
        "crop_source": pa["grounding"]["both"]["reground_visual_tokens_from"],
        "regrounding_timeout_ms": REGROUND_TIMEOUT_MS,
        "browser": f"{M.BROWSER}, headless, one per turn, fresh context (no profile directory), web mode on one "
        "loopback origin per turn at the task's seeded start page, deny-all resolver and connector",
        "search": "off",
        "viewport": list(lean.browser_viewport),
        "state_read": M.STATE_READ_JS,
        "input": "typed (UserText); fake STT/TTS/audio; perception holds only the HUD sentinel frame; no audio "
        "device, no headset or speakers",
        "versions_lock_sha256": G._sha256(REPO / "services" / "versions.lock"),
        "versions_lock": lock,
        "grounding_source_note": GROUNDING_SOURCE_NOTE,
        "anchor": f"{ANCHOR_RUN} at {ANCHOR_COMMIT} (unpaired, quoted)",
    }
    manifest: dict[str, Any] = {
        "regime": regime,
        "live_default": dict(ctx["default"]),
        "arm_keywords": pa["keywords"],
        "arm_diffs": pa["diffs"],
        "arm_grounding": pa["grounding"],
        "tool_guidance_sha256": pa["tool_guidance_sha256"],
        "cross_check": pa["cross_check"],
        "regime_drift": regime_drift(pa["keywords"]["base"]),
        "search": "off",
        "executor_constants": {k: D._jsonable(v) for k, v in D.executor_constants().items()},
        "task_table_sha256": threejs_tasks.task_table_sha(),
        "provenance_sha256": ctx["prov"],
        "hud_sentinel_sha256": hashlib.sha256(sentinel).hexdigest(),
        "gate_file_sha256": {
            rel: G._sha256(REPO / rel) for rel in (*GATE_FILES, T0212_REL) if (REPO / rel).is_file()
        },
        "cli_before": cli_defaults(),
        "stopped": None,
    }
    G._write_json(out / "manifest.json", manifest)
    for name in ("trials.jsonl", "steps.jsonl", "thoughts.jsonl", "turns.jsonl"):
        (out / name).touch()

    # --- P1-P3 (T-0131's), P-M1 and P-M2 (T-0210 C4's), before any turn ------------------------------------------
    probe_start = time.monotonic()
    probes: dict[str, Any] = dict(M.run_probes(base_url, model))
    for name in ("p1", "p2", "p3"):
        probes[name] = {**probes[name], "passed": M._probe_passed(probes[name])}
    try:
        pm1 = dict(M.probe_pm1())
    except Exception as exc:
        pm1 = {"error": repr(exc)}
    probes["pm1"] = {**pm1, "passed": M.pm1_ok(pm1)}
    try:
        pm2 = dict(M.probe_pm2(pm1))
    except Exception as exc:
        pm2 = {"error": repr(exc)}
    probes["pm2"] = {**pm2, "passed": M.pm2_ok(pm2)}
    manifest["probe_vram_min_free_mib"] = vram.min_between(probe_start, time.monotonic())
    manifest["vram_sampling_error"] = vram.error
    G._write_json(out / "probes.json", probes)
    if M.probe_verdicts(probes)["void"]:
        stopped = "not run: a pre-flight probe failed"
        manifest["stopped"] = stopped
        manifest["cli_after"] = cli_defaults()
        G._write_json(out / "manifest.json", manifest)
        report = write_report(out, probes, manifest, stopped)
        _say(
            f"a probe failed: the run is void ({json.dumps(report['probes'])}); label {report['label']}"
        )
        return EXIT_PROBE
    G._write_json(out / "manifest.json", manifest)

    run = M._Run(
        base_url=base_url, out=out, profile=lean, sentinel=sentinel, manifest=manifest, vram=vram
    )
    try:
        asyncio.run(_run_all(run, turns))
    except Exception as exc:
        run.stopped = f"the turn runner raised: {exc!r}"
        _say(f"stopping: {run.stopped}")
    manifest["stopped"] = run.stopped
    manifest["cli_after"] = cli_defaults()
    G._write_json(out / "manifest.json", manifest)

    report = write_report(out, probes, manifest, run.stopped)
    _say(report["reading"]["line"])
    _say(UNPAIRED_NOTE)
    _say(
        f"label={report['label']}; written to {out}; cli defaults after the run (N2): {manifest['cli_after']}"
    )
    if report["label"] == VOID:
        return EXIT_PROBE
    return EXIT_FAILED if run.stopped is not None else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
