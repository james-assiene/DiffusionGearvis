"""T-0202 group C -- the wait-tool GPU A/B: with ``wait`` offered (and its one guidance sentence), does the model look
again at a page that is still loading instead of telling the user to wait, without costing suite successes, safety
or speed? ``/launch`` ONLY.

    uv run python scripts/gpu/t0202_wait_ab.py --base-url http://127.0.0.1:8001/v1 [--out-dir DIR]

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the probe table (``PROBE_TASKS``), the oracles
(``told_to_wait``, ``probe_ok``), the order (``order``), the metrics (``arm_metrics``), the decision (``decide``) and
the report (``build_report``, ``render_markdown``) are pure; everything engine- or browser-facing is imported inside
``main`` and the runners.

C1: imported, never copied. T-0201 C's harness (``scripts/gpu/t0201_screen_reminder_ab.py``, loaded read-only by
path under ``T0201_MODULE_NAME``) for its box preflight (``_git``, ``_nvidia_smi``, ``_loadavg``,
``_listening_ports``, ``_chromium_path``, ``engine_healthy``, ``_other_compute``) and HEAD's live extras
(``head_live_extras``); through it T-0138's harness (``S``: the turn runner ``_one_turn``, ``succeeded``,
``forced_failure``, ``safety_counts``, ``_voiced_text``) and T-0111g's (``G``: ``_percentile``, ``check_loopback``,
the JSON writers). Every turn, probe and suite alike, runs T-0138's own ``S._one_turn`` unchanged. The seams:
``arm_turn`` (for one turn, ``session.Session`` applies HEAD's live extras and the arm's guidance over the runner's
keywords, and ``cli.load_tool_kit`` builds the arm's kit, ``wait_tool=WAIT_TOOL[arm]``), and for a probe turn
``_probe_scope`` (T-0138's module names ``SUITE_DIR``, ``_task`` and ``URL_MARKERS`` point at ``PROBE_DIR``, the
probe row and the suite's markers plus the probe pages; restored after).

Refusals (C2; exit 2), all **before any socket** and before the out-dir is touched: a non-loopback ``--base-url``; a
dirty tree (named by its paths); an existing ``--out-dir``; no ``Merge T-0202-b:`` merge reachable from HEAD;
``cli.load_tool_kit`` without the ``wait_tool`` keyword; a box that is not idle (a GPU compute process other than the
engine, 1-min load >= 2, port 8000 or 1420 bound); Chromium absent. Then ``order.json`` is written, and only then the
first socket: with no healthy engine (``engine_healthy``) the harness exits 1 having sent nothing.

Arms (C3): OFF (``--wait-tool off``: tools v1 ``bbfc53cc``, HEAD's live guidance) and ON (``--wait-tool on``:
``tools_v1_wait.json``, the guidance + one space + H-T202-4). Every other setting is HEAD's live default (T-0138's
``live_default`` and T-0201's ``head_live_extras`` with ``--no-web``, plus the screen reminder at its live default),
stated in the manifest and equal in both arms. Tasks: W1 (``search.html``) and W2 (``list.html``) under
``local_astra/pages/wait_probe/`` (served only here), n = ``N_PROBE`` per arm; T-0138's S6-S15, n = ``N_SUITE`` per
arm; ``order(ORDER_SEED)`` jointly shuffled, concurrency 1, the engine unseeded.

Metrics (C4, ``arm_metrics``) and decision (C5, ``decide``): see their docstrings; readings 1-7 of
``tests/unit/test_t0202_ab_harness.py`` are taken (6 as ruled 2026-10-30: safety FAIL first).

Output (``--out-dir``, default ``logs/t0202-wait-ab-<sha7>/``): ``order.json``, ``manifest.json``, ``trials.jsonl``,
``steps.jsonl``, ``turns.jsonl``, ``sessions/``, ``report.json``, ``report.md`` (every told-to-wait hit and every
wait call named, as a hand-audit list). Exit codes: 0 measured; 1 no engine, a run that could not start or stopped
early; 2 refused.

Regime (state it beside every number): profile lean; versions from ``services/versions.lock`` at HEAD; CL 256, 48
steps, thinking on, repair on; HEAD's live defaults (recorded); T-0138's turn policy (``G.MAX_TOOL_STEPS``,
recorded beside HEAD's); headless Chromium 1280x800, loopback; typed turns, fake STT/TTS/audio; engine unseeded;
order seed 0; W1/W2 x 5 per arm, S6-S15 x 3 per arm (80 turns); clean tree, idle box.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import hashlib
import importlib.util
import inspect
import json
import random
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
#: T-0201 C's harness, read-only (C1: imported, never copied); it imports T-0138's and through it T-0111g's
T0201_PATH: Path = REPO / "scripts" / "gpu" / "t0201_screen_reminder_ab.py"
T0201_MODULE_NAME = "_t0201_screen_reminder_ab_for_t0202"


def _load_t0201() -> ModuleType:
    """T-0201 C's harness, imported read-only by path under ``T0201_MODULE_NAME``. Importing it opens no socket."""
    found = sys.modules.get(T0201_MODULE_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0201_MODULE_NAME, T0201_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0201_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0201_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0201_MODULE_NAME, None)
        raise
    return module


R: ModuleType = _load_t0201()
#: T-0138's harness as T-0201's loaded it (the same module object), and through it T-0111g's, T-0131's, T-0132's
S: ModuleType = R.S
G: ModuleType = S.G
T: ModuleType = S.T
P: ModuleType = S.P

# --- seams ``main`` reads at call time (T-0201 C's objects, unchanged; the tests replace them here) ---------------
_git = R._git
_nvidia_smi = R._nvidia_smi
_loadavg = R._loadavg
_listening_ports = R._listening_ports
_chromium_path = R._chromium_path
engine_healthy = R.engine_healthy

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
ARMS = ("OFF", "ON")
#: C3: the arm's ``--wait-tool``
WAIT_TOOL = {"OFF": False, "ON": True}
ORDER_SEED = 0
#: C3: per-arm repeats per probe task and per suite task
N_PROBE = 5
N_SUITE = 3
PROBE_IDS = ("W1", "W2")
SUITE_IDS: tuple[str, ...] = tuple(S.TASK_IDS)
#: C3: the probe pages' root; served only by this harness (never by ``astra session``)
PROBE_DIR: Path = (REPO / "local_astra" / "pages" / "wait_probe").resolve()
#: C5's bars
OFF_PROBE_INCONCLUSIVE = 8
PROBE_MARGIN = 3
SUITE_MARGIN = 3
MEDIAN_RATIO = Decimal("1.10")
MAX_SUITE_TURNS_WITH_WAIT = 3
#: C2
B_MERGE = "Merge T-0202-b:"
#: recorded in the manifest only (the task: C "should run with T-0203 B merged"; not a refusal)
T0203_MERGE = "Merge T-0203-b:"
BUSY_PORTS = (8000, 1420)
LOAD_MAX = 2.0
#: T-0202 B's refusal texts (H-T202-5), read from the executor when a step is classified
WAIT_NAME = "wait"

#: C4: "told the user to wait" (reading 1: case-insensitive)
TOLD_TO_WAIT = re.compile(
    r"please wait|wait for (?:the )?page|refresh (?:the )?(?:page|screen)", re.IGNORECASE
)


@dataclass(frozen=True)
class ProbeTask:
    """One probe task: what T-0138's runner reads of a suite task (``start_page``, ``request``)."""

    task_id: str
    start_page: str
    request: str


PROBE_TASKS: dict[str, ProbeTask] = {
    "W1": ProbeTask(
        "W1",
        "/search.html",
        "Search this shop for the Brightwell kettle and tell me its price.",
    ),
    "W2": ProbeTask(
        "W2",
        "/list.html",
        "Open the live departures and tell me which platform the 10:42 to Hollins Cross leaves from.",
    ),
}
#: the probe pages as URL markers (a record holding one is a URL leak, as for the suite's start pages)
PROBE_URL_MARKERS = ("/search.html", "/list.html")


def _say(msg: str) -> None:
    print(f"t0202_wait_ab: {msg}", file=sys.stderr, flush=True)


# =====================================================================================================
# C4: told to wait; C3: the probe oracles
# =====================================================================================================


def told_to_wait(text: str) -> bool:
    """C4: the voiced text tells the user to wait or refresh (``TOLD_TO_WAIT``)."""
    return TOLD_TO_WAIT.search(text or "") is not None


def _fold(text: str) -> str:
    return re.sub(r"[\s\-]+", " ", (text or "").casefold())


#: W1's target in digits: "34.99" not inside a longer number ($134.99, 34.995)
_W1_DIGITS = re.compile(r"(?<![\d.])34\.99(?![\d])")
#: W1's target in words: "thirty four ninety nine", "thirty four dollars (and) ninety nine (cents)",
#: "34 dollars and 99 cents"
_W1_WORDS = re.compile(
    r"\b(?:thirty four|34)(?: dollars?)?(?: and)? (?:ninety nine|99)(?: cents?)?\b"
)
_W1_DISTRACTORS = re.compile(r"(?<![\d.])(?:29\.99|41\.50?|134\.99)(?![\d])|\bhundred\b")
_PLATFORM = r"\bplatform (?:number |no\.? )?"
_W2_TARGET = re.compile(_PLATFORM + r"(?:7|seven)\b(?![.,]\d)")
_W2_DISTRACTORS = re.compile(_PLATFORM + r"(?:3|three|12|twelve|17|seventeen)\b")


def probe_ok(task_id: str, voiced: str) -> bool:
    """C3's pure oracles on the voiced answer: W1 says the Brightwell kettle's $34.99 (digits or words), W2 says
    Platform 7. An answer that also names a distractor on the same page, or tells the user to wait, is never a
    success."""
    text = voiced or ""
    if told_to_wait(text):
        return False
    folded = _fold(text)
    if task_id == "W1":
        hit = _W1_DIGITS.search(text) is not None or _W1_WORDS.search(folded) is not None
        return hit and _W1_DISTRACTORS.search(folded) is None
    if task_id == "W2":
        return _W2_TARGET.search(folded) is not None and _W2_DISTRACTORS.search(folded) is None
    raise KeyError(task_id)


# =====================================================================================================
# C3: the order
# =====================================================================================================


def order(seed: int) -> list[tuple[str, str, int]]:
    """Reading 2: arm-major over ``ARMS``, then ``PROBE_IDS`` x ``range(N_PROBE)`` then ``SUITE_IDS`` x
    ``range(N_SUITE)``, then one ``random.Random(seed).shuffle`` (the arms interleave): 80 turns."""
    turns = [
        (arm, task, r)
        for arm in ARMS
        for tasks, n in ((PROBE_IDS, N_PROBE), (SUITE_IDS, N_SUITE))
        for task in tasks
        for r in range(n)
    ]
    random.Random(seed).shuffle(turns)
    return turns


# =====================================================================================================
# C4: metrics, per arm
# =====================================================================================================


def _key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(record.get("arm")), str(record["task"]), int(record["repeat"]))


def _wait_reasons() -> tuple[str, str]:
    from local_astra.computer import executor

    return str(executor.WAIT_RUN_REASON), str(executor.WAIT_TURN_REASON)


def arm_metrics(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """C4 per arm (readings 1, 3, 4, 5): probe successes (``probe_ok`` over T-0138's ``_voiced_text``, never on
    T-0138's ``forced_failure``; a stored ``success`` is not read), told-to-wait turns (once per turn, every turn),
    wait calls (steps whose ``call_name`` is ``wait``), executed waits, wait calls per turn, refusals by
    ``WAIT_RUN_REASON``/``WAIT_TURN_REASON``, suite successes (T-0138's ``succeeded``), suite turns with a wait call,
    nearest-rank median and p90 suite wall time (``G._percentile``) and T-0138's safety counters."""
    run_reason, turn_reason = _wait_reasons()
    by: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {_key(t): [] for t in trials}
    for s in steps:
        if _key(s) in by:
            by[_key(s)].append(s)
    out: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        mine = [t for t in trials if str(t.get("arm")) == arm]
        mine_steps = [s for t in mine for s in by[_key(t)]]
        probes = [t for t in mine if str(t["task"]) in PROBE_IDS]
        suite = [t for t in mine if str(t["task"]) not in PROBE_IDS]
        probe_successes = sum(
            1
            for t in probes
            if not S.forced_failure(t, by[_key(t)]) and probe_ok(str(t["task"]), S._voiced_text(t))
        )
        waits = [s for s in mine_steps if s.get("call_name") == WAIT_NAME]
        refused = [s for s in waits if s.get("status") == "refused"]
        walls = [float(t["wall_s"]) for t in suite if t.get("wall_s") is not None]
        out[arm] = {
            "turns": len(mine),
            "probe_n": len(probes),
            "probe_successes": probe_successes,
            "told_to_wait": sum(1 for t in mine if told_to_wait(S._voiced_text(t))),
            "wait_calls": len(waits),
            "waits_executed": sum(1 for s in waits if s.get("status") == "executed"),
            "wait_calls_per_turn": (len(waits) / len(mine)) if mine else 0,
            "wait_refusals": {
                "run": sum(1 for s in refused if s.get("reason") == run_reason),
                "turn": sum(1 for s in refused if s.get("reason") == turn_reason),
            },
            "suite_n": len(suite),
            "suite_successes": sum(1 for t in suite if S.succeeded(t, by[_key(t)])),
            "suite_turns_with_wait": sum(
                1 for t in suite if any(s.get("call_name") == WAIT_NAME for s in by[_key(t)])
            ),
            "suite_wall_median_s": G._percentile(walls, 50),
            "suite_wall_p90_s": G._percentile(walls, 90),
            "safety": dict(S.safety_counts(mine, mine_steps)),
        }
    return out


# =====================================================================================================
# C5: the decision
# =====================================================================================================


def _dec(value: Any) -> Decimal:
    """A wall time as the decimal it prints as (reading 6: 1.10 x OFF compared exactly in decimal)."""
    return Decimal(str(value))


def safety_hits(metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """One reason per non-zero safety counter, in either arm (reading 6: these are FAIL before anything else)."""
    return [
        f"{arm} safety counter {name} = {value}"
        for arm in ARMS
        for name, value in sorted((metrics[arm].get("safety") or {}).items())
        if value
    ]


def decide(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """C5 with the ruled precedence (reading 6, 2026-10-30): a non-zero safety counter in either arm is FAIL; else
    OFF probe successes >= ``OFF_PROBE_INCONCLUSIVE`` is INCONCLUSIVE; else the positive control (ON executed at
    least one wait, OFF none) failing is INCONCLUSIVE; else PASS iff ON probe >= OFF + 3, ON told-to-wait <= OFF, ON
    suite >= OFF - 3, ON suite median wall <= 1.10 x OFF (exact decimal) and ON suite turns with a wait <= 3; FAIL
    otherwise. Every verdict carries its reasons."""
    off, on = metrics["OFF"], metrics["ON"]
    unsafe = safety_hits(metrics)
    if unsafe:
        return {"verdict": "FAIL", "reasons": unsafe}
    if int(off["probe_successes"]) >= OFF_PROBE_INCONCLUSIVE:
        return {
            "verdict": "INCONCLUSIVE",
            "reasons": [
                f"OFF probe successes {off['probe_successes']} of {off.get('probe_n')} >= "
                f"{OFF_PROBE_INCONCLUSIVE}: the probe does not need the wait tool"
            ],
        }
    control = []
    if int(on["waits_executed"]) < 1:
        control.append(f"ON executed no wait ({on.get('wait_calls')} wait calls)")
    if int(off["waits_executed"]) != 0:
        control.append(f"OFF executed {off['waits_executed']} waits (the tool is not offered)")
    if control:
        return {
            "verdict": "INCONCLUSIVE",
            "reasons": ["positive control failed: " + c for c in control],
        }
    failed: list[str] = []
    passed: list[str] = []

    def bar(ok: bool, text: str) -> None:
        (passed if ok else failed).append(("" if ok else "bar failed: ") + text)

    bar(
        int(on["probe_successes"]) >= int(off["probe_successes"]) + PROBE_MARGIN,
        f"ON probe successes {on['probe_successes']} >= OFF {off['probe_successes']} + {PROBE_MARGIN}",
    )
    bar(
        int(on["told_to_wait"]) <= int(off["told_to_wait"]),
        f"ON told-to-wait {on['told_to_wait']} <= OFF {off['told_to_wait']}",
    )
    bar(
        int(on["suite_successes"]) >= int(off["suite_successes"]) - SUITE_MARGIN,
        f"ON suite successes {on['suite_successes']} >= OFF {off['suite_successes']} - {SUITE_MARGIN}",
    )
    m_on, m_off = on.get("suite_wall_median_s"), off.get("suite_wall_median_s")
    if m_on is None or m_off is None:
        bar(False, f"suite median wall time missing (ON {m_on}, OFF {m_off})")
    else:
        bar(
            _dec(m_on) <= MEDIAN_RATIO * _dec(m_off),
            f"ON suite median wall {m_on} s <= {MEDIAN_RATIO} x OFF {m_off} s",
        )
    bar(
        int(on["suite_turns_with_wait"]) <= MAX_SUITE_TURNS_WITH_WAIT,
        f"ON suite turns with a wait {on['suite_turns_with_wait']} <= {MAX_SUITE_TURNS_WITH_WAIT}",
    )
    if failed:
        return {"verdict": "FAIL", "reasons": failed}
    return {"verdict": "PASS", "reasons": passed}


# =====================================================================================================
# The report
# =====================================================================================================


def audit_rows(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """C5's hand-audit list: every told-to-wait turn (its voiced text) and every wait call (its turn, step, status,
    reason)."""
    told = [
        {
            "arm": t.get("arm"),
            "task": t.get("task"),
            "repeat": t.get("repeat"),
            "voiced": S._voiced_text(t),
            "match": (TOLD_TO_WAIT.search(S._voiced_text(t)) or [""])[0],
        }
        for t in trials
        if told_to_wait(S._voiced_text(t))
    ]
    waits = [
        {
            "arm": s.get("arm"),
            "task": s.get("task"),
            "repeat": s.get("repeat"),
            "step": s.get("step"),
            "status": s.get("status"),
            "reason": s.get("reason"),
        }
        for s in steps
        if s.get("call_name") == WAIT_NAME
    ]
    return {"told_to_wait": told, "wait_calls": waits}


def build_report(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    stopped: str | None,
) -> dict[str, Any]:
    """The metrics, the decision, the probe answers and the audit list. A run that stopped early is FAIL if either
    arm already has a safety hit (reading 6: safety takes precedence over every INCONCLUSIVE), else INCONCLUSIVE."""
    metrics = arm_metrics(trials, steps)
    if stopped is None:
        decision = decide(metrics)
    elif unsafe := safety_hits(metrics):
        decision = {"verdict": "FAIL", "reasons": unsafe}
    else:
        decision = {"verdict": "INCONCLUSIVE", "reasons": [f"the run stopped early: {stopped}"]}
    probes = [
        {
            "arm": t.get("arm"),
            "task": t.get("task"),
            "repeat": t.get("repeat"),
            "ok": probe_ok(str(t["task"]), S._voiced_text(t)),
            "end": t.get("tool_loop_end"),
            "voiced": S._voiced_text(t),
        }
        for t in trials
        if str(t.get("task")) in PROBE_IDS
    ]
    return {
        "metrics": metrics,
        "decision": decision,
        "stopped": stopped,
        "probes": probes,
        "audit": audit_rows(trials, steps),
    }


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """``report.md``: the verdict and reasons, the per-arm table, every probe answer, and the hand-audit list."""
    lines = ["# T-0202 C: wait-tool A/B", ""]
    d = report["decision"]
    lines += [f"**Verdict: {d['verdict']}**", ""] + [f"- {r}" for r in d["reasons"]] + [""]
    if manifest is not None:
        lines += [
            "## Regime",
            "",
            "```json",
            json.dumps(manifest.get("regime"), indent=2),
            "```",
            "",
        ]
    keys = (
        "turns",
        "probe_n",
        "probe_successes",
        "told_to_wait",
        "wait_calls",
        "waits_executed",
        "wait_calls_per_turn",
        "wait_refusals",
        "suite_n",
        "suite_successes",
        "suite_turns_with_wait",
        "suite_wall_median_s",
        "suite_wall_p90_s",
        "safety",
    )
    m = report["metrics"]
    lines += ["## Per arm", "", "| metric | OFF | ON |", "|---|---|---|"]
    lines += [f"| {k} | {m['OFF'].get(k)} | {m['ON'].get(k)} |" for k in keys] + [""]
    lines += ["## Probe answers", ""]
    lines += [
        f"- {p['arm']} {p['task']}/r{p['repeat']} ok={p['ok']} end={p['end']}: {p['voiced']!r}"
        for p in report["probes"]
    ] + [""]
    audit = report["audit"]
    lines += ["## Hand audit: told-to-wait hits", ""]
    lines += [
        f"- [ ] {a['arm']} {a['task']}/r{a['repeat']} ({a['match']!r}): {a['voiced']!r}"
        for a in audit["told_to_wait"]
    ] or ["- none"]
    lines += ["", "## Hand audit: wait calls", ""]
    lines += [
        f"- [ ] {a['arm']} {a['task']}/r{a['repeat']} step {a['step']}: {a['status']}"
        + (f" ({a['reason']})" if a["reason"] else "")
        for a in audit["wait_calls"]
    ] or ["- none"]
    return "\n".join(lines) + "\n"


# =====================================================================================================
# The seams over T-0138's turn runner
# =====================================================================================================


def head_live_extras(profile: Any) -> dict[str, Any]:
    """C3: the ``Session`` keywords of HEAD's loopback live default that T-0138's runner predates: T-0201's
    ``head_live_extras`` (guidance text, page fact, re-ask, action outcome, repeat guard), plus the screen reminder
    when it is on by default (T-0201 D)."""
    from local_astra import cli

    extras = dict(R.head_live_extras(profile))
    ns = argparse.Namespace(no_web=True)
    if cli.screen_reminder_of(ns, "live").screen_reminder:
        extras["screen_reminder"] = True
    return extras


@contextlib.contextmanager
def arm_turn(arm: str, extras: Mapping[str, Any]) -> Iterator[None]:
    """For one T-0138 turn: ``session.Session`` applies ``extras`` and the arm's guidance (ON: + one space +
    H-T202-4) over the runner's keywords, and ``cli.load_tool_kit`` builds the arm's kit
    (``wait_tool=WAIT_TOOL[arm]``). Both looked up by the runner at call time; both restored after."""
    from local_astra import cli
    from local_astra.conductor import session as session_module
    from local_astra.conductor.prompt_builder import with_wait_guidance

    wait = WAIT_TOOL[arm]
    overrides = dict(extras)
    base = overrides.get("tool_guidance")
    if wait and base is not None:
        overrides["tool_guidance"] = with_wait_guidance(str(base))
    real_session = session_module.Session
    real_load = cli.load_tool_kit

    class _ArmSession(real_session):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **{**kwargs, **overrides})

    def load_tool_kit(*args: Any, **kwargs: Any) -> Any:
        return real_load(*args, **{**kwargs, "wait_tool": wait})

    setattr(session_module, "Session", _ArmSession)  # noqa: B010
    setattr(cli, "load_tool_kit", load_tool_kit)  # noqa: B010
    try:
        yield
    finally:
        setattr(cli, "load_tool_kit", real_load)  # noqa: B010
        setattr(session_module, "Session", real_session)  # noqa: B010


@contextlib.contextmanager
def _probe_scope(task_id: str) -> Iterator[None]:
    """For one probe turn: T-0138's ``SUITE_DIR`` is ``PROBE_DIR``, its ``_task`` gives the probe row, and its
    ``URL_MARKERS`` add the probe pages; restored after. (``judge`` stays T-0138's: total, it never raises, and a
    probe trial's ``success`` is not read.)"""
    row = PROBE_TASKS[task_id]
    saved = {name: getattr(S, name) for name in ("SUITE_DIR", "_task", "URL_MARKERS")}

    def task(tid: str) -> Any:
        if tid != task_id:
            raise KeyError(tid)
        return row

    setattr(S, "SUITE_DIR", PROBE_DIR)  # noqa: B010
    setattr(S, "_task", task)  # noqa: B010
    setattr(S, "URL_MARKERS", (*saved["URL_MARKERS"], *PROBE_URL_MARKERS))  # noqa: B010
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(S, name, value)


# =====================================================================================================
# main
# =====================================================================================================


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def _merged(subject: str) -> bool:
    """A merge commit reachable from HEAD whose subject starts ``subject`` (read through this module's ``_git``)."""
    try:
        subjects = _git("log", "--merges", "--format=%s", "HEAD").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    return any(s.startswith(subject) for s in subjects)


def _refusal(args: argparse.Namespace) -> tuple[str | None, str, Path]:
    """C2's refusals in order; ``(reason, sha, out)``. Reads the tree and the box; opens no socket."""
    try:
        G.check_loopback(args.base_url)
    except ValueError as exc:
        return f"--base-url is not loopback: {exc}", "", Path()
    try:
        dirty = _git("status", "--porcelain").rstrip("\n")
        sha = _git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"{REPO} is not a readable git tree ({exc})", "", Path()
    if dirty:
        return f"the tree at {REPO} is dirty:\n{dirty}", sha, Path()
    out: Path = args.out_dir or REPO / "logs" / f"t0202-wait-ab-{sha[:7]}"
    if out.exists():
        return (
            f"{out} exists; the run happens once per SHA (move it aside only with the human)",
            sha,
            out,
        )
    if not _merged(B_MERGE):
        return f"no '{B_MERGE}' merge is reachable from HEAD (T-0202-b must merge first)", sha, out
    from local_astra import cli

    if "wait_tool" not in inspect.signature(cli.load_tool_kit).parameters:
        return "cli.load_tool_kit has no wait_tool keyword (T-0202 B8)", sha, out
    try:
        smi = _nvidia_smi()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"nvidia-smi could not be asked ({exc}); an idle box cannot be established", sha, out
    others = R._other_compute(smi)
    if others:
        return "another GPU compute process is running: " + "; ".join(others), sha, out
    load1 = float(_loadavg()[0])
    if load1 >= LOAD_MAX:
        return f"the 1-min load is {load1:.2f} (must be under {LOAD_MAX})", sha, out
    busy = sorted(p for p in BUSY_PORTS if p in _listening_ports())
    if busy:
        return f"port(s) {busy} are bound (a session or the HUD is up)", sha, out
    if _chromium_path() is None:
        return f"Chromium ({S.BROWSER}) is not available", sha, out
    return None, sha, out


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    reason, sha, out = _refusal(args)
    if reason is not None:
        _say(f"refusing: {reason}")
        return EXIT_REFUSED
    # --- the order on disk before any socket --------------------------------------------------------------------
    out.mkdir(parents=True)
    G._write_json(out / "order.json", [list(t) for t in order(ORDER_SEED)])
    # --- sockets from here on: loopback, unproxied ------------------------------------------------------------
    if not engine_healthy(args.base_url):
        _say(
            f"no healthy engine at {args.base_url}; nothing sent (the engine must already be up: start it before "
            "running this harness)"
        )
        return EXIT_FAILED
    return _measure(args, sha, out)


def _sha256_text(value: Any) -> Any:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if isinstance(value, str) else value


def _measure(args: argparse.Namespace, sha: str, out: Path) -> int:
    """Past every refusal and the engine check: the manifest, the 80 turns, the report."""
    from local_astra import cli
    from local_astra.config import load
    from local_astra.toolcall import tools

    lean = load().profiles["lean"]
    default = S.live_default(lean)
    bad = S.live_default_findings(default)
    if bad:
        _say(f"cannot run: HEAD's live default {default} is not one T-0138's runner takes: {bad}")
        return EXIT_FAILED
    want = {"OFF": S.V1_SHA256, "ON": str(tools.load(tools.TOOLS_V1_WAIT_PATH).version)}
    for arm in ARMS:
        try:
            kit, _ = cli.load_tool_kit(lean, wait_tool=WAIT_TOOL[arm])
        except cli.ToolsUnavailable as exc:
            _say(f"cannot run: {exc}")
            return EXIT_FAILED
        if str(kit.tools_version) != want[arm] or kit.repair_off is not None:
            _say(
                f"cannot run: arm {arm}'s tools {kit.tools_version} (want {want[arm]}), repair off: "
                f"{kit.repair_off}"
            )
            return EXIT_FAILED
    extras = head_live_extras(lean)
    # HEAD's live loopback graph, resolved as cli does for ``--tools live --no-web`` (not lean's tools-off default)
    loop = cli.tool_loop_of(lean, argparse.Namespace(tools="live", no_web=True))
    manifest: dict[str, Any] = {
        "regime": {
            **P.regime_sources(),
            "commit": sha,
            "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "base_url": args.base_url,
            "engine_version": G._get_text(G._root(args.base_url) + "/version"),
            "profile": lean.name,
            "canvas_length": lean.diffusion.canvas_length,
            "max_denoising_steps": lean.diffusion.max_denoising_steps,
            "thinking": lean.diffusion.thinking,
            "tools_sha256": dict(want),
            "wait_tool": dict(WAIT_TOOL),
            "repair": "on",
            "live_default": dict(default),
            "head_live_extras": {
                k: (v if k != "tool_guidance" else _sha256_text(v)) for k, v in extras.items()
            },
            "max_tool_steps_head": loop.max_tool_steps,
            "max_tool_steps_t0138_policy": G.MAX_TOOL_STEPS,
            "t0203_b_merged": _merged(T0203_MERGE),
            "order_seed": ORDER_SEED,
            "engine_seed": None,
            "n_probe": N_PROBE,
            "n_suite": N_SUITE,
            "probe_tasks": {
                k: {"start_page": v.start_page, "request": v.request}
                for k, v in PROBE_TASKS.items()
            },
            "probe_pages_sha256": {
                p.name: G._sha256(p) for p in sorted(PROBE_DIR.iterdir()) if p.is_file()
            },
            "versions_lock_sha256": G._sha256(REPO / "services" / "versions.lock"),
        },
        "stopped": None,
    }
    G._write_json(out / "manifest.json", manifest)
    trials: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    stopped: str | None = None
    try:
        trials, steps, stopped = _run_turns(args.base_url, out, lean, default, extras, want)
    except Exception as exc:
        stopped = f"the run raised: {exc!r}"
    if stopped is not None:
        _say(f"stopping: {stopped}")
    manifest["stopped"] = stopped
    G._write_json(out / "manifest.json", manifest)
    report = build_report(trials, steps, stopped)
    G._write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report, manifest), encoding="utf-8")
    _say(f"verdict={report['decision']['verdict']} ({'; '.join(report['decision']['reasons'])})")
    _say(f"written to {out}")
    return EXIT_FAILED if stopped is not None else EXIT_OK


def _run_turns(
    base_url: str,
    out: Path,
    lean: Any,
    default: Mapping[str, Any],
    extras: Mapping[str, Any],
    want: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """C3: every turn of ``order(ORDER_SEED)`` through T-0138's ``_one_turn`` (unchanged) under ``arm_turn`` (and
    ``_probe_scope`` for W1/W2); trials and steps appended per turn. Stops after ``G.ERRORED_STOP`` consecutive
    errored turns (the turns not run are not written)."""
    profile = P.arm_profile(lean, int(default["visual_tokens"]), bool(default["reground"]), None)
    shared = {k: default[k] for k in S.DEFAULT_KEYS}
    rows = {
        arm: {
            "label": arm,
            **shared,
            "degeneracy_guard": True,
            "page_overview": False,
            "auto_overview": False,
            "overview_frame": bool(default["overview_frame"]),
            "tools_sha256": want[arm],
        }
        for arm in ARMS
    }
    (out / "sessions").mkdir(exist_ok=True)
    for name in ("trials.jsonl", "steps.jsonl", "turns.jsonl"):
        (out / name).touch()
    vram = T._VramSampler()
    vram.start()
    trials: list[dict[str, Any]] = []
    steps_all: list[dict[str, Any]] = []
    try:
        run = S._Run(
            base_url=base_url,
            out=out,
            profile=profile,
            sentinel=G._sentinel_jpeg(),
            manifest={},
            vram=vram,
            arms=rows,
            crop=None,
        )
        errored = 0
        for index, (arm, task_id, repeat) in enumerate(order(ORDER_SEED)):
            if errored >= G.ERRORED_STOP:
                return trials, steps_all, f"{errored} consecutive errored turns (at turn {index})"
            steps: list[dict[str, Any]] = []
            scope = _probe_scope(task_id) if task_id in PROBE_IDS else contextlib.nullcontext()
            try:
                with scope, arm_turn(arm, extras):
                    trial, steps, _regrounds, _thoughts, _line = asyncio.run(
                        S._one_turn(run, arm, task_id, repeat)
                    )
            except Exception as exc:
                trial = S._not_run(
                    {**rows[arm], "reground_visual_tokens": None},
                    task_id,
                    repeat,
                    f"harness: {exc!r}",
                )
            trial["arm"] = arm
            trial["wait_tool"] = WAIT_TOOL[arm]
            steps = [{**s, "arm": arm, "task": task_id, "repeat": repeat} for s in steps]
            errored = errored + 1 if T._errored(trial, steps, []) else 0
            trials.append(trial)
            steps_all += steps
            G._append_jsonl(out / "trials.jsonl", [trial])
            G._append_jsonl(out / "steps.jsonl", steps)
            calls = [s.get("call_name") for s in steps]
            _say(
                f"{index:03d} {arm} {task_id}/r{repeat} steps={len(steps)} calls={calls} "
                f"statuses={[s.get('status') for s in steps]} end={trial.get('tool_loop_end')} "
                f"wall={trial.get('wall_s')}s err={trial.get('exception')} voiced={S._voiced_text(trial)!r}"
            )
    finally:
        vram.stop()
    return trials, steps_all, None


if __name__ == "__main__":
    sys.exit(main())
