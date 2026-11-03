"""T-0135 B -- the locked 2-arm thought-call recovery A/B of ``tools = live``: 200 typed turns through a real
``Session``. ``/launch`` ONLY.

    uv run python scripts/gpu/t0135_thought_call_ab.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the live default (``live_default``), the arm table
(``arm_rows``, ``arm_table_sha``, ``order``), A1/A2's rule over one request's commits (``call_in_thought``), B4's
findings (``configuration_findings``), B3's counts (``arm_metrics``), the decision (``decide``,
``decide_with_reason``), the report (``build_report``, ``render_markdown``) and the record extraction
(``thought_line``, ``from_thought_steps``) are pure or read ``cli`` only; everything engine- or browser-facing is
imported inside ``main``, the probes and the turn runner.

B1: T-0134's harness (``scripts/gpu/t0134_feedback_ab.py``) is **imported, never copied**, and through it T-0132's,
T-0131's and T-0111g's: ``TASKS``, ``LoopMeter``, ``sentinel_hits``, ``url_leaks``, ``log_complete``,
``non_loopback_sockets``, ``turn_records`` (T-0111g's objects, ``G``), ``probe_p1``, ``run_probes``, ``false_claim``,
``config_evidence`` (T-0131's, ``T``), ``wilson``, ``newcombe`` (T-0132's, ``P``) and ``score_v2``, ``over_click``,
``no_see``, ``coord_mismatch``, ``false_claim_v2`` (T-0134's, ``F``), plus T-0131's per-arm section, box-reading
watch, record extraction, VRAM sampler and engine helpers, T-0132's arm profile, eligibility inputs, C6b
first-failing rule and regime reader, and T-0134's feedback watch, client, step lines and thought record.

Arms (B1, O-15): both arms run at HEAD's live default, read from ``cli`` once per ``main`` (``live_default``): the
visual-token budget and ReGround from ``cli.grounding_of(profile, Namespace(), "live")``, the live feedback and the
executor echo from ``cli.feedback_of(profile, Namespace(), "live")`` (O-16: its ``live_feedback`` choice and its
``action_echo``). ``A`` runs ``recover_thought_calls=False`` and ``B`` ``True``; every setting is passed explicitly in
both. The default is recorded in the manifest (``live_default``, and ``live_default_feedback``: ``feedback_of``'s
result with its sources), with its two rows (``regime.arms``) and their sha (``arm_table_sha256``).

Refusals (exit 2), all **before any socket** opens and before any browser, page server or ``nvidia-smi``: T-0134's
(a non-loopback ``--base-url``, a dirty tree named by its paths, an existing ``--out-dir``, a task table off its pin,
T-0134 group A's keywords absent, ``LIVE_TOOL_GUIDANCE`` off its sha, a tool kit that cannot load / is not
``bbfc53ccdd80`` / has repair off, Chromium absent, the session's ``REGROUND_TIMEOUT_MS`` off T-0131's 4000 ms), plus:
group A2's keyword absent (``Session(recover_thought_calls=)``), a ``cli`` without ``feedback_of`` (O-16), and a live
default that is not well formed. Past them the engine is checked (``/v1/models`` and
``/metrics``): with no engine the harness exits 1 having launched no browser.

Probes (B2), before any arm turn: P1-P3 are T-0131's ``run_probes`` (P1 failing: inconclusive; P2 or P3: void; exit
3, a report, no arm turn). P4 is not re-run: the feature adds no request content.

What it runs (*Regime*): ``order(ORDER_SEED)``, 200 ``(arm, task, repeat)`` turns (2 arms x 5 tasks x ``REPEATS``)
shuffled jointly, concurrency 1, the engine unseeded. Each turn is T-0134's turn (one fresh start-page server, one
fresh headless ``BrowserToolExecutor(echo=...)`` behind T-0131's box-reading watch and T-0134's feedback watch, one
fresh ``Session`` with one ``UserText``) at the live default and the arm's ``recover_thought_calls``. The run stops
after ``ERRORED_STOP`` consecutive errored turns, when ``RUN_BUDGET_S`` has passed since ``PROCESS_START_S`` (from
process start, probes included), or when the turn runner raised; turns not run are **not** written, so
``trials.jsonl`` has fewer than 200 lines exactly when ``stopped`` is set. ``report.json`` and ``report.md`` are
written straight after the arms, before anything else can fail.

Output (``--out-dir``, default ``logs/t0135-thought-call-ab-<sha7>/``): ``probes.json``, ``trials.jsonl`` (T-0134's
fields plus ``recover_thought_calls`` and ``recover_thought_calls_logged``, the A8 key of the turn line),
``steps.jsonl`` (T-0134's step fields plus ``from_thought``: a ``tool.call`` record of the step's call carries
``from_thought: true``), ``regrounds.jsonl``, ``thoughts.jsonl`` (one line per turn: per ``generate``/``tool_step``
request its ``llm.commit`` data and the text of the body the meter forwarded), ``clicks.jsonl``, ``turns.jsonl`` and
``sessions/*.jsonl``, ``manifest.json``, ``report.json`` and ``report.md``. Exit codes: 0 the run was measured; 1 no
engine, or the run stopped early; 2 refused; 3 a probe failed (void or inconclusive).

The label (B5, H-T135-1 approved as proposed): "inconclusive" when P1 failed; "void" on any configuration finding
(B4, reason ``configuration: <finding>``) or when P2 or P3 failed; "incomplete" when the run stopped early;
"inconclusive" when any arm is ineligible (the reason names the arm, the clause and the first failing trial); then
"recover" iff success(B) >= success(A) - 3, S3(B) >= S3(A), over-clicks(B) <= over-clicks(A) + 2, false claims(B) <=
false claims(A) + 3 and every safety count is 0, else "no change".

Regime (state it beside every number): profile lean, canvas length 256, 48 denoising steps, thinking on; vLLM
0.29.0, NVFP4 rev 2ea8372, ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16, ``services/vllm.sh`` at 806da12,
one boot; tools native ``bbfc53ccdd80``, live, repair on, loop 8, ``LIVE_TOOL_GUIDANCE`` 768a973b; both arms at
HEAD's live default (recorded), differing only in ``recover_thought_calls``; headless Chromium at 1280x800, DSF 1,
loopback start pages only; typed turns, fake STT/TTS/audio, no headset or speakers; 2 arms x 5 tasks x n = 20 = 200
turns, order seed 0, concurrency 1, the engine unseeded; STT down. History, not like-for-like: T-0132 at 2ccff41
(D-5 29/300) and T-0134's arms at e59aa69.
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
import time
from collections import Counter
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
T0134_MODULE_NAME = "_t0134_feedback_ab_for_t0135"


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
TOOLS_SHA12 = G.TOOLS_SHA12
# --- imported from T-0131, unchanged --------------------------------------------------------------------------
INTENDED_TARGETS = T.INTENDED_TARGETS
false_claim = T.false_claim
reground_voiced_findings = T.reground_voiced_findings
probe_p1 = T.probe_p1
run_probes = T.run_probes
config_evidence = T.config_evidence
BROWSER = T.BROWSER
REGROUND_TIMEOUT_MS = T.REGROUND_TIMEOUT_MS
# --- imported from T-0132, unchanged --------------------------------------------------------------------------
wilson = P.wilson
newcombe = P.newcombe
# --- imported from T-0134, unchanged --------------------------------------------------------------------------
score_v2 = F.score_v2
over_click = F.over_click
no_see = F.no_see
coord_mismatch = F.coord_mismatch
false_claim_v2 = F.false_claim_v2
LIVE_TOOL_GUIDANCE_SHA256 = F.LIVE_TOOL_GUIDANCE_SHA256
#: the tree is read through this name (T-0111g's seam); ``main`` looks it up at call time
_git = G._git

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3
#: B1: the arm labels, in the order the eligibility step checks them
LABELS = ("A", "B")
#: B1: what each arm sets ``recover_thought_calls`` to (the only difference between the arms)
RECOVER = {"A": False, "B": True}
#: B1: the live-default settings both arms carry, read from ``cli`` at run time (never hard-coded)
DEFAULT_KEYS = ("visual_tokens", "reground", "live_feedback", "echo")
#: B1: per (arm, task); 2 arms x 5 tasks x 20 = 200 turns
REPEATS = 20
#: *Regime*: the 200 turns shuffled jointly with seed 0
ORDER_SEED = 0
#: B3's fixed denominators (as T-0134's: the differences use them whatever ran)
TURNS_PER_ARM = 100
S3_TURNS = 20
#: B6: the wall-time budget, measured from ``PROCESS_START_S`` (probes included), inside the locked script's
#: ``timeout -k 60s 60m`` so the report is still written (estimate 28-35 min: T-0132 ran 300 turns in 42 min)
RUN_BUDGET_S = 50 * 60
#: B5 (H-T135-1, approved as proposed 2026-10-22): boundaries compare with this tolerance (an exact margin passes)
EPS = 1e-9
SUCCESS_MARGIN = 3
OVER_CLICK_MARGIN = 2
FALSE_CLAIM_MARGIN = 3
#: B5's labels
RECOVERED = "recover"
NO_CHANGE = "no change"
INCONCLUSIVE = "inconclusive"
VOID = "void"
INCOMPLETE = "incomplete"
#: C6b: the eligibility clauses, in the order they are checked
CLAUSES = ("acceptance", "safety", "engine")
TASK_IDS = tuple(t.task_id for t in TASKS)
#: B4: a stripped thought line this long that reaches a later body of the same turn is a leak
LEAK_MIN_CHARS = 20
#: A1's tags
OPEN_TAG = "<|tool_call>"
CLOSE_TAG = "<tool_call|>"
#: the budgets a live default may name (T-0131's arms)
BUDGETS = (280, 560, 1120)
ECHOES = ("pixels", "model")

#: STT down, enforced by the locked script's port refusal
STT_DOWN = P.STT_DOWN
#: the gate files whose sha256 pins a run to its commit (T-0132's, plus A1's rule)
GATE_FILES = (*P.GATE_FILES, "local_astra/toolcall/thought_call.py")
P4_STANDS = (
    "P4 is not re-run (B2): recovery adds no request content, so T-0132's window (9351 of 16128 tokens at 1120, 4 "
    "frames, a full continuation) holds."
)
#: the runs this A/B is set beside (history, not like-for-like)
HISTORY = (
    "T-0132 crop-budget A/B at 2ccff41 (n = 100 per arm, order seed 0): D-5 (a call left in an open thought, the turn "
    "silent) 29/300 turns, S3 18, S1 7, S5 4; every one failed",
    "T-0134 live-feedback A/B at e59aa69 (n = 100 per arm, order seed 0): success 61/64/70 by score_v2 "
    "(inconclusive: arm C had one engine timeout)",
)
#: B5, stated in the report
POWER = (
    "D-5 hits about 10 of 100 turns per arm at baseline, so recovery touches about 10 turns per arm. The rule is a "
    "non-inferiority check on a bug fix whose mechanism the CPU replay already proves (A2: 29 of 29 firings were "
    "failed, silent turns), not a gain detector; with S3 at 13/20 at baseline one arm's S3 standard error is about "
    "10.7 pp."
)


def _say(msg: str) -> None:
    print(f"t0135_thought_call_ab: {msg}", file=sys.stderr, flush=True)


# =====================================================================================================
# The live default, the arm table and the order (B1)
# =====================================================================================================


def _feedback_dict(lf: Any) -> dict[str, bool] | None:
    """A live-feedback value as the manifest records it (``None``, or its two flags)."""
    if lf is None:
        return None
    if isinstance(lf, Mapping):
        return {"screen_label": bool(lf["screen_label"]), "page_extent": bool(lf["page_extent"])}
    return {"screen_label": bool(lf.screen_label), "page_extent": bool(lf.page_extent)}


#: O-16: ``cli.Feedback.live_feedback``'s choices as (screen_label, page_extent); ``off`` is both parts off, never
#: ``None`` (T-0134c R-m)
FEEDBACK_CHOICES: dict[str, tuple[bool, bool]] = {
    "label": (True, False),
    "label+extent": (True, True),
    "off": (False, False),
}


def _feedback_of(profile: Any) -> Any:
    """O-16: ``cli.feedback_of(profile, Namespace(), "live")``, looked up at call time (``AttributeError`` when
    ``cli`` has none: ``main`` refuses before that)."""
    from local_astra import cli

    return cli.feedback_of(profile, argparse.Namespace(), "live")


def live_default(profile: Any) -> dict[str, Any]:
    """B1 (O-15, O-16): HEAD's live default as ``cli`` gives it, read at call time. The budget and ReGround are
    ``cli.grounding_of(profile, Namespace(), "live")``'s; the live feedback and the echo are
    ``cli.feedback_of(profile, Namespace(), "live")``'s: its ``live_feedback`` choice as ``FEEDBACK_CHOICES`` maps
    it, and its ``action_echo``."""
    from local_astra import cli

    g = cli.grounding_of(profile, argparse.Namespace(), "live")
    fb = _feedback_of(profile)
    label, extent = FEEDBACK_CHOICES[str(fb.live_feedback)]
    return {
        "visual_tokens": None if g.visual_tokens is None else int(g.visual_tokens),
        "reground": bool(g.reground),
        "live_feedback": {"screen_label": label, "page_extent": extent},
        "echo": str(fb.action_echo),
    }


def live_default_findings(default: Mapping[str, Any]) -> list[str]:
    """Why a live default is not one this harness can run (each key present, each value of a type the session
    takes), or ``[]``."""
    out = [f"live_default lacks {k}" for k in DEFAULT_KEYS if k not in default]
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
    return out


def arm_rows(default: Mapping[str, Any]) -> list[dict[str, Any]]:
    """B1: the two arms over a live default: its four settings in both, ``recover_thought_calls`` False in A and
    True in B."""
    shared = {k: default[k] for k in DEFAULT_KEYS}
    return [{"label": label, **shared, "recover_thought_calls": RECOVER[label]} for label in LABELS]


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
# A1/A2's rule over one request's commits (O-4)
# =====================================================================================================


def _open_block(commits: Sequence[Mapping[str, Any]]) -> str | None:
    """O-4: the request's final open thought block (the trailing run of open ``thought`` commits, concatenated), or
    ``None`` when the request kept a ``tool`` commit or its last commit is not an open ``thought``."""
    if any(c.get("segment") == "tool" for c in commits):
        return None
    if (
        not commits
        or commits[-1].get("segment") != "thought"
        or commits[-1].get("thought_open") is not True
    ):
        return None
    block: list[str] = []
    for c in reversed(commits):
        if c.get("segment") != "thought" or c.get("thought_open") is not True:
            break
        block.append(str(c.get("text") or ""))
    return "".join(reversed(block))


def call_in_thought(commits: Sequence[Mapping[str, Any]]) -> str | None:
    """T-0135 A1/A2 (O-4) over one request's ``llm.commit`` data, in order: ``thought_call.recover`` of the final
    open thought block (the text from its **last** ``<|tool_call>``, iff it ends in ``<tool_call|>`` after trailing
    whitespace) when the request kept no ``tool`` commit and its last commit is an open ``thought``; else ``None``."""
    from local_astra.toolcall.thought_call import recover

    block = _open_block(commits)
    return None if block is None else recover(block)


# =====================================================================================================
# B4: configuration findings (pure)
# =====================================================================================================


def _key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(record["arm"]), str(record["task"]), int(record["repeat"]))


def _tid(record: Mapping[str, Any]) -> str:
    """``"<arm> <task>#<repeat>"``: how a finding names a trial."""
    return f"{record['arm']} {record['task']}#{record['repeat']}"


def configuration_findings(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    thoughts: Sequence[Mapping[str, Any]],
    arms: Sequence[Mapping[str, Any]],
) -> list[str]:
    """B4, one string per finding, any of which voids the run, each naming the trial or the setting: the arm rows
    differ in a setting other than ``recover_thought_calls``; in a trial that forwarded a body (O-11, as T-0132 C6a:
    a turn that sent nothing is an acceptance failure, not a finding) a ``from_thought`` step in arm A, a
    ``from_thought`` step on a request the rule does not match, an arm-B request the rule matches without exactly
    one ``from_thought`` call (O-12), or a stripped line of 20 characters or more of a recovered request's thought
    block before the call, not already in that request's own body (O-13) nor in the turn's answer channel (O-17:
    spoken words reach later bodies as ``step_text``), found in a later body of the turn."""
    out: list[str] = []
    rows = list(arms)
    for k in sorted({k for r in rows for k in r} - {"label", "recover_thought_calls"}):
        values = [r.get(k) for r in rows]
        if any(v != values[0] for v in values):
            out.append(f"the arms differ in {k}: {values}")
    by_steps: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for s in steps:
        by_steps.setdefault(_key(s), []).append(s)
    by_thoughts = {_key(r): r for r in thoughts}
    for t in trials:
        if not t.get("scanned_requests"):
            continue
        k, tid = _key(t), _tid(t)
        rec = by_thoughts.get(k) or {}
        attempts = [str(a) for a in rec.get("attempts") or []]
        requests = list(rec.get("requests") or [])
        bodies = list(rec.get("body_texts") or [])
        flagged = [s for s in by_steps.get(k, []) if s.get("from_thought") is True]
        tails = {
            attempts[i]: call_in_thought(requests[i])
            for i in range(min(len(attempts), len(requests)))
        }
        # O-17: the turn's answer channel, each request's answer commits joined (a commit boundary can split a
        # line); a spoken line reaches later bodies legitimately, as ``step_text``
        spoken = "\n".join(
            "".join(str(c.get("text") or "") for c in req if c.get("segment") == "answer")
            for req in requests
        )
        for s in flagged:
            if t["arm"] == "A":
                out.append(f"{tid} step {s['step']}: a from_thought record in arm A")
            if tails.get(str(s.get("attempt_id"))) is None:
                out.append(
                    f"{tid} step {s['step']}: from_thought on a request the rule does not match"
                )
        if t["arm"] == "B":
            for a in attempts:
                if tails.get(a) is None:
                    continue
                n = sum(1 for s in flagged if str(s.get("attempt_id")) == a)
                if n != 1:
                    out.append(f"{tid} request {a}: the rule matches, {n} from_thought calls")
        for i, a in enumerate(attempts):
            if tails.get(a) is None or not any(str(s.get("attempt_id")) == a for s in flagged):
                continue
            block = _open_block(requests[i]) or ""
            before = block[: block.rfind(OPEN_TAG)]
            own = str(bodies[i] or "") if i < len(bodies) else ""
            for line in before.splitlines():
                text = line.strip()
                if len(text) < LEAK_MIN_CHARS or text in own or text in spoken:
                    continue
                for j in range(i + 1, len(bodies)):
                    if bodies[j] and text in str(bodies[j]):
                        out.append(
                            f"{tid} request {a}: a thought line reached request {attempts[j]}: {text[:60]!r}"
                        )
                        break
    return out


# =====================================================================================================
# B5: the decision (H-T135-1)
# =====================================================================================================


def _ge(a: float, b: float) -> bool:
    return float(a) >= float(b) - EPS


def _le(a: float, b: float) -> bool:
    return float(a) <= float(b) + EPS


def _first_ineligible(inputs: Mapping[str, Mapping[str, Any]]) -> tuple[str, str] | None:
    """C6b: the first ineligible arm in ``LABELS`` order and its first failing clause in ``CLAUSES`` order."""
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
    """B5's label and why: P1 failed -> inconclusive; a configuration finding (B4) -> void, reason
    ``configuration: <first>``; P2 or P3 failed (``void``) -> void; stopped early -> incomplete; an arm ineligible ->
    inconclusive naming the arm, the clause and the first failing trial; then ``recover`` iff every H-T135-1 clause
    holds, else ``no change``. ``inputs`` maps ``"A"``/``"B"`` to ``{"accepted", "safety", "engine_up", "success"
    (of 100, score_v2), "s3" (of 20), "false_claims", "over_clicks", "first_failing"}``."""
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
    ok = (
        _ge(b["success"], a["success"] - SUCCESS_MARGIN)
        and _ge(b["s3"], a["s3"])
        and _le(b["over_clicks"], a["over_clicks"] + OVER_CLICK_MARGIN)
        and _le(b["false_claims"], a["false_claims"] + FALSE_CLAIM_MARGIN)
        and int(a["safety"]) == 0
        and int(b["safety"]) == 0
    )
    return (RECOVERED, None) if ok else (NO_CHANGE, None)


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
# B3: the per-arm counts (pure)
# =====================================================================================================


def _turn(record: Mapping[str, Any]) -> tuple[str, int]:
    return (str(record["task"]), int(record["repeat"]))


def _ci(k: int, n: int) -> list[float] | None:
    """Wilson 95 % of ``k`` of ``n``; ``None`` when nothing ran."""
    return list(wilson(k, n)) if n else None


def arm_metrics(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    thoughts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """B3's counts for one arm's records (a turn counts once in every turn count): success by ``score_v2`` (with
    its Wilson interval) and per task, S3, false claims (``false_claim_v2``), over-clicks, the turns with a matching
    request (``call_in_thought``), the turns whose last request matches with no ``from_thought`` step
    (``silent_call_in_thought``), the ``final`` turns with nothing voiced (``silent_final``), and the recoveries: in
    total, by ``live.classify`` status of the recovered text, by step status, and the turns with two or more."""
    from local_astra.toolcall import live

    by = {_turn(t): [s for s in steps if _turn(s) == _turn(t)] for t in trials}
    ok = {_turn(t): score_v2(str(t["task"]), t.get("state"), G._voiced_text(t)) for t in trials}
    th = {_turn(r): r for r in thoughts}
    call_turns = silent_call = two_plus = 0
    by_verdict: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    for t in trials:
        rec = th.get(_turn(t)) or {}
        attempts = [str(a) for a in rec.get("attempts") or []]
        requests = list(rec.get("requests") or [])
        tails = [call_in_thought(req) for req in requests]
        flagged = [s for s in by[_turn(t)] if s.get("from_thought") is True]
        if any(x is not None for x in tails):
            call_turns += 1
        if tails and tails[-1] is not None:
            last = attempts[len(requests) - 1] if len(attempts) >= len(requests) else None
            if not any(str(s.get("attempt_id")) == last for s in flagged):
                silent_call += 1
        for s in flagged:
            a = str(s.get("attempt_id"))
            i = attempts.index(a) if a in attempts else -1
            tail = tails[i] if 0 <= i < len(tails) else None
            by_verdict[live.classify(tail).status if tail is not None else "unmatched"] += 1
            by_status[str(s["status"]) if s.get("status") is not None else "none"] += 1
        if len(flagged) >= 2:
            two_plus += 1
    success = sum(ok.values())
    per_task = {task: sum(v for (tt, _), v in ok.items() if tt == task) for task in TASK_IDS}
    return {
        "success": success,
        "success_ci": _ci(success, len(trials)),
        "per_task": per_task,
        "s3": per_task["S3"],
        "false_claims": sum(1 for t in trials if false_claim_v2(t, by[_turn(t)])),
        "over_clicks": sum(1 for t in trials if over_click(str(t["task"]), t.get("state"))),
        "call_in_thought": call_turns,
        "silent_call_in_thought": silent_call,
        "silent_final": sum(
            1 for t in trials if t.get("tool_loop_end") == "final" and not G._voiced_text(t).strip()
        ),
        "recoveries": {
            "total": sum(by_status.values()),
            "by_verdict": dict(by_verdict),
            "by_status": dict(by_status),
            "turns_2plus": two_plus,
        },
    }


def _arm_section(
    row: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    thoughts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """One arm's section: T-0131's (acceptance, outcomes and false claims by T-0111g's ``score``, steps, latency,
    cache, wall time, VRAM, the engine flag, every safety count) at the arm's budget and ReGround, with its two
    ``score`` keys renamed (``success_v1``, ``false_claims_v1``, as T-0134's), then B3's counts and their Wilson
    intervals."""
    base: dict[str, Any] = dict(
        T._arm_section(int(row["visual_tokens"]), bool(row["reground"]), trials, steps, regrounds)
    )
    base["success_v1"] = base.pop("success")
    base["false_claims_v1"] = base.pop("false_claims")
    m = arm_metrics(trials, steps, thoughts)
    n_s3 = sum(1 for t in trials if t["task"] == "S3")
    n_oc = sum(1 for t in trials if t["task"] in ("S1", "S2", "S5"))
    return {
        **base,
        **m,
        "label": row["label"],
        "recover_thought_calls": row["recover_thought_calls"],
        "live_feedback": row["live_feedback"],
        "echo": row["echo"],
        "n": len(trials),
        "s3_n": n_s3,
        "s3_ci": _ci(m["s3"], n_s3),
        "false_claims_ci": _ci(m["false_claims"], len(trials)),
        "over_clicks_n": n_oc,
        "over_clicks_ci": _ci(m["over_clicks"], n_oc),
        "silent_call_in_thought_ci": _ci(m["silent_call_in_thought"], len(trials)),
        "silent_final_ci": _ci(m["silent_final"], len(trials)),
    }


def _decision_inputs(
    section: Mapping[str, Any], first_failing: Mapping[str, Any]
) -> dict[str, Any]:
    """One arm's B5 input: T-0132's eligibility inputs (acceptance, safety sum, engine), B3's counts and C6b's first
    failing trial per clause."""
    return {
        **P._eligibility(section),
        "success": int(section["success"]),
        "s3": int(section["s3"]),
        "false_claims": int(section["false_claims"]),
        "over_clicks": int(section["over_clicks"]),
        "first_failing": dict(first_failing),
    }


def _diff(k1: int, n1: int, k2: int, n2: int) -> list[float]:
    """Newcombe 95 % of ``k1/n1 - k2/n2`` at B3's fixed denominators."""
    return list(newcombe(k1, n1, k2, n2))


def build_report(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    thoughts: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Any],
    *,
    arms: Sequence[Mapping[str, Any]],
    stopped: str | None = None,
) -> dict[str, Any]:
    """``report.json``: ``void`` (P1, P2 or P3 failed), ``stopped``, the probes, every arm's section, B3's Newcombe
    differences (B-A on success and the two silent counts of 100, on S3 of 20), B4's ``configuration_findings`` and
    the decision (each arm's inputs with C6b's ``first_failing``)."""
    sections: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    rows = {str(r["label"]): r for r in arms}
    for label in LABELS:
        arm_trials = [t for t in trials if t.get("arm") == label]
        arm_steps = [s for s in steps if s.get("arm") == label]
        mine = [g for g in regrounds if g.get("arm") == label]
        arm_thoughts = [r for r in thoughts if r.get("arm") == label]
        section = _arm_section(rows[label], arm_trials, arm_steps, mine, arm_thoughts)
        sections[label] = section
        inputs[label] = _decision_inputs(section, P._first_failing(arm_trials, arm_steps, mine))
    a, b = sections["A"], sections["B"]
    differences = {
        "success B-A": _diff(b["success"], TURNS_PER_ARM, a["success"], TURNS_PER_ARM),
        "s3 B-A": _diff(b["s3"], S3_TURNS, a["s3"], S3_TURNS),
        "silent_call_in_thought B-A": _diff(
            b["silent_call_in_thought"],
            TURNS_PER_ARM,
            a["silent_call_in_thought"],
            TURNS_PER_ARM,
        ),
        "silent_final B-A": _diff(
            b["silent_final"], TURNS_PER_ARM, a["silent_final"], TURNS_PER_ARM
        ),
    }
    findings = configuration_findings(trials, steps, thoughts, arms)
    p1_passed = bool(probes["p1"]["passed"])
    void = not all(bool(probes[p]["passed"]) for p in ("p1", "p2", "p3"))
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
    """``report.md``: the regime, the decision and its reason, the power statement, the probes (P4 standing), every
    arm's numbers, the differences, and the history (not like-for-like)."""
    lines = ["# T-0135 B thought-call recovery A/B (report-only, plus H-T135-1's label rule)", ""]
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
        "## Decision (B5, H-T135-1 approved as proposed)",
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
                f"- arm {label}: eligible; success {inputs['success']}, s3 {inputs['s3']}, false claims "
                f"{inputs['false_claims']}, over-clicks {inputs['over_clicks']}"
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
    p2, p3 = probes.get("p2") or {}, probes.get("p3") or {}
    lines += [
        f"- P2 (one image at {T.P2_BUDGET}): HTTP {_fmt(p2.get('http_status'))}, healthy after "
        f"{_fmt(p2.get('healthy_after'))}, passed {_fmt(p2.get('passed'))}",
        f"- P3 (text only): HTTP {_fmt(p3.get('http_status'))}, passed {_fmt(p3.get('passed'))}",
        f"- {report.get('p4') or P4_STANDS}",
        "",
        "## Arms (B3; each number report-only)",
        "",
        "| arm | recover | success v2 (Wilson 95 %) | S3 | false claims | over-clicks | call_in_thought | "
        "silent_call_in_thought | silent_final | recoveries | step done p95 ms | wall s p50/p95 | min free VRAM MiB "
        "| engine up |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label in LABELS:
        arm = report["arms"][label]
        rec = arm["recoveries"]
        lines.append(
            f"| {label} | {arm['recover_thought_calls']} | {arm['success']}/{arm['n']} "
            f"{_interval(arm['success_ci'])} | {arm['s3']}/{arm['s3_n']} | {arm['false_claims']} "
            f"{_interval(arm['false_claims_ci'])} | {arm['over_clicks']}/{arm['over_clicks_n']} | "
            f"{arm['call_in_thought']} | {arm['silent_call_in_thought']} | {arm['silent_final']} | "
            f"{rec['total']} (2+: {rec['turns_2plus']}) | {_fmt(arm['latency']['done_ms']['p95'])} | "
            f"{_fmt(arm['wall_s']['p50'])}/{_fmt(arm['wall_s']['p95'])} | {_fmt(arm['vram_min_free_mib'])} | "
            f"{arm['engine_up']} |"
        )
    lines += ["", "## Differences (Newcombe 95 %, method 10)", ""]
    lines += [f"- {k}: {_interval(v)}" for k, v in report["differences"].items()]
    for label in LABELS:
        arm = report["arms"][label]
        rec = arm["recoveries"]
        lines += [
            "",
            f"### {label} (recover_thought_calls {arm['recover_thought_calls']}, live_feedback "
            f"{arm['live_feedback']}, echo {arm['echo']})",
            "",
        ]
        lines += [
            f"- {task}: score_v2 {arm['per_task'][task]}/{o['n']}; T-0111g score {o['k']}/{o['n']}"
            for task, o in arm["outcomes"].items()
        ]
        st = arm["steps"]
        lat = arm["latency"]
        acc = arm["acceptance"]
        lines += [
            "- recoveries by verdict: "
            + (", ".join(f"{k} {v}" for k, v in rec["by_verdict"].items()) or "none")
            + "; by status: "
            + (", ".join(f"{k} {v}" for k, v in rec["by_status"].items()) or "none"),
            f"- steps per turn: {T._dist_cell(st['per_turn'])}",
            "- statuses: " + ", ".join(f"{k} {v}" for k, v in st["statuses"].items()),
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


def body_text(body: Mapping[str, Any] | None) -> str | None:
    """The ``"\\n"``-joined text parts of every message of one request body, images dropped; ``None`` without a
    body."""
    if body is None:
        return None
    messages = [m for m in body.get("messages") or [] if isinstance(m, Mapping)]
    return "\n".join(t for m in messages for t in F._texts(m))


def thought_line(
    arm: str,
    task_id: str,
    repeat: int,
    records: Sequence[Mapping[str, Any]],
    bodies: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """``thoughts.jsonl``'s line for one turn: T-0134's thought record (per ``generate``/``tool_step`` attempt, in
    order, its ``llm.commit`` data) plus ``body_texts``, parallel to ``attempts``: the text of the body the meter
    forwarded for that attempt (``body_text``), ``None`` if none was."""
    line = F.thought_record(arm, task_id, repeat, records)
    line["body_texts"] = [body_text(bodies.get(f"generate:{a}")) for a in line["attempts"]]
    return line


def from_thought_steps(
    steps: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The step lines with ``from_thought``: ``True`` iff a ``tool.call`` record of the step's call id carries
    ``from_thought: true`` (A6), read from the session log, never from the arm."""
    flagged = {
        str((r.get("data") or {}).get("call_id"))
        for r in records
        if r.get("kind") == "tool.call" and (r.get("data") or {}).get("from_thought") is True
    }
    return [
        {**s, "from_thought": s.get("call_id") is not None and str(s.get("call_id")) in flagged}
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
        recover_thought_calls_logged=None,
    )
    return trial


# =====================================================================================================
# The turn runner
# =====================================================================================================


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
]:
    """T-0134's turn at the live default (budget, ReGround with its pass-2 client as T-0132's, feedback, echo) and
    the arm's ``recover_thought_calls``; the records read after the session closed. Returns the trial, its step
    lines, its reground lines, its raw box reads and its thoughts line."""
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
            echo=echo,  # explicit in both arms (B1)
        )
        watch = T._GroundingWatch(browser, INTENDED_TARGETS[task_id])
        feedback_watch = F._FeedbackWatch(watch)
        kit, detokenizer = cli.load_tool_kit(profile, live_browser=feedback_watch)
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
            tool_kit=kit,
            tool_guidance=LIVE_TOOL_GUIDANCE,
            visual_tokens=budget,  # every setting explicit in both arms (B1)
            reground=reground,
            reground_llm=second,
            reground_visual_tokens=crop,
            live_feedback=feedback,
            recover_thought_calls=recover,  # the only difference between the arms
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
        for s in from_thought_steps(
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
        # A8 as the session logged it on the turn line (``None`` when the key is absent)
        recover_thought_calls_logged=extras.get("recover_thought_calls"),
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
    trial["deferred_prefill_aborts"] = sum(len(m.deferred_aborts) for m in meters)
    reads = [
        {"arm": arm, "task": task_id, "repeat": repeat, **c}
        for c in (watch.clicks.values() if watch is not None else [])
    ]
    thoughts = thought_line(arm, task_id, repeat, records, raw) if records else None
    return trial, steps, regrounds, reads, thoughts


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
        try:
            trial, steps, regrounds, reads, thoughts = await _one_turn(run, arm, task_id, repeat)
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
        run.turn_s.append(round(time.monotonic() - started, 1))
        _say(
            f"{index:03d} {arm} {task_id}/r{repeat} steps={len(steps)} "
            f"statuses={[s.get('status') for s in steps]} "
            f"from_thought={[s.get('from_thought') for s in steps]} "
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
        G._read_jsonl(out / "thoughts.jsonl"),
        probes,
        arms=arms,
        stopped=stopped,
    )
    G._write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report, manifest), encoding="utf-8")
    return report


def _group_a2_missing() -> str | None:
    """Why group A2's keyword is not there (``Session(recover_thought_calls=)``), then T-0134 group A's, or
    ``None``. Reads signatures only, at call time."""
    from local_astra.conductor import session as session_module

    if "recover_thought_calls" not in inspect.signature(session_module.Session.__init__).parameters:
        return "Session has no recover_thought_calls keyword (T-0135 A4)"
    missing = F._group_a_missing()
    return None if missing is None else str(missing)


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
    out: Path = args.out_dir or REPO / "logs" / f"t0135-thought-call-ab-{sha[:7]}"
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
    # 5. group A2's keyword (and T-0134 group A's), and the guidance every arm sends
    missing = _group_a2_missing()
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
    # 6. the live default, read from cli once (B1, O-15, O-16), and the arm table over it
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    if not callable(getattr(cli, "feedback_of", None)):
        _say(
            "refusing: cli has no feedback_of (T-0134c A2); live_default cannot read the live feedback and the "
            "action echo the arms must run at (O-16)"
        )
        return EXIT_REFUSED
    feedback = _feedback_of(lean)
    if str(feedback.live_feedback) not in FEEDBACK_CHOICES:
        _say(
            f"refusing: cli.feedback_of gives live_feedback {feedback.live_feedback!r}, not one of "
            f"{sorted(FEEDBACK_CHOICES)}"
        )
        return EXIT_REFUSED
    default = live_default(lean)
    bad = live_default_findings(default)
    if bad:
        _say(f"refusing: the live default {default} is not one this harness runs: {bad}")
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
    # 7. the tool kit (T-0115's three refusals, repair off included)
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
            kit,
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
    kit: Any,
    vram: Any,
    guidance_sha: str,
    default: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    crop: int | None,
    grounding: Any,
    feedback: Any,
) -> int:
    """Past every refusal and the engine check: the manifest, P1-P3, the turns, the report."""
    sentinel = G._sentinel_jpeg()
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    arms = [dict(r) for r in rows]
    regime = {
        **P.regime_sources(),
        "stt_source": "scripts/locked/t0135_thought_call_ab.sh refuses while 127.0.0.1:8080/8081 are bound",
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
        "live_default": dict(default),
        "live_default_grounding": _grounding_dict(grounding),
        # O-16: cli.feedback_of's result, with each value's source
        "live_default_feedback": dataclasses.asdict(feedback),
        "arm_table_sha256": arm_table_sha(arms),
        "task_table_sha256": G.task_table_sha(TASKS),
        "t0134_merged": _merged("Merge T-0134"),
        "t0134c_merged": _merged("Merge T-0134c"),
        "t0135_merged": _merged("Merge T-0135"),
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
        "turns.jsonl",
    ):
        (out / name).touch()

    # --- B2: P1-P3 (T-0131's), before any arm turn ----------------------------------------------------
    probe_start = time.monotonic()
    probes: dict[str, Any] = dict(run_probes(base_url, model))
    for name in ("p1", "p2", "p3"):
        probes[name] = {**probes[name], "passed": bool(probes[name]["ok"])}
    manifest["probe_vram_min_free_mib"] = vram.min_between(probe_start, time.monotonic())
    manifest["vram_sampling_error"] = vram.error
    G._write_json(out / "probes.json", probes)
    if not all(bool(probes[p]["passed"]) for p in ("p1", "p2", "p3")):
        stopped = "not run: a pre-flight probe failed"
        manifest["stopped"] = stopped
        G._write_json(out / "manifest.json", manifest)
        report = _write_report(out, probes, manifest, arms, stopped)
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


def _grounding_dict(grounding: Any) -> dict[str, Any] | str:
    """``cli.grounding_of`` at HEAD with no flags, in ``live``, with each value's source."""
    return dataclasses.asdict(grounding) if dataclasses.is_dataclass(grounding) else repr(grounding)


if __name__ == "__main__":
    sys.exit(main())
