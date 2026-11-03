"""T-0135 B2-B5 and B7: the 2-arm thought-call recovery A/B re-derived from the run's records and compared with
``report.json`` (``gpu``, ``eval``; report-only plus the H-T135-1 decision rule).

Run through ``/launch`` on the 5090 box (the eval step of ``scripts/locked/t0135_thought_call_ab.sh``)::

    T0135_RUN_DIR=logs/t0135-thought-call-ab-<sha7> uv run pytest tests/eval/test_t0135_ab.py -q -m gpu -rA

Every run row skips when ``T0135_RUN_DIR`` is unset or the directory has no ``report.json``. They spend no GPU time but
are ``gpu`` because the data exists only after a ``/launch``-reviewed run (T-0046: every ``tests/eval`` row is
deselected by a path-only run).

**The oracle is this module, not the harness** (B7: ported, never imported from the harness). It reuses the other
*eval oracles* (T-0111g's ``succeeded``/``voiced_text``, T-0131's ``derive_arm``/``probe_findings``, T-0132's
``wilson``/``newcombe``/``_arm_inputs``/``first_failing``, T-0134's ``score_v2``/``over_click``/``false_claim_v2``) and
the product's ``live.classify`` (the verdict of a recovered call is ``live.classify``'s by A6), never a script under
``scripts/``. T-0135 A1's rule is **ported** here (``call_in_thought``), not imported from
``local_astra/toolcall/thought_call.py``. ``tests/unit/test_t0135_ab_harness.py`` runs the pure rows against this
module too, so a wrong oracle is caught on CPU.

**Record fields read** (pinned in ``tests/unit/test_t0135_ab_harness.py``'s docstring):

- every trial, step and thought line carries ``arm`` (``"A"``/``"B"``), ``task`` and ``repeat``;
- a trial also carries T-0111g's/T-0131's fields (``state``, ``voiced``, ``scanned_requests``, ``tool_loop_end``,
  ``visual_tokens``, ``reground``, ...) and ``recover_thought_calls`` (its arm's value) and
  ``recover_thought_calls_logged`` (the A8 key on its turn line, ``None`` when absent);
- a step also carries T-0131's fields (``attempt_id``, ``call_id``, ``status``, ``reason``, ``call_name``, ...) and
  ``from_thought`` (``True`` iff a ``tool.call`` record of the step's call carries ``from_thought: true``);
- ``thoughts.jsonl``, one line per turn: ``{"arm", "task", "repeat", "attempts": [attempt_id, ...], "requests":
  [[{"segment", "text", "thought_open"}, ...], ...], "body_texts": [str | None, ...]}``, ``requests`` and
  ``body_texts`` parallel to ``attempts`` (one entry per ``generate``/``tool_step`` request in order: its
  ``llm.commit`` data, and the ``"\\n"``-joined text parts of every message of the body the meter forwarded for it,
  images dropped, ``None`` if none was forwarded);
- ``manifest.json``: ``arm_table_sha256``, ``task_table_sha256``, ``order``, ``live_default`` (``{"visual_tokens",
  "reground", "live_feedback": None | {"screen_label", "page_extent"}, "echo"}`` as the harness read it from
  ``cli`` at run time) and ``regime.arms`` (``arm_rows`` of that default). Nothing here hard-codes the default: the
  arm rows, the arm-table sha and each arm's T-0131 budget/ReGround are re-derived from ``live_default``;
- ``report.json``: ``{"void", "stopped", "probes", "arms": {"A", "B"}, "differences", "configuration_findings",
  "decision": {"label", "reason", "inputs"}}``.

**Readings** (test-author, reported; see the unit module's docstring for the full list):

- *Silent endings* are two counts: ``silent_call_in_thought``, turns whose **last** request matches A1/A2's rule and
  carries no ``from_thought`` step (the turn ended with its call left in a thought), and ``silent_final``, turns whose
  ``tool_loop_end`` is ``final`` with nothing voiced. ``call_in_thought`` (turns with any matching request) is
  reported beside them: it is the mechanism's rate, expected alike in both arms.
- B4's leak finding reads "a line of the thought block" as a line (stripped, at least 20 characters) of the block
  **before** the recovered call -- the call itself reaches the next request by design, as ``render_call``'s text,
  which is byte-identical to the model's -- not already in the recovered request's own body (a line the thought
  quoted from the context reaches later bodies through the context, not through the thought), and (O-17) not in an
  answer-channel commit of the same turn (spoken words reach later bodies as ``step_text``; each request's answer
  commits are joined before the match, since a commit boundary can split a line).
- An ineligible arm already covers a safety count above 0 (T-0132 C6b's ``safety`` clause), so B5's last
  ``recover`` clause is never what decides; ``decide`` keeps it.
- ``differences`` use B3's fixed denominators (100 per arm, 20 for S3), as T-0134's did.

Regime (state it beside every number): T-0135 *Regime*, GPU -- the T-0134 regime verbatim: profile lean, CL 256, 48
steps, thinking on; vLLM 0.29.0, NVFP4 rev 2ea8372; ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16;
``services/vllm.sh`` at 806da12, one boot; tools native bbfc53ccdd80, live, repair on, loop 8; both arms at the live
default ``cli`` gives at the run's HEAD (recorded in the manifest; by the human's 2026-10-23 ruling, T-0134's arm B
-- model-unit echo plus the screen-now label -- once T-0134c lands), differing only in ``recover_thought_calls``
(A ``False``, B ``True``); headless Chromium at 1280x800;
typed turns, fake STT/TTS/audio, no headset or speakers; 2 arms x 5 tasks x 20 = 200 turns, order seed 0,
concurrency 1; the engine unseeded; STT down.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall import live
from tests.eval import test_live_smoke as g0111
from tests.eval import test_t0131_ab as t0131
from tests.eval import test_t0132_ab as t0132
from tests.eval import test_t0134_ab as t0134

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0135_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]

#: B1: the live-default settings every arm carries, read from ``cli`` by the harness at run time and recorded in the
#: manifest as ``live_default`` (never hard-coded: the human's 2026-10-23 ruling makes T-0134's arm B -- model-unit
#: echo plus the screen-now label -- the default once T-0134c lands)
DEFAULT_KEYS = ("visual_tokens", "reground", "live_feedback", "echo")
TASK_TABLE_SHA256 = t0131.TASK_TABLE_SHA256
LABELS = ("A", "B")
TASK_IDS = ("S1", "S2", "S3", "S4", "S5")
REPEATS = 20
TURNS_PER_ARM = 100
TOTAL_TURNS = 200
EPS = 1e-9
#: B5 (H-T135-1, approved as proposed 2026-10-22)
SUCCESS_MARGIN = 3
OVER_CLICK_MARGIN = 2
FALSE_CLAIM_MARGIN = 3
#: B5's labels
RECOVER = "recover"
NO_CHANGE = "no change"
INCONCLUSIVE = "inconclusive"
VOID = "void"
INCOMPLETE = "incomplete"
CLAUSES = ("acceptance", "safety", "engine")
#: B4: a thought line this long (stripped) that reaches a later body is a leak
LEAK_MIN_CHARS = 20
#: A1's tags
OPEN_TAG = "<|tool_call>"
CLOSE_TAG = "<tool_call|>"

wilson = t0132.wilson
newcombe = t0132.newcombe
score_v2 = t0134.score_v2
over_click = t0134.over_click
false_claim_v2 = t0134.false_claim_v2


def arm_rows(default: Mapping[str, Any]) -> list[dict[str, Any]]:
    """B1: the two arms over a live default: its four settings in both, ``recover_thought_calls`` False in A and
    True in B."""
    shared = {k: default[k] for k in DEFAULT_KEYS}
    return [
        {"label": "A", **shared, "recover_thought_calls": False},
        {"label": "B", **shared, "recover_thought_calls": True},
    ]


def arm_table_sha(arms: Sequence[Mapping[str, Any]]) -> str:
    """sha256 of ``json.dumps(list(arms), sort_keys=True, separators=(",", ":"))``."""
    return hashlib.sha256(
        json.dumps(list(arms), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def order(seed: int) -> list[tuple[str, str, int]]:
    """The 200 ``(arm, task, repeat)`` turns, arm-major, then one shuffle of the whole list (jointly)."""
    turns = [(a, t, r) for a in LABELS for t in TASK_IDS for r in range(REPEATS)]
    random.Random(seed).shuffle(turns)
    return turns


# --------------------------------------------------------------------------------------------- A1/A2's rule, ported


def _open_block(commits: Sequence[Mapping[str, Any]]) -> str | None:
    """O-4: the request's final open thought block (the trailing run of open ``thought`` commits, concatenated), or
    ``None`` when the request kept a ``tool`` commit or its last non-tool commit is not an open ``thought``."""
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
    """T-0135 A1/A2 (O-4) over one request's ``llm.commit`` data, in order: the recovered text (the final open thought
    block from its **last** ``<|tool_call>`` on) iff the request kept no ``tool`` commit, its last commit is an open
    ``thought`` and that tail ends, after trailing whitespace, in ``<tool_call|>``; else ``None``."""
    block = _open_block(commits)
    if block is None:
        return None
    k = block.rfind(OPEN_TAG)
    if k < 0:
        return None
    tail = block[k:]
    return tail if tail.rstrip().endswith(CLOSE_TAG) else None


# --------------------------------------------------------------------------------------------- B4


def _key(r: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(r["arm"]), str(r["task"]), int(r["repeat"]))


def _tid(r: Mapping[str, Any]) -> str:
    return f"{r['arm']} {r['task']}#{r['repeat']}"


def configuration_findings(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    thoughts: Sequence[Mapping[str, Any]],
    arms: Sequence[Mapping[str, Any]],
) -> list[str]:
    """B4, one string per finding, each naming the trial (``"<arm> <task>#<repeat>"``) or the setting. Only trials
    that forwarded a body (non-empty ``scanned_requests``; O-11, as T-0132 C6a) are judged."""
    out: list[str] = []
    rows = list(arms)
    keys = sorted({k for r in rows for k in r} - {"label", "recover_thought_calls"})
    for k in keys:
        values = [r.get(k) for r in rows]
        if any(v != values[0] for v in values):
            out.append(f"the arms differ in {k}: {values}")
    judged = [t for t in trials if t.get("scanned_requests")]
    by_steps: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for s in steps:
        by_steps.setdefault(_key(s), []).append(s)
    by_thoughts = {_key(r): r for r in thoughts}
    for t in judged:
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
        # O-17: the turn's answer channel, per request; a spoken line reaches later bodies as ``step_text``
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
            tail = tails.get(a)
            if tail is None or not any(str(s.get("attempt_id")) == a for s in flagged):
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
            "engine": bool(i["engine_up"]),
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
    """B5: P1 failed -> inconclusive; a configuration finding -> void (``configuration: <first>``); P2/P3 failed
    (``void``) -> void; stopped -> incomplete; an arm ineligible -> inconclusive naming the arm, the clause and the
    first failing trial; then ``recover`` iff every H-T135-1 clause holds, else ``no change``."""
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
    return (RECOVER, None) if ok else (NO_CHANGE, None)


def decide(inputs: Mapping[str, Mapping[str, Any]], **kw: Any) -> str:
    return decide_with_reason(inputs, **kw)[0]


# --------------------------------------------------------------------------------------------- B3


def derive_arm(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    thoughts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """B3's counts for one arm's records (a turn counts once in every turn count)."""

    def k(r: Mapping[str, Any]) -> tuple[str, int]:
        return (str(r["task"]), int(r["repeat"]))

    by = {k(t): [s for s in steps if k(s) == k(t)] for t in trials}
    ok = {k(t): score_v2(str(t["task"]), t.get("state"), g0111.voiced_text(t)) for t in trials}
    th = {k(r): r for r in thoughts}
    call_turns = silent_call = 0
    by_verdict: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    two_plus = 0
    for t in trials:
        rec = th.get(k(t)) or {}
        attempts = [str(a) for a in rec.get("attempts") or []]
        requests = list(rec.get("requests") or [])
        tails = [call_in_thought(req) for req in requests]
        flagged = [s for s in by[k(t)] if s.get("from_thought") is True]
        if any(x is not None for x in tails):
            call_turns += 1
        if tails and tails[-1] is not None:
            last = attempts[len(requests) - 1] if len(attempts) >= len(requests) else None
            if not any(str(s.get("attempt_id")) == last for s in flagged):
                silent_call += 1
        for s in flagged:
            a = str(s.get("attempt_id"))
            tail = (
                tails[attempts.index(a)]
                if a in attempts and attempts.index(a) < len(tails)
                else None
            )
            by_verdict[live.classify(tail).status if tail is not None else "unmatched"] += 1
            by_status[str(s.get("status")) if s.get("status") is not None else "none"] += 1
        if len(flagged) >= 2:
            two_plus += 1
    success = sum(ok.values())
    per_task = {task: sum(v for (tt, _), v in ok.items() if tt == task) for task in TASK_IDS}
    return {
        "success": success,
        # O-11: an arm with no trials (a probe-void run) has no interval
        "success_ci": list(wilson(success, len(trials))) if trials else None,
        "per_task": per_task,
        "s3": per_task["S3"],
        "false_claims": sum(1 for t in trials if false_claim_v2(t, by[k(t)])),
        "over_clicks": sum(1 for t in trials if over_click(str(t["task"]), t.get("state"))),
        "call_in_thought": call_turns,
        "silent_call_in_thought": silent_call,
        "silent_final": sum(
            1
            for t in trials
            if t.get("tool_loop_end") == "final" and not g0111.voiced_text(t).strip()
        ),
        "recoveries": {
            "total": sum(by_status.values()),
            "by_verdict": dict(by_verdict),
            "by_status": dict(by_status),
            "turns_2plus": two_plus,
        },
    }


#: T-0131's per-arm sections the harness carries over (``success``/``false_claims`` renamed ``success_v1``/
#: ``false_claims_v1``, as T-0134's: T-0111g's ``score`` reported beside ``score_v2``)
T0131_SECTIONS = t0134.T0131_SECTIONS


def t0131_section(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """One arm's T-0131 section (acceptance, outcomes, steps per turn and statuses, step latency, cache, turn wall,
    VRAM, the engine flag, every safety count), re-derived by T-0131's oracle at the arm's budget and ReGround."""
    sec = dict(
        t0131.derive_arm(int(row["visual_tokens"]), bool(row["reground"]), trials, steps, regrounds)
    )
    sec["success_v1"] = sec.pop("success")
    sec["false_claims_v1"] = sec.pop("false_claims")
    return sec


def decision_inputs(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    thoughts: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """B5's inputs for one arm, re-derived: T-0132's eligibility over T-0131's section and C6b's ``first_failing``
    (T-0132's oracle: the arm-level clause per trial), then B3's counts (``derive_arm``)."""
    sec = t0131.derive_arm(
        int(row["visual_tokens"]), bool(row["reground"]), trials, steps, regrounds
    )
    m = derive_arm(trials, steps, thoughts)
    return {
        **t0132._arm_inputs(sec),
        "success": m["success"],
        "s3": m["s3"],
        "false_claims": m["false_claims"],
        "over_clicks": m["over_clicks"],
        "first_failing": t0132.first_failing(trials, steps, regrounds),
    }


def probe_verdicts(probes: Mapping[str, Any]) -> dict[str, bool]:
    """B2, re-derived from ``probes.json`` (never from the report or a recorded ``passed``): P1 by T-0131's
    ``probe_findings`` and the recorded deltas; P2 an HTTP 200 with the engine healthy after; P3 an HTTP 200;
    ``void`` iff any failed. P4 is not re-run (B2)."""
    p1 = t0131.probe_findings(probes) == [] and bool((probes.get("p1") or {}).get("ok"))
    if p1:
        p1 = all(
            t0131._p1_delta_ok(int(b), row.get(which))
            for b, row in probes["p1"]["budgets"].items()
            for which in ("frame_delta", "crop_delta")
        )
    p2 = (probes.get("p2") or {}).get("http_status") == 200 and probes["p2"].get(
        "healthy_after"
    ) is True
    p3 = (probes.get("p3") or {}).get("http_status") == 200
    return {"p1": p1, "p2": p2, "p3": p3, "void": not (p1 and p2 and p3)}


def differences(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
    """B3: Newcombe 95 % of B-A at B3's fixed denominators (100 per arm, 20 for S3), from ``derive_arm``'s counts."""
    a, b = arms["A"], arms["B"]
    return {
        "success B-A": newcombe(b["success"], TURNS_PER_ARM, a["success"], TURNS_PER_ARM),
        "s3 B-A": newcombe(b["s3"], REPEATS, a["s3"], REPEATS),
        "silent_call_in_thought B-A": newcombe(
            b["silent_call_in_thought"], TURNS_PER_ARM, a["silent_call_in_thought"], TURNS_PER_ARM
        ),
        "silent_final B-A": newcombe(
            b["silent_final"], TURNS_PER_ARM, a["silent_final"], TURNS_PER_ARM
        ),
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


def _row(run: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    """The arm's row, re-derived from the manifest's recorded live default (never the harness's table)."""
    rows = arm_rows((run.get("manifest") or {}).get("live_default") or {})
    return next(r for r in rows if r["label"] == arm)


def live_default_findings(default: Mapping[str, Any]) -> list[str]:
    """B1: the recorded live default is well formed (each key present, each value of a type the session takes)."""
    out = [f"live_default lacks {k}" for k in DEFAULT_KEYS if k not in default]
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
    return out


@pytest.fixture(scope="module")
def run_dir() -> Path:
    where = os.environ.get(ENV)
    if not where:
        pytest.skip(f"{ENV} is unset: the T-0135 A/B has not run (docs/tasks/T-0135.md B6)")
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
        "thoughts": _jsonl(run_dir / "thoughts.jsonl"),
    }


def _inputs(run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        arm: decision_inputs(
            _of(run["trials"], arm),
            _of(run["steps"], arm),
            _of(run["regrounds"], arm),
            _of(run["thoughts"], arm),
            _row(run, arm),
        )
        for arm in LABELS
    }


def test_the_run_is_complete(run: dict[str, Any]) -> None:
    """B7 completeness: the trials ran in the 200-turn seed-0 joint order; the manifest's order is that order; the
    manifest's arm rows are B1's two rows over its recorded live default, and its arm-table sha is theirs; the
    task-table sha matches its pin."""
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
    """B1: the manifest records a well-formed live default (read from ``cli`` at run time), every arm row carries
    it, and every trial that ran a session ran at it (its ``visual_tokens``, ``reground``, ``live_feedback`` and
    ``echo``)."""
    recorded = run["manifest"].get("live_default") or {}
    assert live_default_findings(recorded) == [], (recorded, live_default_findings(recorded))
    for row in _manifest_arms(run):
        assert {k: row.get(k) for k in DEFAULT_KEYS} == {k: recorded[k] for k in DEFAULT_KEYS}, row
    for t in run["trials"]:
        if t.get("scanned_requests"):
            got = {k: t.get(k) for k in DEFAULT_KEYS}
            assert got == {k: recorded[k] for k in DEFAULT_KEYS}, (_tid(t), got, recorded)


#: T-0135b A7: the trials of a known run that wrote no turn line, by the run's ``manifest.regime.commit`` (the e1ae76e
#: run: B S2#5 hit the turn timeout, ``timed_out: true``, ``session_id: null``)
NO_TURN_LINE_BY_RUN = {"e1ae76e36f70aa5404be7dbb14566296ac3fbc5a": ["B S2#5"]}


def wrote_no_turn_line(t: Mapping[str, Any]) -> bool:
    """T-0135b A7: a trial that hit the turn timeout (``timed_out`` true) or never had a session (``session_id``
    ``None``) wrote no turn line, so it has no A8 key to log."""
    return t.get("timed_out") is True or t.get("session_id") is None


def arm_setting_findings(trials: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """A8 evidence, as corrected by T-0135b A7: ``(findings, exempted)``. Every trial that scanned requests must carry
    its arm's manifest-level ``recover_thought_calls`` (``False`` in A, ``True`` in B), the timed-out ones too; the
    turn-line key ``recover_thought_calls_logged`` must equal it too, except on a trial that wrote no turn line
    (``wrote_no_turn_line``), whose ``_tid`` is listed in ``exempted``. Never keyed on ``success``."""
    want = {"A": False, "B": True}
    findings: list[str] = []
    exempted: list[str] = []
    for t in trials:
        if not t.get("scanned_requests"):
            continue
        arm, tid = str(t["arm"]), _tid(t)
        if t.get("recover_thought_calls") is not want[arm]:
            findings.append(f"{tid}: recover_thought_calls {t.get('recover_thought_calls')!r}")
        if wrote_no_turn_line(t):
            exempted.append(tid)
            continue
        if t.get("recover_thought_calls_logged") is not want[arm]:
            findings.append(
                f"{tid}: recover_thought_calls_logged {t.get('recover_thought_calls_logged')!r}"
            )
    return findings, exempted


def test_arm_setting_reached_every_turn(run: dict[str, Any]) -> None:
    """A8 evidence: every trial that ran a session carries its arm's ``recover_thought_calls`` (``False`` in A,
    ``True`` in B) and logged the same on its turn line.

    T-0135b A7 (the eval exceeded T-0135 A8's "every live turn line"): only the ``recover_thought_calls_logged``
    assertion is exempted, and only for a trial that wrote no turn line (``timed_out`` true or ``session_id``
    ``None``); the manifest-level assertion still runs on it. On the e1ae76e run exactly one trial is exempted, B
    S2#5. Adversarial, on records built here: a trial with a turn line but no ``_logged`` key still fails (whatever
    its ``success``), and a timed-out trial with the wrong manifest-level value still fails."""
    findings, exempted = arm_setting_findings(run["trials"])
    assert findings == [], findings
    commit = str(((run.get("manifest") or {}).get("regime") or {}).get("commit") or "")
    if commit in NO_TURN_LINE_BY_RUN:
        assert exempted == NO_TURN_LINE_BY_RUN[commit], (
            f"T-0135b A7: exactly {NO_TURN_LINE_BY_RUN[commit]} exempted on {commit[:7]}; got {exempted}"
        )
    base = {"task": "S2", "repeat": 99, "scanned_requests": ["x"], "recover_thought_calls": True}
    turn_line = {**base, "arm": "B", "timed_out": False, "session_id": "s-1"}
    for success in (False, True):
        got, ex = arm_setting_findings([{**turn_line, "success": success}])
        assert got == ["B S2#99: recover_thought_calls_logged None"] and ex == [], (
            success,
            got,
            ex,
        )
    timed_out = {**base, "arm": "B", "timed_out": True, "session_id": None, "success": False}
    assert arm_setting_findings([timed_out]) == ([], ["B S2#99"])
    got, ex = arm_setting_findings([{**timed_out, "recover_thought_calls": False}])
    assert got == ["B S2#99: recover_thought_calls False"] and ex == ["B S2#99"], (got, ex)


@pytest.mark.parametrize("arm", LABELS)
def test_arm_metrics_re_derived(run: dict[str, Any], arm: str) -> None:
    """B3: every per-arm count re-derived from the jsonl files equals report.json's."""
    want = derive_arm(_of(run["trials"], arm), _of(run["steps"], arm), _of(run["thoughts"], arm))
    got = run["report"]["arms"][arm]
    for key, value in want.items():
        if key == "success_ci" and (value is None or got[key] is None):
            assert got[key] is value is None, (key, got[key], value)
        elif key == "success_ci":
            assert all(abs(x - y) < 1e-6 for x, y in zip(got[key], value, strict=True)), (
                key,
                got[key],
                value,
            )
        else:
            assert got[key] == value, (arm, key, got[key], value)


@pytest.mark.parametrize("arm", LABELS)
def test_t0131_sections_re_derived(run: dict[str, Any], arm: str) -> None:
    """B3 (the rest): steps per turn, step latency, turn wall p50/p95, the prefix-cache hit, the minimum free VRAM,
    acceptance, the engine flag and every safety count, re-derived per arm by T-0131's oracle."""
    want = t0131_section(
        _of(run["trials"], arm), _of(run["steps"], arm), _of(run["regrounds"], arm), _row(run, arm)
    )
    got = run["report"]["arms"][arm]
    for key in (*T0131_SECTIONS, "success_v1", "false_claims_v1"):
        assert got.get(key) == want[key], (arm, key, got.get(key), want[key])


@pytest.mark.parametrize("arm", LABELS)
def test_safety_counts_zero(run: dict[str, Any], arm: str) -> None:
    """B3: every T-0131/T-0132 safety count, re-derived from the records, is 0."""
    want = t0131_section(
        _of(run["trials"], arm), _of(run["steps"], arm), _of(run["regrounds"], arm), _row(run, arm)
    )
    assert all(int(v) == 0 for v in want["safety"].values()), (arm, want["safety"])


def test_configuration_findings_re_derived(run: dict[str, Any]) -> None:
    """B4: the findings re-derived from the records and the manifest's arm rows equal report.json's."""
    want = configuration_findings(run["trials"], run["steps"], run["thoughts"], _manifest_arms(run))
    assert sorted(run["report"]["configuration_findings"]) == sorted(want), (
        run["report"]["configuration_findings"],
        want,
    )


def test_probes_re_derived(run: dict[str, Any]) -> None:
    """B2: P1-P3 re-derived from probes.json; each recorded ``passed`` and the report's ``void`` agree with the
    re-derivation, and a failed P1 is labelled ``inconclusive``, a failed P2/P3 ``void``."""
    probes = run["probes"]
    v = probe_verdicts(probes)
    for p in ("p1", "p2", "p3"):
        assert bool(probes[p].get("passed")) is v[p], (p, probes[p], v[p])
    assert bool(run["report"].get("void")) is v["void"], (run["report"].get("void"), v)
    if not v["p1"]:
        assert run["report"]["decision"]["label"] == INCONCLUSIVE, run["report"]["decision"]
    elif v["void"]:
        assert run["report"]["decision"]["label"] == VOID, run["report"]["decision"]


@pytest.mark.parametrize("arm", LABELS)
def test_decision_inputs_re_derived(run: dict[str, Any], arm: str) -> None:
    """B5: each arm's decision inputs (eligibility, ``first_failing``, B3's counts) re-derived from the records,
    never read from the report's sections."""
    want = _inputs(run)[arm]
    assert run["report"]["decision"]["inputs"][arm] == want, (
        arm,
        run["report"]["decision"]["inputs"][arm],
        want,
    )


def test_decision_re_derived(run: dict[str, Any]) -> None:
    """B5: the label and reason from re-derived inputs, probes, findings and the trial count (``stopped`` is the
    report's string only when fewer than 200 trials ran: ``test_stopped_re_derived``)."""
    v = probe_verdicts(run["probes"])
    label, reason = decide_with_reason(
        _inputs(run),
        p1_passed=v["p1"],
        void=v["void"],
        stopped=run["report"].get("stopped") if len(run["trials"]) < TOTAL_TURNS else None,
        configuration_findings=configuration_findings(
            run["trials"], run["steps"], run["thoughts"], _manifest_arms(run)
        ),
    )
    d = run["report"]["decision"]
    assert (d["label"], d.get("reason")) == (label, reason)


def test_differences_re_derived(run: dict[str, Any]) -> None:
    """B3: the Newcombe intervals for B-A on success, S3 and the two silent-ending counts, from counts re-derived
    from the records (never the report's sections)."""
    arms = {
        arm: derive_arm(_of(run["trials"], arm), _of(run["steps"], arm), _of(run["thoughts"], arm))
        for arm in LABELS
    }
    got = run["report"]["differences"]
    for key, (lo, hi) in differences(arms).items():
        assert abs(got[key][0] - lo) < 1e-6 and abs(got[key][1] - hi) < 1e-6, (
            key,
            got[key],
            (lo, hi),
        )
