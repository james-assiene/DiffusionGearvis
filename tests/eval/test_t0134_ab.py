"""T-0134 B3-B6 and B8: the 3-arm live-feedback A/B re-derived from the run's records and compared with
``report.json`` (``gpu``, ``eval``; report-only plus the H-T134-1 decision rule).

Run through ``/launch`` on the 5090 box (the eval step of ``scripts/locked/t0134_feedback_ab.sh``)::

    T0134_RUN_DIR=logs/t0134-feedback-ab-<sha7> uv run pytest tests/eval/test_t0134_ab.py -q -m gpu -rA

Every row skips when ``T0134_RUN_DIR`` is unset or the directory has no ``report.json``. They spend no GPU time but are
``gpu`` because the data exists only after a ``/launch``-reviewed run.

**The oracle is this module, not the harness** (B8: ported, never imported from the harness). It reuses the other
*eval oracles* (T-0111g's ``succeeded``, T-0131's ``false_claim`` shape, T-0132's ``wilson``/``newcombe``), never a
script under ``scripts/``. ``tests/unit/test_t0134_ab_harness.py`` runs its pure rows against this module too, so a
wrong oracle is caught on CPU.

**Record fields read** (pinned in ``tests/unit/test_t0134_ab_harness.py``'s docstring): every trial, step and reground
line carries ``arm`` (``"A"``/``"B"``/``"C"``), ``task``, ``repeat`` and T-0131's fields; a trial also
``scanned_requests``, ``budgets_sent`` and ``guidance_sent`` (O-11: only a trial with a non-empty ``scanned_requests``
is judged by B5, as T-0132 C6a; one that sent nothing may carry ``None``); a step also ``has_image`` (its body carried an image), ``label`` (the text
part right after the current frame in its body, or ``None``), ``label_count`` (label stems in the whole body),
``extent_known`` (the session logged extent numbers for it), ``executed_arguments`` and ``action`` (an executed
click/type's arguments and action string). ``thoughts.jsonl``: ``{"arm", "task", "repeat", "thoughts": [...],
"requests": [[commit, ...], ...]}`` per turn, one list per ``generate``/``tool_step`` request holding that request's
``llm.commit`` data (``segment``, ``text``, ``thought_open``) in order (O-10: D-5 is counted by T-0135 A2's
structural rule over these, not by a thought regex). ``report.json``: ``{"void", "stopped", "probes", "arms": {A, B, C: section},
"differences": {...}, "decision": {"label", "reason", "inputs"}, "configuration_findings": [...]}``.

Regime (state it beside every number): T-0134 *Regime*, GPU -- profile lean, CL 256, 48 steps, thinking on; vLLM
0.29.0, NVFP4 rev 2ea8372; ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16; ``services/vllm.sh`` at 806da12, one
boot; tools native bbfc53ccdd80, live, repair on, loop 8, ``LIVE_TOOL_GUIDANCE`` 768a973b; 1120 visual tokens and
ReGround off in every arm; headless Chromium 156 at 1280x800; typed turns, fake STT/TTS/audio; 3 arms x 5 tasks x 20 =
300 turns, order seed 0, concurrency 1; the engine unseeded; STT down.
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

from local_astra.conductor.tool_executor import to_pixels
from tests.eval import test_live_smoke as g0111
from tests.eval import test_t0131_ab as t0131
from tests.eval import test_t0132_ab as t0132

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0134_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]

#: B1, typed from docs/tasks/T-0134.md (arm A's echo is "pixels", B1's sentence below the table)
ARM_TABLE: tuple[dict[str, Any], ...] = (
    {
        "label": "A",
        "visual_tokens": 1120,
        "reground": False,
        "live_feedback": None,
        "echo": "pixels",
    },
    {
        "label": "B",
        "visual_tokens": 1120,
        "reground": False,
        "live_feedback": {"screen_label": True, "page_extent": False},
        "echo": "model",
    },
    {
        "label": "C",
        "visual_tokens": 1120,
        "reground": False,
        "live_feedback": {"screen_label": True, "page_extent": True},
        "echo": "model",
    },
)
#: sha256 of ``json.dumps(list(ARM_TABLE), sort_keys=True, separators=(",", ":"))``, by the test-author 2026-10-22
ARM_TABLE_SHA256 = "7a87f4278b5811e7996e9d554f3fe3c191e049a5471cc50609732dfeb532b352"
TASK_TABLE_SHA256 = t0131.TASK_TABLE_SHA256
LABELS = ("A", "B", "C")
TASK_IDS = ("S1", "S2", "S3", "S4", "S5")
REPEATS = 20
TOTAL_TURNS = 300
MAIN = 1120
EPS = 1e-9
#: B3 (H-T134-3): the S4 price forms, as written in the criterion
S4_PRICE_FORMS = (
    "4.75",
    "$4.75",
    "four dollars and seventy-five cents",
    "four dollars seventy-five",
    "four seventy-five",
    "four dollars and seventy five cents",
)
S4_NAME = "cotton tea towel"
#: B4's mechanism regexes (D-3, D-4)
NO_SEE = re.compile(r"(don't|do not|cannot|can't) see the (updated|new) screenshot", re.IGNORECASE)
COORD = re.compile(r"clicked at \(\d+, \d+\)")
MISMATCH_WORDS = re.compile(
    r"not the button|far off|different from|off from|expected", re.IGNORECASE
)
#: how near (characters, either side) a mismatch phrase must be to the quoted ``clicked at (n, n)``
NEAR = 200
STEM = "This screenshot is the browser screen now"
CLAUSE = "Page beyond this screen"
A3_LABEL = re.compile(
    r"^\[This screenshot is the browser screen now(\.|, after your last action\.|; your last action did not change "
    r"it\.)( Page beyond this screen: (nothing|about \d+\.\d screens) below, (nothing|about \d+\.\d screens) "
    r"above\.)?\]$"
)
#: B6 labels
BOTH = "feedback + extent"
FEEDBACK = "feedback only"
NO_CHANGE = "no change"
INCONCLUSIVE = "inconclusive"
VOID = "void"
INCOMPLETE = "incomplete"
VIA_C = "via C vs A"
CLAUSES = ("acceptance", "safety", "engine")

wilson = t0132.wilson
newcombe = t0132.newcombe


def arm_table_sha(arms: Sequence[Mapping[str, Any]] = ARM_TABLE) -> str:
    return hashlib.sha256(
        json.dumps(list(arms), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def order(seed: int) -> list[tuple[str, str, int]]:
    """The 300 ``(arm, task, repeat)`` turns, shuffled jointly (one shuffle of the whole list)."""
    turns = [(a, t, r) for a in LABELS for t in TASK_IDS for r in range(REPEATS)]
    random.Random(seed).shuffle(turns)
    return turns


# --------------------------------------------------------------------------------------------- B3: score_v2


def _norm(text: object) -> str:
    return " ".join(str(text).replace("-", " ").split()).casefold()


def _names_price(voiced: str) -> bool:
    said = _norm(voiced)
    if _norm(S4_NAME) not in said:
        return False
    for form in S4_PRICE_FORMS:
        f = _norm(form)
        if re.search(rf"(?<![\w.]){re.escape(f)}(?![\w]|\.\d)", said):
            return True
    return False


def score_v2(task_id: str, state: Any, voiced: str) -> bool:
    """T-0111g's ``score`` except S4: an empty cart, the name, and a pinned price form (casefolded, whitespace
    collapsed, hyphens as spaces; a form is a token sequence of its own: ``$14.75`` does not name ``4.75``)."""
    if task_id != "S4":
        return g0111.succeeded(task_id, state, voiced)
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
    return NO_SEE.search(thought) is not None


def coord_mismatch(thought: str) -> bool:
    for m in COORD.finditer(thought):
        window = thought[max(0, m.start() - NEAR) : m.end() + NEAR]
        if MISMATCH_WORDS.search(window):
            return True
    return False


_OPEN_TAG = "<|tool_call>"
_CLOSE_TAG = "<tool_call|>"


def silent_call_in_thought(commits: Sequence[Mapping[str, Any]]) -> bool:
    """B4 ``silent_call_in_thought`` (O-10; T-0135 A1/A2's rule, ported): one request's ``llm.commit`` records, in
    order. It fires iff the request kept no ``tool`` commit, its last non-tool commit is a ``thought`` commit with
    ``thought_open: true``, and the final thought block (the trailing run of ``thought`` commits, texts concatenated)
    from its **last** ``<|tool_call>`` on ends, after trailing whitespace, in ``<tool_call|>``."""
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
    """T-0131's proxy with ``score_v2`` as the success test."""
    if trial.get("exception") or trial.get("timed_out") or not steps:
        return False
    if score_v2(str(trial["task"]), trial.get("state"), g0111.voiced_text(trial)):
        return False
    last = max(steps, key=lambda s: int(s["step"]))
    if last.get("status") is not None or last.get("call_name") is not None:
        return False
    said = " ".join(str(v) for v in last.get("voiced_sources") or [])
    return bool(said.strip()) and "?" not in said


# --------------------------------------------------------------------------------------------- B5


def _tid(r: Mapping[str, Any]) -> str:
    return f"{r['arm']} {r['task']}#{r['repeat']}"


_ACTION = re.compile(r"^clicked at \((\d+), (\d+)\)")


def echo_finding(step: Mapping[str, Any], viewport: tuple[int, int] = (1280, 800)) -> str | None:
    """B5: an executed click/type whose action coordinates are not the call's own (B, C) or not ``to_pixels`` of
    them (A); a point where both readings agree is not judged."""
    if step.get("status") != "executed" or step.get("executed_name") not in ("click_at", "type_at"):
        return None
    m = _ACTION.match(str(step.get("action") or ""))
    args = step.get("executed_arguments") or {}
    if m is None or "x" not in args:
        return f"{_tid(step)} step {step['step']}: no action coordinates"
    got = (int(m.group(1)), int(m.group(2)))
    own = (int(args["x"]), int(args["y"]))
    px = (to_pixels(own[0], viewport[0]), to_pixels(own[1], viewport[1]))
    if own == px:
        return None
    want = px if step["arm"] == "A" else own
    if got != want:
        return f"{_tid(step)} step {step['step']}: echo {got}, expected {want}"
    return None


def label_finding(step: Mapping[str, Any]) -> str | None:
    """B5's label findings for one step line (see the module docstring for the fields)."""
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
    out: list[str] = []
    for t in trials:
        if not t.get("scanned_requests"):
            # O-11 (T-0132 C6a): only a trial that forwarded a body is judged; a turn that raised before sending is
            # an acceptance failure (C6b's ``inconclusive``), never a configuration finding (``void``)
            continue
        bad = [b for b in t.get("budgets_sent") or [] if b != MAIN]
        if bad or not t.get("budgets_sent"):
            out.append(f"{_tid(t)}: body budget {bad or 'none'}")
        if t.get("guidance_sent") is not True:
            out.append(f"{_tid(t)}: guidance_sent false")
    for r in regrounds:
        out.append(f"{_tid(r)}: a reground line")
    for s in steps:
        for f in (label_finding(s), echo_finding(s)):
            if f is not None:
                out.append(f)
    return out


# --------------------------------------------------------------------------------------------- B6


def _ge(a: float, b: float) -> bool:
    return a >= b - EPS


def _le(a: float, b: float) -> bool:
    return a <= b + EPS


def _first_ineligible(inputs: Mapping[str, Mapping[str, Any]]) -> tuple[str, str] | None:
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
    """B6: P1 failed -> inconclusive; P2/P3/P5 (``void``) or a configuration finding -> void; stopped ->
    incomplete; an arm ineligible -> inconclusive naming arm, clause and first failing trial; then (y), then (x)."""
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
    y = (
        _ge(b["success"], a["success"] - 3)
        and _le(b["false_claims_non_s4"], a["false_claims_non_s4"])
        and _le(b["over_clicks"], a["over_clicks"])
    )
    x = _ge(c["s4"] - b["s4"], 5) and _ge(c["success"], b["success"] - 3)
    if y and x:
        return BOTH, None
    if y:
        return FEEDBACK, None
    if _ge(c["success"] - a["success"], 5) and _le(c["false_claims"], a["false_claims"]):
        return BOTH, VIA_C
    return NO_CHANGE, None


def decide(inputs: Mapping[str, Mapping[str, Any]], **kw: Any) -> str:
    return decide_with_reason(inputs, **kw)[0]


def p5_ok(p5: Mapping[str, Any]) -> bool:
    """B2's P5: at scroll 0 ``below >= 0.3`` screens and ``above`` under 0.05 (``nothing``); after ``scroll down``
    ``above > 0``; the hostile fixture reported its true values."""
    try:
        s0, s1 = p5["scroll0"], p5["scrolled"]
        vp = float(s0["viewport_px"])
        return (
            float(s0["below_px"]) / vp >= 0.3 - EPS
            and float(s0["above_px"]) / vp < 0.05
            and float(s1["above_px"]) > 0
            and p5.get("hostile_true") is True
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


# --------------------------------------------------------------------------------------------- per-arm metrics


def derive_arm(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    thoughts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """B4's counts for one arm's records."""

    def k(r: Mapping[str, Any]) -> tuple[str, int]:
        return (str(r["task"]), int(r["repeat"]))

    by = {k(t): [s for s in steps if k(s) == k(t)] for t in trials}
    ok = {k(t): score_v2(str(t["task"]), t.get("state"), g0111.voiced_text(t)) for t in trials}
    old = {
        k(t): g0111.succeeded(str(t["task"]), t.get("state"), g0111.voiced_text(t)) for t in trials
    }
    s4 = [t for t in trials if t["task"] == "S4"]
    scrolls = sum(
        1
        for t in s4
        if any(
            s.get("executed_name") == "scroll" and s.get("status") == "executed" for s in by[k(t)]
        )
    )
    fc = [t for t in trials if false_claim_v2(t, by[k(t)])]
    th = {k(r): r for r in thoughts}
    return {
        "success": sum(ok.values()),
        # O-11: an arm with no trials (a probe-void run) has no interval
        "success_ci": wilson(sum(ok.values()), len(trials)) if trials else None,
        "per_task": {task: sum(v for (tt, _), v in ok.items() if tt == task) for task in TASK_IDS},
        "s4_v2": sum(ok[k(t)] for t in s4),
        "s4_v1": sum(old[k(t)] for t in s4),
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


# --------------------------------------------------------------------------------------------- the run (gpu)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


#: T-0131's per-arm sections the harness carries over unchanged (``success``/``false_claims`` are renamed
#: ``success_v1``/``false_claims_v1`` there, B3: T-0111g's ``score`` reported beside ``score_v2``)
T0131_SECTIONS = tuple(s for s in t0131.ARM_SECTIONS if s not in ("success", "false_claims"))


def t0131_section(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """One arm's T-0131 section (acceptance, outcomes, clicks, steps/statuses, latency, cache, wall time, VRAM, the
    engine flag, safety), re-derived by T-0131's oracle over the arm's records (1120, ReGround off in every arm),
    with B3's two renames."""
    sec = dict(t0131.derive_arm(MAIN, False, trials, steps, regrounds))
    sec["success_v1"] = sec.pop("success")
    sec["false_claims_v1"] = sec.pop("false_claims")
    return sec


def decision_inputs(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    thoughts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """B6's inputs for one arm, re-derived: T-0132's eligibility (``_arm_inputs`` over T-0131's section) and C6b's
    ``first_failing`` (T-0132's oracle), then B4's v2 counts (``derive_arm``)."""
    sec = t0131.derive_arm(MAIN, False, trials, steps, regrounds)
    m = derive_arm(trials, steps, thoughts)
    return {
        **t0132._arm_inputs(sec),
        "success": m["success"],
        "s4": m["s4_v2"],
        "false_claims": m["false_claims"],
        "false_claims_non_s4": m["false_claims_non_s4"],
        "over_clicks": m["over_clicks"],
        "first_failing": t0132.first_failing(trials, steps, regrounds),
    }


def probe_verdicts(probes: Mapping[str, Any]) -> dict[str, bool]:
    """B2, re-derived from ``probes.json`` (never from the report or a recorded ``passed``): P1 by T-0131's
    ``probe_findings`` over the recorded deltas and its conjunction; P2 an HTTP 200 with the engine healthy after;
    P3 an HTTP 200; P5 by ``p5_ok``; ``void`` iff any of P1, P2, P3, P5 failed."""
    p1 = t0131.probe_findings(probes) == [] and bool(probes["p1"].get("ok"))
    if p1:
        p1 = all(
            t0131._p1_delta_ok(int(b), row.get(which))
            for b, row in probes["p1"]["budgets"].items()
            for which in ("frame_delta", "crop_delta")
        )
    p2 = (
        probes.get("p2", {}).get("http_status") == 200 and probes["p2"].get("healthy_after") is True
    )
    p3 = probes.get("p3", {}).get("http_status") == 200
    p5 = p5_ok(probes.get("p5") or {})
    return {"p1": p1, "p2": p2, "p3": p3, "p5": p5, "void": not (p1 and p2 and p3 and p5)}


@pytest.fixture(scope="module")
def run_dir() -> Path:
    where = os.environ.get(ENV)
    if not where:
        pytest.skip(f"{ENV} is unset: the T-0134 A/B has not run (docs/tasks/T-0134.md B7)")
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


def _of(rows: Sequence[Mapping[str, Any]], arm: str) -> list[Mapping[str, Any]]:
    return [r for r in rows if r.get("arm") == arm]


def test_the_run_is_complete(run: dict[str, Any]) -> None:
    """B8: the turn order equals the 300-turn seed-0 set; the arm-table and task-table shas match."""
    report, manifest = run["report"], run["manifest"]
    if report.get("stopped"):
        pytest.skip(f"the run stopped early: {report['stopped']}")
    got = [(t["arm"], t["task"], int(t["repeat"])) for t in run["trials"]]
    assert got == order(0), "B1: the trials ran in the seed-0 joint order"
    assert manifest.get("arm_table_sha256") == ARM_TABLE_SHA256
    assert manifest.get("task_table_sha256") == TASK_TABLE_SHA256


@pytest.mark.parametrize("arm", LABELS)
def test_arm_metrics_re_derived(run: dict[str, Any], arm: str) -> None:
    """B4: every per-arm count re-derived from the jsonl files equals report.json's."""
    want = derive_arm(_of(run["trials"], arm), _of(run["steps"], arm), _of(run["thoughts"], arm))
    got = run["report"]["arms"][arm]
    for key, value in want.items():
        if key == "success_ci" and (value is None or got[key] is None):
            assert got[key] is value is None, (key, got[key], value)
        elif key == "success_ci":
            assert all(abs(a - b) < 1e-6 for a, b in zip(got[key], value, strict=True)), (
                key,
                got[key],
                value,
            )
        else:
            assert got[key] == value, (arm, key, got[key], value)


def test_configuration_findings_re_derived(run: dict[str, Any]) -> None:
    """B5: the findings re-derived from the records equal report.json's."""
    want = configuration_findings(run["trials"], run["steps"], run["regrounds"])
    assert sorted(run["report"]["configuration_findings"]) == sorted(want)


def test_probes_re_derived(run: dict[str, Any]) -> None:
    """B2: P1-P3 and P5 re-derived from probes.json (T-0131's ``probe_findings``, the HTTP statuses, ``p5_ok``);
    each recorded ``passed`` and the report's ``void`` agree with the re-derivation, and a void run is labelled
    ``inconclusive`` (P1) or ``void``."""
    probes = run["probes"]
    v = probe_verdicts(probes)
    for p in ("p1", "p2", "p3", "p5"):
        assert bool(probes[p].get("passed")) is v[p], (p, probes[p], v[p])
    assert bool(run["report"].get("void")) is v["void"], (run["report"].get("void"), v)
    if not v["p1"]:
        assert run["report"]["decision"]["label"] == INCONCLUSIVE, run["report"]["decision"]
    elif v["void"]:
        assert run["report"]["decision"]["label"] == VOID, run["report"]["decision"]


@pytest.mark.parametrize("arm", LABELS)
def test_t0131_sections_re_derived(run: dict[str, Any], arm: str) -> None:
    """B4 (the rest): acceptance, outcomes, clicks, steps and statuses, latency, cache, wall time, VRAM, the engine
    flag and the safety counts, re-derived per arm by T-0131's oracle over the records filtered on ``arm``."""
    want = t0131_section(
        _of(run["trials"], arm), _of(run["steps"], arm), _of(run["regrounds"], arm)
    )
    got = run["report"]["arms"][arm]
    for key in (*T0131_SECTIONS, "success_v1", "false_claims_v1"):
        assert got.get(key) == want[key], (arm, key, got.get(key), want[key])


@pytest.mark.parametrize("arm", LABELS)
def test_safety_counts_zero(run: dict[str, Any], arm: str) -> None:
    """B4: every T-0131/T-0132 safety count, re-derived, is 0."""
    want = t0131_section(
        _of(run["trials"], arm), _of(run["steps"], arm), _of(run["regrounds"], arm)
    )
    assert all(int(v) == 0 for v in want["safety"].values()), (arm, want["safety"])


@pytest.mark.parametrize("arm", LABELS)
def test_decision_inputs_re_derived(run: dict[str, Any], arm: str) -> None:
    """B6: each arm's decision inputs (eligibility, ``first_failing``, B4's v2 counts) re-derived from the records,
    never read from the report's sections."""
    want = decision_inputs(
        _of(run["trials"], arm),
        _of(run["steps"], arm),
        _of(run["regrounds"], arm),
        _of(run["thoughts"], arm),
    )
    assert run["report"]["decision"]["inputs"][arm] == want, (
        arm,
        run["report"]["decision"]["inputs"][arm],
        want,
    )


def test_decision_re_derived(run: dict[str, Any]) -> None:
    """B6: the label and reason from re-derived inputs, probes, findings and ``stopped``."""
    inputs = {
        arm: decision_inputs(
            _of(run["trials"], arm),
            _of(run["steps"], arm),
            _of(run["regrounds"], arm),
            _of(run["thoughts"], arm),
        )
        for arm in LABELS
    }
    v = probe_verdicts(run["probes"])
    label, reason = decide_with_reason(
        inputs,
        p1_passed=v["p1"],
        void=v["void"],
        stopped=run["report"].get("stopped"),
        configuration_findings=configuration_findings(
            run["trials"], run["steps"], run["regrounds"]
        ),
    )
    d = run["report"]["decision"]
    assert (d["label"], d.get("reason")) == (label, reason)


def test_differences_re_derived(run: dict[str, Any]) -> None:
    """B4: the Newcombe intervals for B-A, C-B, C-A (success), C-B (S4), B-A (non-S4 false claims)."""
    arms = run["report"]["arms"]
    want = {
        "success B-A": newcombe(arms["B"]["success"], 100, arms["A"]["success"], 100),
        "success C-B": newcombe(arms["C"]["success"], 100, arms["B"]["success"], 100),
        "success C-A": newcombe(arms["C"]["success"], 100, arms["A"]["success"], 100),
        "s4 C-B": newcombe(arms["C"]["s4_v2"], 20, arms["B"]["s4_v2"], 20),
        "false_claims_non_s4 B-A": newcombe(
            arms["B"]["false_claims_non_s4"], 80, arms["A"]["false_claims_non_s4"], 80
        ),
    }
    got = run["report"]["differences"]
    for key, (lo, hi) in want.items():
        assert abs(got[key][0] - lo) < 1e-6 and abs(got[key][1] - hi) < 1e-6, (
            key,
            got[key],
            (lo, hi),
        )


def test_stopped_re_derived(run: dict[str, Any]) -> None:
    """B8: ``stopped`` is set exactly when fewer than 300 trials were written."""
    assert bool(run["report"].get("stopped")) == (len(run["trials"]) < TOTAL_TURNS)
