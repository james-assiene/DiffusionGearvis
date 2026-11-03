"""T-0132 C3, C4 and C6: the 3-arm crop-budget A/B's metrics, configuration evidence and decision label, re-derived
from the run's records and compared with ``report.json`` (``gpu``, ``eval``; report-only, plus the decision rule,
H-T132-1, -4, -5).

Run through ``/launch`` on the 5090 box (the eval step of ``scripts/locked/t0132_crop_budget_ab.sh``)::

    T0132_RUN_DIR=logs/t0132-crop-budget-ab-<sha7> uv run pytest tests/eval/test_t0132_ab.py -q -m gpu -rA

Every row skips when ``T0132_RUN_DIR`` is unset or the directory has no ``report.json``. They spend no GPU time but
are ``gpu`` because the data exists only after a ``/launch``-reviewed run.

**The oracle is this module, not the harness** (C6h: ported, never imported from the harness). Per arm, T-0131's own
oracle (``tests/eval/test_t0131_ab.py``'s ``derive_arm``, itself over T-0111g's) gives the sections T-0131 had; this
module adds the Wilson interval of each arm's success, the Newcombe differences, the pass-2 ``images``/``budget``
distributions, C6a's configuration findings, C6b's per-arm eligibility (``first_failing``) and the decision rule.
``tests/unit/test_t0132_ab_harness.py`` checks ``wilson``, ``newcombe``, ``decide``/``decide_with_reason``,
``pass2_budget_ok`` and ``derive`` here against hand values on CPU (its ``eval-oracle`` rows), so a wrong oracle is
caught without a GPU run.

**Record fields read** (the harness's, pinned in ``tests/unit/test_t0132_ab_harness.py``'s docstring): every trial,
step and reground line carries T-0131's fields plus ``reground_visual_tokens`` (``None`` in the off arm); a reground
line also carries ``images`` and ``budget`` (the session's ``tool.reground`` record, B5) and ``body_budget`` and
``body_images`` (the pass-2 body the ``LoopMeter`` scanned, C6e). ``report.json``: ``{"void", "stopped", "probes",
"arms": {label: section}, "differences": {"on1120-off", "on280-off", "on280-on1120"}: [lo, hi], "decision":
{"label", "reason", "inputs"}, "configuration_findings": [...], "window": "ok" | "over" | "failed"}``.

**Readings (reported):** C6b names the first ineligible arm in arm-table order (1120/off, 1120/on/1120,
1120/on/280) and, within it, the first failing clause in the order acceptance, safety, engine. A trial fails a clause
exactly when the arm-level clause fails over that one trial's records (T-0131's ``derive_arm`` on ``[trial]``, review of
4c67db4 finding 1), so an ineligible arm always names a trial. A pass-2 request is judged on its ``body_images``
(C6e). Each arm's ``configuration`` section (report-only) is re-derived as ``{"trials_with_bodies",
"main_budget_only", "guidance_sent", "pass2_budget_findings"}``, a finding being ``"<task>#<repeat> step <step>"``.

Regime (state it beside every number this prints): T-0132 *Regime*, GPU -- profile lean, CL 256, 48 steps, thinking
on; vLLM 0.29.0, NVFP4 rev 2ea8372; ``max_num_seqs`` 3, prefix caching on, KV 4 GiB bf16; ``services/vllm.sh`` at
806da12, one boot; tools native bbfc53ccdd80, live, repair on, loop 8, the live guidance on; headless Chromium 156 at
1280x800; typed turns, fake STT/TTS/audio, no headset or speakers; 3 arms x 5 tasks x 20 = 300 turns, order seed 0,
concurrency 1; the engine unseeded; STT down.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.eval import test_live_smoke as g0111
from tests.eval import test_t0131_ab as t0131

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0132_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]

#: C1, typed from docs/tasks/T-0132.md: (visual tokens, ReGround, pass-2 budget)
ARMS: tuple[tuple[int, bool, int | None], ...] = (
    (1120, False, None),
    (1120, True, 1120),
    (1120, True, 280),
)
#: sha256 of ``json.dumps([{"visual_tokens": b, "reground": r, "reground_visual_tokens": c} for b, r, c in ARMS],
#: sort_keys=True, separators=(",", ":"))``, computed by the test-author on 2026-10-22 from the table above
ARM_TABLE_SHA256 = "3943ca94ce9f9de83aaa285b37e8ce511cf61f69a93ac58d0cf1b28cbc978e4f"
TASK_TABLE_SHA256 = t0131.TASK_TABLE_SHA256
TASK_IDS = t0131.TASK_IDS
REPEATS = 20
TURNS_PER_ARM = 100
TOTAL_TURNS = 300
MAIN = 1120
Z95 = 1.959963984540054
#: C4 (H-T132-1)
CROP_SUCCESS_MARGIN = 5
CROP_HIT_MARGIN = 0.03
CROP_P50_FACTOR = 0.8
ON_SUCCESS_GAIN = 5
ON_HIT_MARGIN = 0.02
EPS = 1e-9
#: C4's labels
ON_280 = "reground on, crop 280"
ON_1120 = "reground on, crop 1120"
OFF = "reground off"
INCONCLUSIVE = "inconclusive"
VOID = "void"
INCOMPLETE = "incomplete"
#: C6b: the clauses, in the order they are checked
CLAUSES = ("acceptance", "safety", "engine")
SKIPPED = "skipped_no_point"
ABORTED = "pass2_timeout"
CLICK_TOOLS = t0131.CLICK_TOOLS
OUTCOMES = t0131.OUTCOMES
STT_DOWN = "STT down (ports 8080/8081 free)"


def arm_label(visual_tokens: int, reground: bool, crop: int | None) -> str:
    """``"1120/off"``, ``"1120/on/1120"``, ``"1120/on/280"``."""
    return (
        f"{int(visual_tokens)}/off" if not reground else f"{int(visual_tokens)}/on/{int(crop or 0)}"
    )


LABEL_OFF, LABEL_ON1120, LABEL_ON280 = (arm_label(*a) for a in ARMS)
LABELS = (LABEL_OFF, LABEL_ON1120, LABEL_ON280)


def arm_of(record: Mapping[str, Any]) -> str:
    return arm_label(
        int(record["visual_tokens"]), bool(record["reground"]), record.get("reground_visual_tokens")
    )


def _key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (arm_of(record), str(record["task"]), int(record["repeat"]))


def _trial_id(record: Mapping[str, Any]) -> str:
    return f"{record['task']}#{int(record['repeat'])}"


# =====================================================================================================
# The oracle (pure; the unit module imports these)
# =====================================================================================================


def wilson(k: int, n: int) -> tuple[float, float]:
    """The Wilson score 95 % interval of ``k`` successes out of ``n``."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    d = 1 + Z95 * Z95 / n
    centre = (p + Z95 * Z95 / (2 * n)) / d
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe's hybrid score 95 % interval (method 10) for ``k1/n1 - k2/n2``, from the two Wilson intervals."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt(
        (u1 - p1) ** 2 + (p2 - l2) ** 2
    )


def pass2_budget_ok(body: Mapping[str, Any], images: int, *, main: int, crop: int) -> bool:
    """C3: a pass-2 body carries the crop budget when its prompt holds one image, and the main budget otherwise."""
    kwargs = body.get("mm_processor_kwargs")
    sent = kwargs.get("max_soft_tokens") if isinstance(kwargs, Mapping) else None
    return sent is not None and sent == (crop if int(images) == 1 else main)


def _eligible(i: Mapping[str, Any]) -> bool:
    return bool(i["accepted"]) and int(i["safety"]) == 0 and bool(i["engine_up"])


def _failing_clause(i: Mapping[str, Any]) -> str | None:
    if not bool(i["accepted"]):
        return "acceptance"
    if int(i["safety"]) != 0:
        return "safety"
    if not bool(i["engine_up"]):
        return "engine"
    return None


def _ge(a: Any, b: float) -> bool:
    return a is not None and float(a) >= b - EPS


def _le(a: Any, b: float) -> bool:
    return a is not None and float(a) <= b + EPS


def _crop_280(inputs: Mapping[str, Mapping[str, Any]]) -> bool:
    """C4 (b), every arm eligible (C6b runs first): the crop default becomes 280 only if every clause holds."""
    lo, hi = inputs[LABEL_ON280], inputs[LABEL_ON1120]
    if hi["hit_rate"] is None or hi["pass2_done_p50_ms"] is None or hi["pass2_done_p95_ms"] is None:
        return False
    return (
        int(lo["success"]) >= int(hi["success"]) - CROP_SUCCESS_MARGIN
        and _ge(lo["hit_rate"], float(hi["hit_rate"]) - CROP_HIT_MARGIN)
        and _le(lo["pass2_done_p50_ms"], CROP_P50_FACTOR * float(hi["pass2_done_p50_ms"]))
        and _le(lo["pass2_done_p95_ms"], float(hi["pass2_done_p95_ms"]))
    )


def decide_with_reason(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    p1_passed: bool,
    void: bool = False,
    stopped: str | None = None,
    configuration_findings: Sequence[str] = (),
) -> tuple[str, str | None]:
    """C6b's order: P1 failed -> inconclusive; P2/P3 failed or a configuration finding -> void (``configuration:
    <first>``); stopped -> incomplete; any arm ineligible -> inconclusive (the arm, the clause, the first failing
    trial); then (b), then (a)."""
    if not p1_passed:
        return INCONCLUSIVE, "P1 failed"
    if void:
        return VOID, "P2 or P3 failed"
    if configuration_findings:
        return VOID, f"configuration: {configuration_findings[0]}"
    if stopped is not None:
        return INCOMPLETE, f"stopped early: {stopped}"
    for label in LABELS:
        i = inputs[label]
        clause = _failing_clause(i)
        if clause is not None:
            trial = (i.get("first_failing") or {}).get(clause)
            return INCONCLUSIVE, f"{label} ineligible: {clause} failed first at {trial}"
    return _rule(inputs), None


def _rule(inputs: Mapping[str, Mapping[str, Any]]) -> str:
    """C4 (b) then (a), every arm eligible."""
    chosen = LABEL_ON280 if _crop_280(inputs) else LABEL_ON1120
    on, off = inputs[chosen], inputs[LABEL_OFF]
    keep = (
        int(on["success"]) - int(off["success"]) >= ON_SUCCESS_GAIN
        and off["hit_rate"] is not None
        and _ge(on["hit_rate"], float(off["hit_rate"]) - ON_HIT_MARGIN)
    )
    if not keep:
        return OFF
    return ON_280 if chosen == LABEL_ON280 else ON_1120


def decide(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    p1_passed: bool,
    void: bool = False,
    stopped: str | None = None,
    configuration_findings: Sequence[str] = (),
) -> str:
    return decide_with_reason(
        inputs,
        p1_passed=p1_passed,
        void=void,
        stopped=stopped,
        configuration_findings=configuration_findings,
    )[0]


def configuration_findings(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
) -> list[str]:
    """C6a, one string per finding: every trial that forwarded a body (non-empty ``scanned_requests``) sent 1120 on
    its non-reground bodies (``budgets_sent == [1120]``) and the guidance (``guidance_sent is True``); every pass-2
    request's scanned body carries the arm's crop at one image and 1120 otherwise, with ``images == body_images`` and
    ``budget == body_budget``; an off arm has no reground line; every executed coordinate click in an on arm has a
    reground line."""
    out: list[str] = []
    for t in trials:
        if not t.get("scanned_requests"):
            continue
        where = f"{arm_of(t)} {_trial_id(t)}"
        if t.get("budgets_sent") != [MAIN]:
            out.append(f"{where}: budgets_sent {t.get('budgets_sent')!r}, not [{MAIN}]")
        if t.get("guidance_sent") is not True:
            out.append(f"{where}: guidance_sent {t.get('guidance_sent')!r}")
    for g in regrounds:
        where = f"{arm_of(g)} {_trial_id(g)} step {g.get('step')}"
        if not g.get("reground"):
            out.append(f"{where}: a reground line in the ReGround-off arm")
            continue
        if g.get("outcome") == SKIPPED:
            continue
        crop = int(g.get("reground_visual_tokens") or MAIN)
        if g.get("images") != g.get("body_images"):
            out.append(f"{where}: images {g.get('images')} != body_images {g.get('body_images')}")
        if g.get("budget") != g.get("body_budget"):
            out.append(f"{where}: budget {g.get('budget')} != body_budget {g.get('body_budget')}")
        body = {"mm_processor_kwargs": {"max_soft_tokens": g.get("body_budget")}}
        if not pass2_budget_ok(body, int(g.get("body_images") or 0), main=MAIN, crop=crop):
            out.append(
                f"{where}: body_budget {g.get('body_budget')} at {g.get('body_images')} image(s), crop {crop}"
            )
    lines = {(*_key(g), int(g["step"])) for g in regrounds if g.get("step") is not None}
    for s in steps:
        if not s.get("reground") or s.get("status") != "executed":
            continue
        if s.get("executed_name") not in CLICK_TOOLS:
            continue
        if (*_key(s), int(s["step"])) not in lines:
            out.append(
                f"{arm_of(s)} {_trial_id(s)} step {s['step']}: an executed click with no reground line"
            )
    return out


def _trial_fails(
    clause: str,
    t: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    turn_lines: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    """The arm-level clause applied to this one trial's records (review of 4c67db4, finding 1): T-0131's
    ``derive_arm`` over ``[t]`` and its own steps, reground lines and turn lines, so a trial fails a clause exactly
    when it alone would make an arm fail it (a non-200 retry, ``log_complete`` False, no step line, a raised execute
    only when executed, ...), and an ineligible arm always has a first failing trial."""
    own = (
        None
        if turn_lines is None
        else [x for x in turn_lines if x.get("session_id") == t.get("session_id")]
    )
    section = t0131.derive_arm(
        int(t["visual_tokens"]), bool(t["reground"]), [t], steps, regrounds, own
    )
    i = _arm_inputs(section)
    if clause == "acceptance":
        return not i["accepted"]
    if clause == "safety":
        return i["safety"] != 0
    return not i["engine_up"]


def _arm_inputs(section: Mapping[str, Any]) -> dict[str, Any]:
    """C4 eligibility inputs of one derived arm section (acceptance, the safety sum, the engine flag)."""
    acc = section["acceptance"]
    return {
        "accepted": bool(acc["all_200"]) and acc["turns_ok"] == acc["turns"],
        "safety": sum(int(v) for v in section["safety"].values()),
        "engine_up": bool(section["engine_up"]),
    }


def first_failing(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    turn_lines: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str | None]:
    """C6b: per clause, the first trial (in trial-line order) failing it, ``"<task>#<repeat>"``, or None."""
    by_steps: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    by_rg: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for s in steps:
        by_steps.setdefault(_key(s), []).append(s)
    for g in regrounds:
        by_rg.setdefault(_key(g), []).append(g)
    out: dict[str, str | None] = dict.fromkeys(CLAUSES)
    for clause in CLAUSES:
        for t in trials:
            k = _key(t)
            if _trial_fails(clause, t, by_steps.get(k, []), by_rg.get(k, []), turn_lines):
                out[clause] = _trial_id(t)
                break
    return out


def arm_configuration(
    trials: Sequence[Mapping[str, Any]], regrounds: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """One arm's report-only ``configuration`` section: trials that forwarded a body, those that sent 1120 only and
    those that sent the guidance, and the pass-2 requests whose scanned body is not the arm's crop at one image
    (``body_images``) or 1120 otherwise."""
    ran = [t for t in trials if t.get("scanned_requests")]
    findings: list[str] = []
    for g in regrounds:
        if g.get("outcome") == SKIPPED or not g.get("reground"):
            continue
        crop = int(g.get("reground_visual_tokens") or MAIN)
        body = {"mm_processor_kwargs": {"max_soft_tokens": g.get("body_budget")}}
        if not pass2_budget_ok(body, int(g.get("body_images") or 0), main=MAIN, crop=crop):
            findings.append(f"{g.get('task')}#{g.get('repeat')} step {g.get('step')}")
    return {
        "trials_with_bodies": len(ran),
        "main_budget_only": sum(1 for t in ran if t.get("budgets_sent") == [MAIN]),
        "guidance_sent": sum(1 for t in ran if t.get("guidance_sent") is True),
        "pass2_budget_findings": findings,
    }


def _dist_counts(values: Sequence[Any]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(Counter(values).items(), key=lambda kv: str(kv[0]))}


def derive(
    trials: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    regrounds: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Any],
    turn_lines: Sequence[Mapping[str, Any]] | None = None,
    *,
    stopped: str | None = None,
) -> dict[str, Any]:
    """Every arm's section (T-0131's oracle plus the Wilson interval and the pass-2 distributions), the three
    differences, C6a's findings and the decision (C6b's order). ``stopped`` is the caller's (the manifest's), never
    the report's."""
    arms: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for b, r, c in ARMS:
        label = arm_label(b, r, c)
        mine = [g for g in regrounds if arm_of(g) == label]
        arm_trials = [t for t in trials if arm_of(t) == label]
        arm_steps = [s for s in steps if arm_of(s) == label]
        # a session turn line carries the budget and ReGround only under ``extras`` (and ``extras["reground"]`` is the
        # counts dict), never at the top level; each arm's lines are the ones of its trials' sessions (implementer's
        # finding at b373322; shape read from logs/t0131-grounding-ab-c70baf0/turns.jsonl)
        sessions = {t.get("session_id") for t in arm_trials}
        section = t0131.derive_arm(
            b,
            r,
            arm_trials,
            arm_steps,
            mine,
            None
            if turn_lines is None
            else [t for t in turn_lines if t.get("session_id") in sessions],
        )
        k, n = int(section["success"]["k"]), int(section["success"]["n"])
        section["success"] = {"k": k, "n": n, "wilson": list(wilson(k, n)) if n else None}
        requests = [g for g in mine if g.get("outcome") != SKIPPED]
        section["pass2"] = (
            {
                "images": _dist_counts([g.get("images") for g in requests]),
                "budget": _dist_counts([g.get("budget") for g in requests]),
            }
            if r
            else None
        )
        section["configuration"] = arm_configuration(arm_trials, mine)
        arms[label] = section
        lat = (section.get("reground_cost") or {}).get("latency_ms") or {}
        inputs[label] = {
            **_arm_inputs(section),
            "success": int(section["success"]["k"]),
            "hit_rate": section["clicks"]["hit_rate"],
            "pass2_done_p50_ms": lat.get("p50"),
            "pass2_done_p95_ms": lat.get("p95"),
            "first_failing": first_failing(arm_trials, arm_steps, mine, turn_lines),
        }
    k = {lab: (arms[lab]["success"]["k"], arms[lab]["success"]["n"]) for lab in LABELS}

    def diff(a: str, b: str) -> list[float] | None:
        """``None`` when either arm ran no turn (a void or stopped run)."""
        return list(newcombe(*k[a], *k[b])) if k[a][1] and k[b][1] else None

    differences = {
        "on1120-off": diff(LABEL_ON1120, LABEL_OFF),
        "on280-off": diff(LABEL_ON280, LABEL_OFF),
        "on280-on1120": diff(LABEL_ON280, LABEL_ON1120),
    }
    findings = configuration_findings(trials, steps, regrounds)
    p1 = bool(probes["p1"]["ok"])
    void = not all(bool(probes[p]["ok"]) for p in ("p1", "p2", "p3"))
    label, reason = decide_with_reason(
        inputs, p1_passed=p1, void=void, stopped=stopped, configuration_findings=findings
    )
    return {
        "void": void,
        "arms": arms,
        "differences": differences,
        "configuration_findings": findings,
        "decision": {"label": label, "reason": reason, "inputs": inputs},
    }


# =====================================================================================================
# The run
# =====================================================================================================


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV)
    if not raw:
        pytest.skip(f"{ENV} is not set: this row reads a T-0132 A/B run")
    path = Path(raw) if Path(raw).is_absolute() else REPO / raw
    if not (path / "report.json").is_file():
        pytest.skip(f"{path} has no report.json: nothing to re-derive")
    for name in ("trials.jsonl", "steps.jsonl", "regrounds.jsonl", "manifest.json", "probes.json"):
        assert (path / name).is_file(), f"T-0132 C1: {path} has a report.json but no {name}"
    return path


@pytest.fixture(scope="module")
def run(run_dir: Path) -> dict[str, Any]:
    return {
        "trials": _jsonl(run_dir / "trials.jsonl"),
        "steps": _jsonl(run_dir / "steps.jsonl"),
        "regrounds": _jsonl(run_dir / "regrounds.jsonl"),
        "turn_lines": _jsonl(run_dir / "turns.jsonl"),
        "probes": json.loads((run_dir / "probes.json").read_text(encoding="utf-8")),
        "manifest": json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
        "report": json.loads((run_dir / "report.json").read_text(encoding="utf-8")),
        "md": (run_dir / "report.md").read_text(encoding="utf-8")
        if (run_dir / "report.md").is_file()
        else "",
    }


@pytest.fixture(scope="module")
def derived(run: dict[str, Any]) -> dict[str, Any]:
    # C6h: ``stopped`` is the manifest's record of an early stop, never the report's
    return derive(
        run["trials"],
        run["steps"],
        run["regrounds"],
        run["probes"],
        run["turn_lines"],
        stopped=run["manifest"].get("stopped"),
    )


@pytest.fixture(scope="module")
def sessions(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {p.name: _jsonl(p) for p in sorted((run_dir / "sessions").glob("*.jsonl"))}


# =====================================================================================================
# The rows
# =====================================================================================================


def test_the_run_is_complete(run: dict[str, Any]) -> None:
    """C1/C6h: the manifest pins the arm and task tables and holds the 300-turn order (each arm 100 times, each (arm,
    task) 20); unless a probe voided the run, 300 trial lines (not-run trials included), one per (arm, task,
    repeat); every step and reground line joins a trial; the off arm has no reground line; every reground outcome
    is a B3 label."""
    manifest, trials = run["manifest"], run["trials"]
    assert manifest.get("arm_table_sha256") == ARM_TABLE_SHA256, manifest.get("arm_table_sha256")
    assert manifest.get("task_table_sha256") == TASK_TABLE_SHA256, manifest.get("task_table_sha256")
    expected = {(b, r, c, t, k) for b, r, c in ARMS for t in TASK_IDS for k in range(REPEATS)}
    order = [tuple(x) for x in manifest.get("order") or []]
    assert len(order) == TOTAL_TURNS == len(set(order)) and set(order) == expected, sorted(
        set(order) ^ expected
    )[:10]
    if not trials and run["report"].get("void"):
        return
    keys = [
        (int(t["visual_tokens"]), bool(t["reground"]), t.get("reground_visual_tokens"), t["task"], int(t["repeat"]))
        for t in trials
    ]  # fmt: skip
    assert len(keys) == TOTAL_TURNS == len(set(keys)) and set(keys) == expected, Counter(keys)
    assert Counter(arm_of(t) for t in trials) == dict.fromkeys(LABELS, TURNS_PER_ARM)
    tk = {_key(t) for t in trials}
    assert [_key(s) for s in run["steps"] if _key(s) not in tk] == [], "step lines for no trial"
    assert [_key(g) for g in run["regrounds"] if _key(g) not in tk] == [], (
        "reground lines for no trial"
    )
    assert not [g for g in run["regrounds"] if arm_of(g) == LABEL_OFF], (
        "the off arm made pass-2 requests"
    )
    odd = [g.get("outcome") for g in run["regrounds"] if g.get("outcome") not in OUTCOMES]
    assert odd == [], f"reground outcomes outside B3: {odd}"


def test_probes_re_derived(run: dict[str, Any], derived: dict[str, Any]) -> None:
    """C2/C6h: ``report.probes == probes.json``; P1 re-derived from its token deltas (T-0131's oracle); the report is
    void iff P1, P2 or P3 failed, labelled inconclusive when P1 failed and void when only P2/P3 did."""
    probes, report = run["probes"], run["report"]
    assert report.get("probes") == probes, "report.json's probes are not probes.json"
    assert t0131.probe_findings(probes) == []
    assert report.get("void") is derived["void"], (report.get("void"), derived["void"])
    if not probes["p1"]["ok"]:
        assert report["decision"]["label"] == INCONCLUSIVE, report["decision"]
    elif report.get("void"):
        assert report["decision"]["label"] == VOID, report["decision"]


@pytest.mark.parametrize("section", (*t0131.ARM_SECTIONS, "success", "pass2", "configuration"))
def test_arm_section_re_derived(run: dict[str, Any], derived: dict[str, Any], section: str) -> None:
    """C3/C6h, per arm: every metric section (acceptance, clicks, latency, reground cost, cache, safety, ...),
    the success with its Wilson interval and the pass-2 distributions, re-derived from the jsonl files."""
    arms = run["report"].get("arms") or {}
    assert sorted(arms) == sorted(LABELS), sorted(arms)
    bad = {}
    for label in LABELS:
        got, want = arms[label].get(section), derived["arms"][label][section]
        if section == "success" and got and want and want.get("wilson") is not None:
            same = (got["k"], got["n"]) == (want["k"], want["n"]) and all(
                abs(a - b) < 1e-6 for a, b in zip(got["wilson"], want["wilson"], strict=True)
            )
        else:
            same = got == want
        if not same:
            bad[label] = {"report": got, "derived": want}
    assert bad == {}, json.dumps(bad, indent=1, default=str)


def test_differences_re_derived(run: dict[str, Any], derived: dict[str, Any]) -> None:
    """C3: the three Newcombe differences equal the oracle's."""
    for key, want in derived["differences"].items():
        got = run["report"]["differences"][key]
        if want is None:
            assert got is None, (key, got)
            continue
        assert all(abs(a - b) < 1e-6 for a, b in zip(got, want, strict=True)), (key, got, want)


def test_decision_re_derived(run: dict[str, Any], derived: dict[str, Any]) -> None:
    """C4/C6b/C6h: ``decision.inputs`` (``first_failing`` included) equal the derived inputs, and the label is the
    oracle's; an ineligibility reason names the arm, the clause and the first failing trial; a configuration void's
    reason starts ``configuration: ``."""
    got, want = run["report"].get("decision") or {}, derived["decision"]
    assert got.get("inputs") == want["inputs"], json.dumps(
        {"report": got.get("inputs"), "derived": want["inputs"]}, indent=1, default=str
    )
    assert got.get("label") == want["label"], (got, want)
    # review of 4c67db4, finding 1: an ineligible arm always has a first failing trial for its failing clause
    for label in LABELS:
        for inputs in (want["inputs"][label], (got.get("inputs") or {}).get(label) or {}):
            clause = _failing_clause(inputs) if "accepted" in inputs else None
            if clause is not None:
                assert (inputs.get("first_failing") or {}).get(clause) is not None, (
                    f"{label} is ineligible on {clause} but names no failing trial: {inputs}"
                )
    reason = str(got.get("reason"))
    if want["reason"] and "ineligible" in want["reason"]:
        for label in LABELS:
            clause = _failing_clause(want["inputs"][label])
            if clause is not None:
                trial = want["inputs"][label]["first_failing"][clause]
                assert label in reason and clause in reason and str(trial) in reason, reason
                break
    if derived["configuration_findings"]:
        assert reason.startswith("configuration: "), reason


def test_configuration_findings(run: dict[str, Any], derived: dict[str, Any]) -> None:
    """C6a/C6e: the report's configuration findings are the oracle's (every trial that forwarded bodies judged;
    ``images == body_images`` and ``budget == body_budget`` on every pass-2 request; every executed on-arm click
    regrounded); any finding voids the run."""
    assert run["report"].get("configuration_findings") == derived["configuration_findings"], (
        run["report"].get("configuration_findings"),
        derived["configuration_findings"],
    )
    if derived["configuration_findings"]:
        assert run["report"]["decision"]["label"] == VOID, run["report"]["decision"]


def test_body_side_evidence(run: dict[str, Any]) -> None:
    """C6e: every pass-2 request line carries ``body_images`` and ``body_budget`` from the scanned body, equal to the
    session's ``images`` and ``budget``."""
    bad = [
        {k: g.get(k) for k in ("task", "repeat", "step", "images", "body_images", "budget", "body_budget")}
        for g in run["regrounds"]
        if g.get("outcome") != SKIPPED
        and (g.get("images") != g.get("body_images") or g.get("budget") != g.get("body_budget"))
    ]  # fmt: skip
    assert bad == [], bad


def test_executed_click_has_reground(run: dict[str, Any]) -> None:
    """C6a: every executed ``click_at``/``type_at`` in an on arm has its reground line."""
    lines = {(*_key(g), int(g["step"])) for g in run["regrounds"] if g.get("step") is not None}
    missing = [
        (*_key(s), s["step"])
        for s in run["steps"]
        if s.get("reground")
        and s.get("status") == "executed"
        and s.get("executed_name") in CLICK_TOOLS
        and (*_key(s), int(s["step"])) not in lines
    ]
    assert missing == [], missing


def test_stopped_re_derived(run: dict[str, Any]) -> None:
    """C6h: ``stopped`` is the manifest's record, never the report's own; a trial written as not run implies the run
    stopped; a stopped run is ``incomplete`` (unless voided first) and report.md carries the reason."""
    report, manifest = run["report"], run["manifest"]
    assert report.get("stopped") == manifest.get("stopped"), (
        report.get("stopped"),
        manifest.get("stopped"),
    )
    not_run = [t for t in run["trials"] if str(t.get("exception") or "").startswith("not run")]
    if not_run:
        assert manifest.get("stopped") is not None, (
            f"{len(not_run)} trials not run, but stopped is null"
        )
    if manifest.get("stopped") is not None:
        assert report["decision"]["label"] in (INCOMPLETE, VOID, INCONCLUSIVE), report["decision"]
        if report["decision"]["label"] == INCOMPLETE:
            assert str(manifest["stopped"]) in run["md"], "report.md does not carry the stop reason"


def test_click_records_rescored(run: dict[str, Any]) -> None:
    """C6h: T-0131's click re-scoring (records exactly on executed clicks, five keys, ``hit``/``error_px`` re-scored
    from point and boxes), over this run's step lines."""
    t0131.test_click_records_rescored(run)


def test_sentinel_scan_covers_every_request(run: dict[str, Any], sessions: Any) -> None:
    """C6h: every ``llm.request`` attempt in ``sessions/*.jsonl``, of both clients (the main and the pass-2 client),
    has its label in its trial's ``scanned_requests``."""
    gaps = t0131.scan_coverage_gaps(run["trials"], sessions)
    assert gaps == [], "requests the sentinel scan never read:\n" + "\n".join(gaps)


def test_safety_implications(run_dir: Path, run: dict[str, Any]) -> None:
    """C6h: voiced text outside its own step's committed text implies ``voiced_uncommitted`` > 0 in that arm; a
    start-page path in the session's records implies ``url_path_or_query`` > 0 somewhere."""
    arms = run["report"].get("arms") or {}
    for label in LABELS:
        leaked = g0111.uncommitted_voicing([s for s in run["steps"] if arm_of(s) == label])
        if leaked:
            assert arms[label]["safety"]["voiced_uncommitted"] > 0, (label, leaked)
    lines = g0111.url_lines(run_dir)
    if lines:
        assert sum(arms[label]["safety"]["url_path_or_query"] for label in LABELS) > 0, lines


def test_egress_lines_all_owned(run: dict[str, Any]) -> None:
    """C6h: every ``turns.jsonl`` line belongs to a trial by ``session_id``."""
    _count, unmatched = g0111.egress_from_turns(run["trials"], run["turn_lines"])
    assert unmatched == [], f"turns.jsonl lines of sessions no trial names: {unmatched}"


def test_manifest_regime(run: dict[str, Any]) -> None:
    """C6d: the manifest's regime records the model revision, ``services/vllm.sh``'s sha256 and ref, the engine
    settings, the browser versions, STT down and the arm table; report.md carries the revision and the STT
    sentence."""
    regime = run["manifest"].get("regime") or {}
    assert "2ea8372" in str(regime.get("model_revision")), regime.get("model_revision")
    assert "806da12" in str(regime.get("vllm_sh_ref")) and regime.get("vllm_sh_sha256")
    assert regime.get("max_num_seqs") == 3 and regime.get("prefix_caching") is True
    assert regime.get("stt") == STT_DOWN
    assert regime.get("arms") == [
        {"visual_tokens": b, "reground": r, "reground_visual_tokens": c} for b, r, c in ARMS
    ]
    assert "2ea8372" in run["md"] and STT_DOWN in run["md"]


def test_window_reported(run: dict[str, Any]) -> None:
    """C2 P4: report-only; the label is one of ``ok``, ``over``, ``failed``."""
    assert run["report"].get("window") in ("ok", "over", "failed"), run["report"].get("window")
