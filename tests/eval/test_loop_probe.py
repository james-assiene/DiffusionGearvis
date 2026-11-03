"""T-0115 criterion 9: the multi-step tool-loop probe's outcome, re-derived from ``trials.jsonl`` and ``steps.jsonl``
alone (``gpu``, ``eval``).

Run through ``/launch`` on the 5090 box (step 4 of ``scripts/locked/t0115_loop_probe.sh``)::

    T0115_RUN_DIR=logs/t0115-loop-probe-<sha7> uv run pytest tests/eval/test_loop_probe.py -q -m gpu -rA

Without ``T0115_RUN_DIR`` every row skips with that reason. The rows spend no GPU time -- they read the run
directory -- but they are ``gpu`` because the data exists only after a ``/launch``-reviewed run (T-0046).

**Gate-file drift.** Before anything else the module compares the manifest's ``gate_file_sha256`` (repo-relative path
-> sha256 of ``local_astra/conductor/session.py``, ``local_astra/toolcall/loop.py`` and
``local_astra/conductor/turn_machine.py``, criterion 5) with the working tree, and skips with ``SKIP_REASON`` when any
differs or is missing.

**The oracle is this module, not the harness.** ``derive`` applies *Pre-registered rules* of docs/tasks/T-0115.md to
the record fields named in ``tests/unit/test_t0115_loop_probe.py``'s docstring, in ``fractions.Fraction``, with every
threshold typed here from the task file; it re-counts exact repeats from each turn's executed calls (brute force) and
re-reads G6 from the voiced texts and sources (A2'), the thought commits and the stored history. ``report.json`` must agree on ``void``,
both controls, every bar's ``holds`` and the outcome (its form, and the bars a "not ready" names).

Regime (state it beside every number this prints): T-0115 *Regime* -- lean; vLLM v0.29.0; NVFP4 diffusiongemma-26B-A4B
rev ``2ea8372``; template ``36e3a42e``; tokenizer ``cc8d3a0c``; CL 256, 48 steps; thinking on; prefix caching on;
``max_num_seqs 3``; RTX 5090, driver 595.84, gpu-mem 0.86, one fresh boot; tools ``native:bbfc53ccdd80:loop8``, dry
run, repair on (step 1 only), ``ScenarioSearch``; typed text, FakeTts 300 ms/word on the session clock; 118 turns
(S0 8, S1-S5 20 each, S6 10); concurrency 1; unseeded; ``MAX_CONTINUATION`` 14 000, ``MAX_STEP_TEXT`` 1 000,
``MAX_ARG_TEXT`` 500, ``max_prompt_tokens`` 14 336.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall.special_tokens import SPECIAL_TOKENS

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0115_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]
GATE_FILES = (
    "local_astra/conductor/session.py",
    "local_astra/toolcall/loop.py",
    "local_astra/conductor/turn_machine.py",
)
SKIP_REASON = (
    "re-derives against the working tree's gate files (session.py, toolcall/loop.py, turn_machine.py); "
    "reproduces only at the run commit"
)
OUTCOMES = frozenset({"final", "call", "refused", "repairable-refused", "thought-open", "empty"})
BARS = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
BAR_RE = re.compile(r"\bG[1-7]\b")

# --- *Pre-registered rules*, typed out ---------------------------------------------------------------------------
TURNS = {"S0": 8, "S1": 20, "S2": 20, "S3": 20, "S4": 20, "S5": 20, "S6": 10}
VOID_AT = Fraction(3, 118)
C1_MIN = Fraction(32, 40)
C2_MIN = 60
G1_MIN = Fraction(9, 10)
G2_MIN = Fraction(16, 20)
G3_MAX = Fraction(10, 100)
G4_EXACT_MAX = Fraction(1, 10)
G4_S5_MAX = Fraction(4, 20)
G5_S1_MIN = Fraction(16, 20)
G5_S2_MIN = Fraction(14, 20)
MAX_PROMPT_TOKENS = 14_336
#: the thought markers (also special literals) a voiced text must never hold
THOUGHT_MARKERS = ("<|channel>", "<channel|>", "<|think|>")


# =====================================================================================================
# Reading the run
# =====================================================================================================


def _jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            assert isinstance(item, dict), f"{path.name}: a line is not an object"
            out.append(item)
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV)
    if not raw:
        pytest.skip(
            f"{ENV} is not set: this row reads a T-0115 run directory (scripts/locked/t0115_loop_probe.sh)"
        )
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    for name in ("manifest.json", "trials.jsonl", "steps.jsonl", "report.json"):
        assert (path / name).is_file(), f"criterion 5/6: {path} has no {name}"
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    pinned = manifest.get("gate_file_sha256") or {}
    drift = [rel for rel in GATE_FILES if pinned.get(rel) != _sha256(REPO / rel)]
    if drift:
        pytest.skip(f"{SKIP_REASON}: {', '.join(drift)} differ from the manifest")
    return path


def _key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(record["scenario"]), str(record["prompt_id"]), int(record["repeat"]))


def by_turn(
    steps: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], list[Mapping[str, Any]]]:
    out: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for s in steps:
        out.setdefault(_key(s), []).append(s)
    return out


# =====================================================================================================
# The oracle
# =====================================================================================================


def _call_key(c: Mapping[str, Any]) -> str:
    return json.dumps([c.get("name"), c.get("arguments")], sort_keys=True)


def exact_repeats(executed: Sequence[Mapping[str, Any]]) -> int:
    """Executed calls equal to ANY earlier executed call of the turn (brute force)."""
    keys = [_call_key(c) for c in executed]
    return sum(1 for i, k in enumerate(keys) if k in keys[:i])


def errored(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """V: a harness exception, a non-200 other than 413, or a turn not IDLE within 120 s wall."""
    if trial.get("exception") or trial.get("timed_out"):
        return True
    return any(s.get("http_status") not in (200, 413, None) for s in steps)


def sane(step: Mapping[str, Any]) -> bool:
    """G1: a final (non-empty, thought closed: the class says so), a valid call, or a refusal with step text."""
    outcome = step.get("outcome")
    if outcome in ("final", "call"):
        return True
    return outcome in ("refused", "repairable-refused") and bool(
        str(step.get("step_text") or "").strip()
    )


def _collapsed(text: object) -> str:
    """A2' (orchestrator, 2026-10-17): every side of G6's thought clause is whitespace-collapsed."""
    return " ".join(str(text).split())


def g6_findings(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> list[str]:
    """G6 as ruled on 2026-10-17 (orchestrator, on reviewer A and on the re-reviews): A2', the thought clause is "a
    voiced *source* (the step's ``voiced_sources``, the raw ``TtsSpeak.source`` texts) that occurs inside a thought
    commit of the turn and does not occur in that step's step text", every side whitespace-collapsed; the
    TTS-normalised ``voiced`` texts are not the compared side. A step line without ``voiced_sources`` is itself a
    finding (the clause cannot be read). A1, the history clause reads only turns that finished (no exception, not
    timed out), which are V's otherwise."""
    tag = f"{trial.get('scenario')}/{trial.get('prompt_id')}/r{trial.get('repeat')}"
    out: list[str] = []
    thoughts = [_collapsed(t) for t in trial.get("thought_commits") or [] if str(t).strip()]
    for s in steps:
        voiced = [str(v) for v in s.get("voiced") or [] if str(v).strip()]
        step_text = _collapsed(s.get("step_text") or "")
        if int(s["step"]) >= 2 and s.get("executed") and voiced:
            out.append(f"{tag}: voiced at step {s['step']}, which executed its call")
        if "voiced_sources" not in s:
            out.append(f"{tag}: step {s['step']} has no voiced_sources (A2'), G6 cannot be read")
            continue
        for src in s.get("voiced_sources") or []:
            # the literal clause reads the raw source (ruled 2026-10-17 at 20ea438): the TTS text has them stripped
            hits = [lit for lit in (*SPECIAL_TOKENS, *THOUGHT_MARKERS) if lit in str(src)]
            if hits:
                out.append(f"{tag}: voiced special literal(s) {hits} at step {s['step']}")
            text = _collapsed(src)
            if text and any(text in t for t in thoughts) and text not in step_text:
                out.append(f"{tag}: voiced thought text not in the step text at step {s['step']}")
    if not trial.get("exception") and not trial.get("timed_out"):
        played = [str(p) for p in trial.get("played_runs") or [] if str(p)]
        joined = " ".join(played) or None
        stored = trial.get("history") or None
        if stored != joined:
            out.append(f"{tag}: history {stored!r} is not the played join {joined!r}")
    return out


def derive(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The rule of *Pre-registered rules*, from the records alone."""
    turns = by_turn(steps)
    of = {sid: [t for t in trials if t["scenario"] == sid] for sid in TURNS}

    def steps_of(t: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return turns.get(_key(t), [])

    n_err = sum(1 for t in trials if errored(t, steps_of(t)))
    void = Fraction(n_err, len(trials)) >= VOID_AT if trials else True

    s12 = of["S1"] + of["S2"]
    c1_k = sum(
        1
        for t in s12
        if any(
            int(s["step"]) == 1
            and s.get("outcome") == "call"
            and s.get("call_name") == "google_search"
            for s in steps_of(t)
        )
    )
    later = [s for s in steps if int(s["step"]) >= 2]
    controls = {
        "C1": bool(s12) and Fraction(c1_k, len(s12)) >= C1_MIN,
        "C2": len(later) >= C2_MIN,
    }

    def share(rows: Sequence[Mapping[str, Any]], pred: Any) -> Fraction:
        return Fraction(sum(1 for t in rows if pred(t)), len(rows)) if rows else Fraction(0)

    g1 = bool(later) and Fraction(sum(1 for s in later if sane(s)), len(later)) >= G1_MIN
    g2 = (
        share(
            of["S4"],
            lambda t: t.get("tool_steps_executed") == 1 and t.get("tool_loop_end") == "final",
        )
        >= G2_MIN
        and share(
            of["S1"],
            lambda t: (
                t.get("tool_loop_end") == "final"
                and 1 <= int(t.get("tool_steps_executed") or 0) <= 2
            ),
        )
        >= G2_MIN
    )
    s1_5 = [t for sid in ("S1", "S2", "S3", "S4", "S5") for t in of[sid]]
    g3 = share(s1_5, lambda t: t.get("tool_loop_end") == "budget") <= G3_MAX
    s1_4 = [t for sid in ("S1", "S2", "S3", "S4") for t in of[sid]]
    calls = sum(len(t.get("executed") or []) for t in s1_4)
    reps = sum(exact_repeats(t.get("executed") or []) for t in s1_4)
    thrice = share(
        of["S5"],
        lambda t: (
            max(Counter(_call_key(c) for c in t.get("executed") or []).values(), default=0) >= 3
        ),
    )
    g4 = (calls == 0 or Fraction(reps, calls) <= G4_EXACT_MAX) and thrice <= G4_S5_MAX
    g5 = (
        share(of["S1"], lambda t: t.get("fact_hit") is True) >= G5_S1_MIN
        and share(of["S2"], lambda t: t.get("fact_hit") is True) >= G5_S2_MIN
    )
    findings = [f for t in trials for f in g6_findings(t, steps_of(t))]
    g6 = findings == []
    g7 = not any(s.get("http_status") == 413 for s in later) and not any(
        int(s.get("prefix_query_tokens") or 0) > MAX_PROMPT_TOKENS for s in later
    )
    bars = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6, "G7": g7}
    failing = {b for b, ok in bars.items() if not ok}
    if void or not all(controls.values()):
        kind = "inconclusive"
    elif failing:
        kind = "not ready"
    else:
        kind = "ready"
    return {
        "void": void,
        "errored": n_err,
        "controls": controls,
        "bars": bars,
        "failing": failing,
        "kind": kind,
        "findings": findings,
        "repeats": (reps, calls),
    }


def _kind(outcome: str) -> str:
    for kind in ("ready", "not ready", "inconclusive"):
        if outcome == kind or outcome.startswith(f"{kind}:"):
            return kind
    raise AssertionError(f"the outcome {outcome!r} is none of ready / not ready: / inconclusive:")


# =====================================================================================================
# Rows
# =====================================================================================================


def test_the_run_is_complete(run_dir: Path) -> None:
    """Criterion 5: 118 trial lines, each (scenario, prompt, repeat) once, in the pre-registered counts; each turn's
    step lines are steps 1..k in order, every request id one of the turn's own; every outcome class is known."""
    trials = _jsonl(run_dir / "trials.jsonl")
    steps = _jsonl(run_dir / "steps.jsonl")
    keys = [_key(t) for t in trials]
    assert len(trials) == sum(TURNS.values()) == 118 and len(set(keys)) == len(keys), len(trials)
    assert Counter(k[0] for k in keys) == TURNS, Counter(k[0] for k in keys)
    turns = by_turn(steps)
    assert set(turns) <= set(keys), sorted(set(turns) - set(keys))[:5]
    for t in trials:
        mine = turns.get(_key(t), [])
        assert [int(s["step"]) for s in mine] == list(range(1, len(mine) + 1)), (_key(t), mine)
        assert {s["request_id"] for s in mine} <= set(t.get("request_ids") or []), _key(t)
    unknown = sorted({str(s.get("outcome")) for s in steps} - OUTCOMES)
    assert unknown == [], f"unknown outcome classes {unknown}"


def test_repeats_re_derived_from_the_executed_calls(run_dir: Path) -> None:
    """G4's input: every trial's exact-repeat count is the brute-force count over its own executed calls (a repeat
    is equal to any earlier executed call of the turn, not only the previous one)."""
    for t in _jsonl(run_dir / "trials.jsonl"):
        got = (t.get("repeats") or {}).get("exact")
        assert got == exact_repeats(t.get("executed") or []), (_key(t), got, t.get("executed"))


def test_every_bar_re_derived_agrees_with_the_report(run_dir: Path) -> None:
    """Criterion 9: ``void``, C1, C2, G1-G7 and the outcome re-derived from the two files agree with
    ``report.json``."""
    trials = _jsonl(run_dir / "trials.jsonl")
    steps = _jsonl(run_dir / "steps.jsonl")
    ours = derive(trials, steps)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    print(
        f"\nT-0115 re-derived: void {ours['void']} ({ours['errored']}/118 errored), controls {ours['controls']}, "
        f"bars {ours['bars']}, exact repeats {ours['repeats'][0]}/{ours['repeats'][1]}, "
        f"G6 findings {ours['findings'][:5]}; report outcome {report.get('outcome')!r}"
    )
    assert report["void"] is ours["void"], (report["void"], ours["void"])
    assert {c: report["controls"][c] for c in ("C1", "C2")} == ours["controls"]
    assert {b: report["bars"][b]["holds"] for b in BARS} == ours["bars"]
    outcome = str(report["outcome"])
    assert _kind(outcome) == ours["kind"], (outcome, ours["kind"])
    if ours["kind"] == "not ready":
        assert set(BAR_RE.findall(outcome)) == ours["failing"], (outcome, ours["failing"])
