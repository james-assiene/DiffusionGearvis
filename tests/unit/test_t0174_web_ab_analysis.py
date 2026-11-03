"""T-0174 group C-web, CW2 and CW3: the quick real-web v1/v2 A/B's analysis, a CPU function over the run directory.

**The interface this module pins (the test-author's choice; the implementer provides it):**

``scripts/gpu/t0174_web_ab_analysis.py`` (loaded by path, like the other ``scripts/gpu`` harnesses) defines

- ``DOUBT_RE: re.Pattern[str]``, compiled from C2's text exactly (``DOUBT_RE.pattern`` equals the task file's
  backticked C2 line, character for character);
- ``analyse(run_dir: pathlib.Path) -> dict[str, Any]``, pure apart from reading ``run_dir`` (it writes nothing; the
  script writes ``report.json`` from its return value and ``trials.jsonl`` from its ``"trials"``).

The run directory (what ``scripts/gpu/t0174_web_ab.sh`` writes, see ``test_t0174_web_ab_script.py``):
``order.json`` is a JSON list of ``[arm, task, repeat]`` triples (arm ``"A"`` = ``--live-guidance v1``, ``"B"`` =
``--live-guidance v2``); each trial's files are under ``turns/<arm>-<task>-<repeat>/`` exactly as the T-0160 smoke
writes them per turn: ``sessions/**/*.jsonl`` (the session log), ``turns.jsonl`` (the turn line), ``probe.out``,
``session.out`` (the session's startup output) and ``aux/connect.log`` (strace). Any of them may be missing.

The returned report (keys pinned here; more keys are allowed)::

    {
      "arms": {
        "A": {"guidance": "v1",
              "outcomes": {"W1": {"k", "n", "wilson95": [lo, hi]}, "W2": {...}},   # T-0160's rules and rounding
              "successes": int,                       # W1 k + W2 k
              "m1": int,                              # trials whose step >= 2 first-attempt thought matches DOUBT_RE
              "degenerate_aborts": int,               # trials whose turn line has extras.tool_loop_end "degenerate"
              "reasks": {"asked": int,                # sum of extras.announced_call_reask.asked
                         "outcomes": {outcome: n}},   # count of each non-null extras.announced_call_reask.outcome
              "tool_timeouts": int,                   # trials whose extras.tool_loop_end is "tool_timeout"
              "timeouts": int,                        # trials whose probe.out lacks "turn completed: True"
              "hosts_seen": [str], "step_ms_p50": float | None,
              "safety": {"voiced_uncommitted": int, "non_public_connects": int}},   # as the T-0160 smoke
        "B": {... "guidance": "v2" ...}},
      "m1_b_at_most_half_a": bool,                    # M1_B <= M1_A / 2 (report-only)
      "b_successes_at_least_a": bool,                 # B's W1+W2 successes >= A's (report-only)
      "void": [str],                                  # configuration findings; [] when none; each names the trial
      "trials": [{"arm", "task", "repeat", "m1": bool, ...}],
    }

M1 (C2, CW2): for each step s >= 2 the *first* ``llm.request`` of phase ``tool_step`` with ``step == s`` in log order
is that step's first attempt; its thought is the concatenation of the ``llm.commit`` texts of segment ``thought``
carrying its ``attempt_id``. Step 1 (phase ``generate``), re-ask attempts (phase ``announce_reask``), a second
``tool_step`` attempt of the same step (T-0171's engine-empty resend) and answer commits never count.

Void (CW1): an arm-A trial whose ``session.out`` lacks ``live guidance: v1 (flag)``, or an arm-B trial whose
``session.out`` lacks ``live guidance: v2 (flag)`` (a missing ``session.out`` lacks it).

Expected values are worked by hand below (Wilson values as in ``test_t0160_smoke_harness.test_wilson_hand_values``:
2/5 -> (0.1176, 0.7693), 3/5 -> (0.2307, 0.8824)). Regime: CPU, a synthetic run directory in tmp; no GPU, no network.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE = REPO / "scripts" / "gpu" / "t0174_web_ab_analysis.py"
TASK = REPO / "docs" / "tasks" / "T-0174.md"

#: C2's pre-registered regex, typed from the task file (U+2019 and the curly quotes as written there)
DOUBT_TEXT = (
    "(?i)(?:do not|don't|don’t|cannot|can't|can’t|haven't|have not) (?:yet )?(?:see|seen|have|got) "
    "(?:the |a |any )?[\"“”']?(?:new|next|updated|resulting|latest|current)[\"“”']? screenshot"
    "|wait(?:ing)? for (?:the |a )?[\"“”']?(?:new|next|updated|latest|current)[\"“”']? screenshot"
    "|(?:new|next|updated) screenshot (?:yet|has not|hasn't|is not|isn't)"
)
GUIDANCE = {"A": "v1", "B": "v2"}
TASKS = ("W1", "W2")
N = 5

#: voiced texts scored by T-0160's rules (W1 ``8[ ,.]?8(48|49)``, W2 contains "Example Domain")
OK = {"W1": "It is 8,849 metres high.", "W2": "The heading is Example Domain."}
BAD = {"W1": "I cannot tell yet.", "W2": "There is no heading."}
#: thoughts: a doubt each alternative of C2 counts, and plain ones it does not
DOUBT_1 = "I clicked Go. I don't have the new screenshot yet, so I should wait."  # alternative 1
DOUBT_2 = "Now I am waiting for the next screenshot before I go on."  # alternative 2
DOUBT_3 = "The new screenshot has not arrived, I think."  # alternative 3
DOUBT_V2 = "I haven't seen the “latest” screenshot, so I cannot answer."  # alternative 1, amended (v2's words)
PLAIN = "The page shows the article. The height is in the infobox."
CONTENT = "The latest screenshot is not showing the heading yet."  # content description, not doubt (C2 note)


# --------------------------------------------------------------------------------------------- the module


def analysis() -> ModuleType:
    assert MODULE.is_file(), f"T-0174 CW3: {MODULE.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0174_web_ab_analysis_unit", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{MODULE.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(spec.name, None)
    return module


def analyse(run_dir: Path) -> dict[str, Any]:
    fn = getattr(analysis(), "analyse", None)
    assert callable(fn), f"T-0174 CW3: {MODULE.name} has no analyse(run_dir)"
    report = fn(run_dir)
    assert isinstance(report, dict), report
    return report


# --------------------------------------------------------------------------------------------- the synthetic run


def trial(
    arm: str,
    task: str,
    rep: int,
    *,
    ok: bool = False,
    attempts: Iterable[tuple[str, int | None, str, str]] = (),
    end: str = "final",
    reask: tuple[int, str | None] = (0, None),
    timed_out: bool = False,
    guidance_line: str | None = "default",
    hosts: Iterable[str] = (),
    step_ms: int = 100,
    voiced: Iterable[str] | None = None,
    connects: Iterable[str] = (),
) -> dict[str, Any]:
    """One trial's spec. ``attempts``: (phase, step, segment, text) per attempt, in log order; each becomes one
    ``llm.request`` with a fresh ``attempt_id``, one ``llm.commit`` of that attempt and one ``llm.done``."""
    return {
        "arm": arm,
        "task": task,
        "rep": rep,
        "voiced": list(voiced) if voiced is not None else [OK[task] if ok else BAD[task]],
        "attempts": list(attempts),
        "end": end,
        "reask": reask,
        "timed_out": timed_out,
        "guidance_line": guidance_line,
        "hosts": list(hosts),
        "step_ms": step_ms,
        "connects": list(connects),
    }


def write_run(root: Path, trials: list[dict[str, Any]]) -> Path:
    out = root / "logs" / "t0174-web-ab-abc1234"
    out.mkdir(parents=True)
    order = [[t["arm"], t["task"], t["rep"]] for t in trials]
    (out / "order.json").write_text(json.dumps(order))
    for t in trials:
        d = out / "turns" / f"{t['arm']}-{t['task']}-{t['rep']}"
        (d / "sessions").mkdir(parents=True)
        (d / "aux").mkdir()
        sid = f"s{t['arm']}{t['task']}{t['rep']}"
        recs: list[dict[str, Any]] = []
        clock = 1000

        def rec(kind: str, t_ms: int, _recs: list[dict[str, Any]] = recs, **data: Any) -> None:
            _recs.append({"seq": len(_recs) + 1, "kind": kind, "t_ms": t_ms, "data": data})

        for i, (phase, step, segment, text) in enumerate(t["attempts"], start=1):
            rid = "t1-gen" if step in (None, 1) else f"t1-gen-s{step}"
            aid = f"{sid}-{rid}-{i}"
            req: dict[str, Any] = {"phase": phase, "request_id": rid, "attempt_id": aid}
            if step is not None and phase != "generate":
                req["step"] = step
            rec("llm.request", clock, **req)
            rec(
                "llm.commit",
                clock + 10,
                request_id=rid,
                attempt_id=aid,
                canvas_idx=0,
                segment=segment,
                text=text,
            )
            rec("llm.done", clock + t["step_ms"], request_id=rid, commits=1)
            clock += 1000
        for i, h in enumerate(t["hosts"]):
            rec("tool.call", clock + i, call_id=f"c{i}", status="executed", resolved={"hosts": [h]})
        for i, v in enumerate(t["voiced"]):
            rec(
                "llm.commit",
                clock + 50 + i,
                request_id="t1-gen",
                attempt_id=f"{sid}-v",
                segment="answer",
                text=v,
            )
            rec("tts.utterance", clock + 100 + i, text=v)
        (d / "sessions" / f"{sid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        asked, outcome = t["reask"]
        line = {
            "session_id": sid,
            "outcome": "error" if t["end"] == "degenerate" else "spoken",
            "extras": {
                "tool_loop_end": t["end"],
                "degeneracy_guard": {"on": True, "tripped": 1 if t["end"] == "degenerate" else 0},
                "announced_call_reask": {"on": True, "asked": asked, "outcome": outcome},
                "browser_egress_blocked": [],
            },
        }
        (d / "turns.jsonl").write_text(json.dumps(line) + "\n")
        (d / "probe.out").write_text(
            "turn completed: False  wall 61.0s\n"
            if t["timed_out"]
            else "turn completed: True  wall 9.0s\n"
        )
        g = t["guidance_line"]
        if g is not None:
            shown = (
                f"astra session: live guidance: {GUIDANCE[t['arm']]} (flag)\n"
                if g == "default"
                else g
            )
            (d / "session.out").write_text(
                "astra session: reask: announced_call_reask=on (default)\n"
                + shown
                + "astra session: web browsing is on: Gearvis can open public websites.\n"
            )
        (d / "aux" / "connect.log").write_text("".join(c + "\n" for c in t["connects"]))
    return out


def step1(text: str = PLAIN) -> tuple[str, int | None, str, str]:
    return ("generate", None, "thought", text)


def step(
    n: int, text: str, phase: str = "tool_step", segment: str = "thought"
) -> tuple[str, int | None, str, str]:
    return (phase, n, segment, text)


def main_trials() -> list[dict[str, Any]]:
    """The worked run. Arm A (v1):

    - W1-0 ok; step-2 first attempt says DOUBT_1 -> M1.
    - W1-1 fail; step-2 first attempt plain, the re-ask attempt (``announce_reask``, step 2) says DOUBT_1 -> not M1;
      re-ask asked 1, outcome ``no_call``.
    - W1-2 fail; step-2 first attempt DOUBT_2 -> M1; degenerate abort.
    - W1-3 ok; only step 1's thought doubts (DOUBT_1) -> not M1.
    - W1-4 fail; tool timeout; no doubt.
    - W2-0 ok; step 2 plain, step-3 first attempt DOUBT_3 -> M1.
    - W2-1 fail; the probe timed out; DOUBT_1 in the step-2 *answer* only -> not M1.
    - W2-2 ok; step-2 first attempt plain, a second ``tool_step`` attempt of step 2 (engine-empty resend) DOUBT_1
      -> not M1.
    - W2-3 fail; step-2 thought CONTENT (describes content) -> not M1.
    - W2-4 fail; step-2 first attempt "I can't see the updated screenshot" -> M1; re-ask asked 1, outcome ``call``.

    A: W1 2/5, W2 2/5, successes 4, M1 4, degenerate 1, re-asks asked 2 {no_call 1, call 1}, tool timeouts 1,
    timeouts 1.

    Arm B (v2): W1-0 ok plain; W1-1 ok, step-2 first attempt DOUBT_V2 -> M1; W1-2 ok plain; W1-3 fail, step-1 doubt
    only; W1-4 fail, degenerate; W2-0, W2-1, W2-2 ok plain; W2-3 fail, re-ask asked 1 ``no_call`` (the re-ask
    attempt says DOUBT_2: not M1); W2-4 fail plain.

    B: W1 3/5, W2 3/5, successes 6, M1 1, degenerate 1, re-asks asked 1 {no_call 1}, tool timeouts 0, timeouts 0.
    Comparisons: 1 <= 4/2 -> True; 6 >= 4 -> True.
    """
    a = [
        trial("A", "W1", 0, ok=True, attempts=[step1(), step(2, DOUBT_1)]),
        trial(
            "A",
            "W1",
            1,
            attempts=[step1(), step(2, PLAIN), step(2, DOUBT_1, phase="announce_reask")],
            reask=(1, "no_call"),
        ),
        trial("A", "W1", 2, attempts=[step1(), step(2, DOUBT_2)], end="degenerate"),
        trial("A", "W1", 3, ok=True, attempts=[step1(DOUBT_1), step(2, PLAIN)]),
        trial("A", "W1", 4, attempts=[step1(), step(2, PLAIN)], end="tool_timeout"),
        trial("A", "W2", 0, ok=True, attempts=[step1(), step(2, PLAIN), step(3, DOUBT_3)]),
        trial(
            "A",
            "W2",
            1,
            attempts=[step1(), step(2, PLAIN), step(2, DOUBT_1, segment="answer")],
            timed_out=True,
        ),
        trial("A", "W2", 2, ok=True, attempts=[step1(), step(2, PLAIN), step(2, DOUBT_1)]),
        trial("A", "W2", 3, attempts=[step1(), step(2, CONTENT)]),
        trial(
            "A",
            "W2",
            4,
            attempts=[step1(), step(2, "I can't see the updated screenshot here.")],
            reask=(1, "call"),
        ),
    ]
    b = [
        trial("B", "W1", 0, ok=True, attempts=[step1(), step(2, PLAIN)]),
        trial("B", "W1", 1, ok=True, attempts=[step1(), step(2, DOUBT_V2)]),
        trial("B", "W1", 2, ok=True, attempts=[step1(), step(2, PLAIN)]),
        trial("B", "W1", 3, attempts=[step1(DOUBT_2), step(2, PLAIN)]),
        trial("B", "W1", 4, attempts=[step1(), step(2, PLAIN)], end="degenerate"),
        trial("B", "W2", 0, ok=True, attempts=[step1(), step(2, PLAIN)]),
        trial("B", "W2", 1, ok=True, attempts=[step1(), step(2, PLAIN)]),
        trial("B", "W2", 2, ok=True, attempts=[step1(), step(2, PLAIN)]),
        trial(
            "B",
            "W2",
            3,
            attempts=[step1(), step(2, PLAIN), step(2, DOUBT_2, phase="announce_reask")],
            reask=(1, "no_call"),
        ),
        trial("B", "W2", 4, attempts=[step1(), step(2, PLAIN)]),
    ]
    # interleave the arms so nothing depends on the file order
    return [x for pair in zip(a, b, strict=True) for x in pair]


def arm(report: dict[str, Any], name: str) -> dict[str, Any]:
    arms = report.get("arms")
    assert isinstance(arms, dict) and set(arms) == {"A", "B"}, report.keys()
    got: dict[str, Any] = arms[name]
    return got


def m1_set(report: dict[str, Any]) -> set[tuple[str, str, int]]:
    trials = report.get("trials")
    assert isinstance(trials, list) and len(trials) == 2 * len(TASKS) * N, trials
    return {(t["arm"], t["task"], int(t["repeat"])) for t in trials if t["m1"]}


# --------------------------------------------------------------------------------------------- C2's regex


def test_doubt_re_is_c2s_text() -> None:
    """C2: the module's ``DOUBT_RE`` is C2's backticked line, character for character (and so is this module's)."""
    task = TASK.read_text(encoding="utf-8")
    found = re.findall(r"(?m)^`(\(\?i\)\(\?:do not.*)`$", task)
    assert found == [DOUBT_TEXT], found
    rx = getattr(analysis(), "DOUBT_RE", None)
    assert isinstance(rx, re.Pattern), f"T-0174 CW2: {MODULE.name} has no compiled DOUBT_RE"
    assert rx.pattern == DOUBT_TEXT, rx.pattern


def test_premise_thought_texts() -> None:
    """Active premise: the hand-written thoughts are classified by C2's text as this module claims."""
    rx = re.compile(DOUBT_TEXT)
    for text in (DOUBT_1, DOUBT_2, DOUBT_3, DOUBT_V2, "I can't see the updated screenshot here."):
        assert rx.search(text), text
    for text in (PLAIN, CONTENT, BAD["W1"], OK["W2"]):
        assert rx.search(text) is None, text


# --------------------------------------------------------------------------------------------- CW2, per arm


def test_outcomes_per_arm_with_wilson(tmp_path: Path) -> None:
    """CW2: W1/W2 k/5 per arm with Wilson 95 % (hand values), and the arm's guidance."""
    report = analyse(write_run(tmp_path, main_trials()))
    want = {
        "A": {"W1": (2, (0.1176, 0.7693)), "W2": (2, (0.1176, 0.7693))},
        "B": {"W1": (3, (0.2307, 0.8824)), "W2": (3, (0.2307, 0.8824))},
    }
    for name, tasks in want.items():
        got = arm(report, name)
        assert got["guidance"] == GUIDANCE[name], got.get("guidance")
        for task, (k, (lo, hi)) in tasks.items():
            o = got["outcomes"][task]
            assert (o["k"], o["n"]) == (k, N), (name, task, o)
            assert abs(o["wilson95"][0] - lo) < 1e-4 and abs(o["wilson95"][1] - hi) < 1e-4, (
                name,
                task,
                o,
            )
        assert got["successes"] == 2 * tasks["W1"][0], (name, got["successes"])


def test_m1_first_attempt_step_two_or_later_only(tmp_path: Path) -> None:
    """CW2 / C2: M1 counts the trials whose step >= 2 *first-attempt thought* matches: A 4 (W1-0, W1-2, W2-0 at
    step 3, W2-4), B 1 (W1-1, v2's words). Adversarial: step 1's doubt (A W1-3, B W1-3), a re-ask attempt's doubt
    (A W1-1, B W2-3), a second attempt's doubt (A W2-2), an answer's doubt (A W2-1) and a content description
    (A W2-3) never count."""
    report = analyse(write_run(tmp_path, main_trials()))
    assert m1_set(report) == {
        ("A", "W1", 0),
        ("A", "W1", 2),
        ("A", "W2", 0),
        ("A", "W2", 4),
        ("B", "W1", 1),
    }
    assert arm(report, "A")["m1"] == 4 and arm(report, "B")["m1"] == 1, (
        arm(report, "A")["m1"],
        arm(report, "B")["m1"],
    )


@pytest.mark.parametrize(
    ("label", "attempts", "counted"),
    [
        ("step-1 doubt", [step1(DOUBT_1), step(2, PLAIN)], False),
        (
            "re-ask attempt doubt",
            [step1(), step(2, PLAIN), step(2, DOUBT_1, phase="announce_reask")],
            False,
        ),
        ("second tool_step attempt doubt", [step1(), step(2, PLAIN), step(2, DOUBT_1)], False),
        (
            "answer-segment doubt",
            [step1(), step(2, PLAIN), step(2, DOUBT_1, segment="answer")],
            False,
        ),
        ("step-2 first-attempt doubt", [step1(), step(2, DOUBT_1)], True),
        (
            "step-4 first-attempt doubt",
            [step1(), step(2, PLAIN), step(3, PLAIN), step(4, DOUBT_3)],
            True,
        ),
    ],
    ids=[
        "step1",
        "reask_attempt",
        "second_attempt",
        "answer_segment",
        "step2_first",
        "step4_first",
    ],
)
def test_m1_one_trial(tmp_path: Path, label: str, attempts: list[Any], counted: bool) -> None:
    """Adversarial, one doubt at a time: only a step >= 2 first-attempt thought makes the trial count (A W1-0 holds
    the case; every other trial is plain)."""
    trials = [
        trial(a, t, r, attempts=[step1(), step(2, PLAIN)])
        for a in GUIDANCE
        for t in TASKS
        for r in range(N)
    ]
    trials[0] = trial("A", "W1", 0, attempts=attempts)
    report = analyse(write_run(tmp_path, trials))
    assert arm(report, "A")["m1"] == (1 if counted else 0), (label, arm(report, "A")["m1"])
    assert arm(report, "B")["m1"] == 0, label


def test_degenerate_reasks_timeouts(tmp_path: Path) -> None:
    """CW2: degenerate-thought aborts, re-asks and their outcomes (from ``extras.announced_call_reask``), tool
    timeouts (``tool_loop_end`` "tool_timeout") and probe timeouts, per arm, against the hand counts."""
    report = analyse(write_run(tmp_path, main_trials()))
    a, b = arm(report, "A"), arm(report, "B")
    assert (a["degenerate_aborts"], b["degenerate_aborts"]) == (1, 1)
    assert a["reasks"] == {"asked": 2, "outcomes": {"no_call": 1, "call": 1}}, a["reasks"]
    assert b["reasks"] == {"asked": 1, "outcomes": {"no_call": 1}}, b["reasks"]
    assert (a["tool_timeouts"], b["tool_timeouts"]) == (1, 0)
    assert (a["timeouts"], b["timeouts"]) == (1, 0)


def test_hosts_latency_and_safety_per_arm(tmp_path: Path) -> None:
    """CW2 "as the T-0160 smoke": hosts reached, step-latency p50 (nearest rank over ``generate``/``tool_step``
    requests) and the safety counters, per arm. A: hosts en.wikipedia.org, every step 100 ms, clean. B: hosts
    example.com, every step 300 ms; one trial voices a special token (voiced-uncommitted 1) and connects to
    10.0.0.5:443 (non-public 1)."""
    bad_connect = 'connect(6, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("10.0.0.5")}, 16) = -1'
    trials = []
    for a in GUIDANCE:
        for t in TASKS:
            for r in range(N):
                trials.append(
                    trial(
                        a,
                        t,
                        r,
                        attempts=[step1(), step(2, PLAIN)],
                        hosts=["en.wikipedia.org"] if a == "A" else ["example.com"],
                        step_ms=100 if a == "A" else 300,
                    )
                )
    trials[-1] = trial(
        "B",
        "W2",
        4,
        attempts=[step1(), step(2, PLAIN)],
        hosts=["example.com"],
        step_ms=300,
        voiced=["<|turn>model hi"],
        connects=[bad_connect],
    )
    report = analyse(write_run(tmp_path, trials))
    a, b = arm(report, "A"), arm(report, "B")
    assert a["hosts_seen"] == ["en.wikipedia.org"] and b["hosts_seen"] == ["example.com"], (a, b)
    assert a["step_ms_p50"] == 100 and b["step_ms_p50"] == 300, (a["step_ms_p50"], b["step_ms_p50"])
    assert a["safety"] == {"voiced_uncommitted": 0, "non_public_connects": 0}, a["safety"]
    assert b["safety"] == {"voiced_uncommitted": 1, "non_public_connects": 1}, b["safety"]


# --------------------------------------------------------------------------------------------- the comparisons


def test_comparisons_on_the_worked_run(tmp_path: Path) -> None:
    """CW2 report-only: M1_B = 1 <= M1_A / 2 = 2 -> True; B's successes 6 >= A's 4 -> True."""
    report = analyse(write_run(tmp_path, main_trials()))
    assert report["m1_b_at_most_half_a"] is True and report["b_successes_at_least_a"] is True, (
        report
    )


def comparison_run(root: Path, m1_a: int, m1_b: int, ok_a: int, ok_b: int) -> Path:
    """A run where arm X has ``m1_x`` step-2 doubt trials and ``ok_x`` successes (W1 first, then W2)."""
    trials = []
    for a, m1, ok in (("A", m1_a, ok_a), ("B", m1_b, ok_b)):
        for i, (t, r) in enumerate((t, r) for t in TASKS for r in range(N)):
            trials.append(
                trial(a, t, r, ok=i < ok, attempts=[step1(), step(2, DOUBT_1 if i < m1 else PLAIN)])
            )
    return write_run(root, trials)


@pytest.mark.parametrize(
    ("m1_a", "m1_b", "ok_a", "ok_b", "halved", "not_worse"),
    [
        (4, 2, 4, 4, True, True),  # both at their boundary: 2 = 4/2; 4 = 4
        (4, 3, 4, 3, False, False),  # one past each boundary
        (3, 2, 5, 6, False, True),  # 2 > 3/2 (a ceiling would say True)
        (5, 2, 6, 5, True, False),  # 2 <= 5/2 (a floor would still say True); 5 < 6
        (3, 1, 0, 0, True, True),  # 1 <= 1.5; 0 >= 0
        (1, 1, 2, 1, False, False),  # 1 > 0.5
    ],
)
def test_comparisons_at_boundaries(
    tmp_path: Path, m1_a: int, m1_b: int, ok_a: int, ok_b: int, halved: bool, not_worse: bool
) -> None:
    """CW2: "whether M1_B <= 1/2 M1_A" and "whether B's W1+W2 successes are >= A's", at and next to the boundary."""
    report = analyse(comparison_run(tmp_path, m1_a, m1_b, ok_a, ok_b))
    assert (arm(report, "A")["m1"], arm(report, "B")["m1"]) == (m1_a, m1_b)
    assert (arm(report, "A")["successes"], arm(report, "B")["successes"]) == (ok_a, ok_b)
    assert report["m1_b_at_most_half_a"] is halved, (m1_a, m1_b, report["m1_b_at_most_half_a"])
    assert report["b_successes_at_least_a"] is not_worse, (
        ok_a,
        ok_b,
        report["b_successes_at_least_a"],
    )


# --------------------------------------------------------------------------------------------- void (CW1)


def test_not_void_when_every_startup_line_is_right(tmp_path: Path) -> None:
    report = analyse(write_run(tmp_path, main_trials()))
    assert report["void"] == [], report["void"]


@pytest.mark.parametrize(
    ("who", "line"),
    [
        (("A", "W1", 3), None),  # no session.out at all
        (
            ("A", "W2", 1),
            "astra session: reask: announced_call_reask=on (default)\n",
        ),  # no guidance line
        (("B", "W1", 0), "astra session: live guidance: v1 (flag)\n"),  # arm B ran v1
        (("A", "W2", 4), "astra session: live guidance: v2 (flag)\n"),  # arm A ran v2
        (
            ("A", "W1", 0),
            "astra session: live guidance: v1 (profile)\n",
        ),  # v1, but not from the flag
        (
            ("B", "W2", 2),
            "astra session: live guidance: v2 (default)\n",
        ),  # v2, but not from the flag
    ],
    ids=["missing_out", "no_line", "b_says_v1", "a_says_v2", "a_profile", "b_default"],
)
def test_void_on_a_missing_or_wrong_guidance_line(
    tmp_path: Path, who: tuple[str, str, int], line: str | None
) -> None:
    """CW1: one trial whose startup output lacks its arm's ``live guidance: <v> (flag)`` makes the run void, and the
    finding names that trial (``<arm>-<task>-<repeat>``)."""
    trials = main_trials()
    i = next(i for i, t in enumerate(trials) if (t["arm"], t["task"], t["rep"]) == who)
    trials[i]["guidance_line"] = line
    report = analyse(write_run(tmp_path, trials))
    void = report["void"]
    name = "-".join(map(str, who))
    assert isinstance(void, list) and void, f"not void with {name}'s startup line {line!r}"
    assert any(name in str(v) for v in void), void
