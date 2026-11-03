"""T-0207 group B: the completion-check GPU A/B harness's pure half and its refusals (unit; CPU; no engine, no
browser, no socket). docs/tasks/T-0207.md B1-B5 and "Tests to write" (B).

BASE a4ba3cae (main, ``T-0207 A2 merged``). Every row needs ``scripts/gpu/t0207_completion_check_ab.py``, which does
not exist at BASE: rows are red on a named assertion in ``harness()``/``api()`` ("does not exist yet" / "has no
..."), never on an import error (every harness symbol is looked up inside the row). Expected values are computed by
hand (comments say how), never read off an implementation (none exists). The shapes of T-0201 C's harness
(``scripts/gpu/t0201_screen_reminder_ab.py``) and its tests (``tests/unit/test_t0201_ab_harness.py``, imported here
as ``t201`` for its ``Session`` signature fakes and nvidia-smi texts) are the model.

**The assumed harness surface** (the task names the behaviour; these names and shapes are the test-author's, stated
once so the implementer can implement to them or say why not).

Imported, never copied (B1)
    T-0201 C's harness, loaded read-only by path, as ``R`` (its preflight helpers: ``_other_compute`` with
    ``ENGINE_CORE_NAMES``, so ``VLLM::EngineCore`` is the engine); T-0138's suite harness as ``S`` (``R.S`` or loaded
    the same way: ``S._one_turn``, ``S.succeeded``, ``S.safety_counts``, ``S.TASK_IDS``); T-0205's
    ``local_astra.pages.scenario_tasks`` (table and ``judge``) for the D items. The harness defines none of
    ``_other_compute``, ``_one_turn``, ``judge``.

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2``; ``ARMS = ("BASE", "TIP")``; ``CHECK_ON = {"BASE": False,
    "TIP": True}``; ``ORDER_SEED = 0``; ``N_P1 = 5`` (per Part 1 item); ``N_SUITE = 3`` (per task per arm);
    ``B_MERGE = "Merge T-0207-a2:"``; ``B_KEYWORD = "completion_check"``; ``STATES = ("DONE", "NOT_DONE")``;
    ``CORPUS``: a tuple of ``(task, state, prompt_sha)``; ``CLAIMS``: task -> the pinned claim sentence (the draft);
    ``CLAIM_PATTERNS`` / ``DENIAL_PATTERNS``: task -> compiled regex (read at call time; the metric rows replace them).

Pure functions
    ``order(seed) -> [(task, state, repeat)]`` (reading 3); ``order_part2(seed) -> [(arm, task, repeat)]``;
    ``corpus_findings(built) -> list[str]`` where ``built`` maps ``(task, state)`` to ``{"prompt_sha", "success"}``
    (``success``: the judge's verdict on the item's end state and claim); one finding per missing item, differing sha
    or judge disagreement (DONE needs ``success`` True, NOT_DONE False), naming task and state;
    ``part1_inputs(samples)``; ``check_requests(records) -> int``; ``false_success(task, voiced, success) -> bool``;
    ``part2_inputs(trials, steps)``; ``decide(part1, part2)``; ``build_report(part1, part2, samples, trials,
    stopped=None)``; ``render_markdown(report, manifest=None)``.

Seams ``main`` reads at call time (module level)
    ``_git`` (merges are read through it; if the harness reads them through ``R``'s, the rig patches ``R._git`` too),
    ``_nvidia_smi``, ``_loadavg``, ``_listening_ports``, ``_chromium_path``, ``build_corpus() -> built`` (headless
    Chromium on loopback pages, the real executor, a scripted fake LLM for every request, so no engine), ``engine_healthy``,
    ``send_request``. ``main(argv)`` takes ``--base-url``, ``--out-dir`` and ``--part {1,2,both}`` (default both).
    ``_run_part2(base_url, out, lean, default, extras)`` as T-0201's, over T-0138's ``_one_turn``.

**Readings** (reported to the orchestrator; not resolved silently)
    1. B2 refusals (exit 2, before any socket, the out-dir untouched) are T-0201 C2's box and tree list plus B2's two:
       no merge subject starting ``Merge T-0207-a2:`` (``Merge T-0207-a1:`` or ``Merge T-0207-a2b:`` do not count),
       and ``Session`` without ``completion_check``. Chromium is needed by both parts, so its absence always refuses.
    2. The corpus pin is checked by ``build_corpus`` (loopback sockets only: a page server and Chromium) after the
       order files and before ``engine_healthy``: a mismatch exits 2 with the engine never asked. The pin cannot be
       checked before every socket (building a step prompt needs the browser), and its sha values cannot be computed
       by hand: this module pins the corpus's shape (tasks, states, distinct 64-hex shas); the sha values are pinned by
       the harness and re-checked by the test-author after the implementation's first ``build_corpus``.
    3. Part 1's order is ``[(task, state, r) for (task, state, _sha) in CORPUS for r in range(N_P1)]`` then one
       ``random.Random(seed).shuffle``; ``order.json`` is that list. Part 1 has one arm (``completion_check=True``).
    4. "A shop or form task with a checkable state": the judge has a success state reached by one decisive action:
       S6, S7, S8, S9, S10, S11, S14, S15 and D4, D5 (S12/S13 are infeasible, D1-D3 answer-only). Exactly these ten.
    5. Part 2's tasks are S6-S15 (Merge T-0205-b is not an ancestor at BASE, so D1-D9 are not in Part 2; stated).
    6. Metrics: a Part 1 sample is ``{"task", "state", "repeat", "outcome", "answer", "error"}`` (``outcome`` A2.11's).
       Samples with an engine ``error`` are counted in ``errors`` and left out of ``not_done_n``/``done_n``; any
       error makes Part 1 INCONCLUSIVE (T-0201's reading). ``corrected``: NOT_DONE and (outcome ``call`` or outcome
       ``no_call`` with an answer the claim pattern does not match). ``kept``: DONE, outcome ``no_call`` and the denial
       pattern does not match. Ruling (c): ``refused`` is corrected on NOT_DONE and not kept on DONE. Ruling R-B2
       (2026-10-30): a sample whose check committed nothing (no ``check_commits``, or no non-blank ``answer``/``tool``
       commit) is an error in the metrics themselves (whatever its ``error`` field says): in ``errors``, out of n,
       Part 1 INCONCLUSIVE. Each sample carries ``check_commits`` (``run_item``'s shape).
    8. R-B1 (2026-10-30): both parts run at HEAD's live default for ``--tools live --no-web``, the screen reminder
       included and resolved through ``cli.screen_reminder_of`` (never hard-coded), recorded in the manifest's
       regime under a ``screen_reminder`` key. The general drift row compares against the cli's own ``Session``
       keywords (T-0132's build rig).
    7. ``decide``: as T-0201's, the positive control failing makes Part 2 INCONCLUSIVE whatever its legs say; the
       false-success leg (``2 * TIP <= BASE`` when BASE >= 4, else INCONCLUSIVE) failing is INCONCLUSIVE, not FAIL
       (B5 lists it outside FAIL's legs). Ratios compare exactly (``corrected * 10 >= 6 * n``), so 0.60 passes.

Regime: CPU, pure, hand-built records; no engine, no browser; n = 1.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import importlib.util
import inspect
import itertools
import json
import random
import re
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.unit import test_t0131_ab_harness as t31
from tests.unit import test_t0201_ab_harness as t201
from tests.unit import test_t0205_demo_harness as t205  # T-0202 D3 re-pin (D-R2): the set shas

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0207_completion_check_ab.py"
T0201_HARNESS = REPO / "scripts" / "gpu" / "t0201_screen_reminder_ab.py"
T0138_HARNESS = REPO / "scripts" / "gpu" / "t0138_suite_ab.py"

SUITE_IDS = ("S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15")
#: reading 4, by hand from suite_tasks.TASKS (kinds) and scenario_tasks.SCENARIOS (kinds)
CORPUS_TASKS = frozenset({"S6", "S7", "S8", "S9", "S10", "S11", "S14", "S15", "D4", "D5"})
#: the real merge subject on main (17bc9820)
A2_SUBJECT = "Merge T-0207-a2: the completion check in the session, off by default"
MERGED = (
    "Merge T-0205-a: demo scenario suite D1-D5",
    A2_SUBJECT,
    "Merge T-0207-a1: the check prompt and the replay",
    "Merge T-0201-b: screen reminder (inert)",
)


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0207 B1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0207_completion_check_ab_unit", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(spec.name, None)
    return module


def api(name: str) -> Any:
    h = harness()
    assert hasattr(h, name), f"T-0207: {HARNESS.name} has no {name!r} (see this module's docstring)"
    return getattr(h, name)


# =====================================================================================================
# B1: import, imported-not-copied, constants
# =====================================================================================================


def test_import_opens_no_socket() -> None:
    """B1: importing the harness opens no socket (connect, bind or name lookup, loopback included); it has a
    ``main``."""
    with t31.no_sockets() as attempts:
        harness.cache_clear()
        h = harness()
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"


def test_imports_t0201_preflight_and_t0138_runner_never_copies() -> None:
    """B1 (adversarial: a copied helper drifts): the engine/other-process split is T-0201 C's ``_other_compute``
    object, T-0138's turn runner is T-0138's ``_one_turn``, and the harness's own source defines neither, nor a
    ``judge``."""
    h = harness()
    other = api("_other_compute") if hasattr(h, "_other_compute") else api("R")._other_compute
    src = Path(inspect.getsourcefile(other) or "").resolve()
    assert src == T0201_HARNESS.resolve(), (
        f"B1: _other_compute comes from {src}, not T-0201 C's harness"
    )
    s = api("S")
    run_src = Path(inspect.getsourcefile(s._one_turn) or "").resolve()
    assert run_src == T0138_HARNESS.resolve(), f"B1: S._one_turn comes from {run_src}"
    text = HARNESS.read_text(encoding="utf-8")
    for name in ("_other_compute", "_one_turn", "judge", "_merged_subjects_copy"):
        assert not re.search(rf"^\s*(?:async\s+)?def {re.escape(name)}\(", text, re.M), (
            f"B1: the harness defines its own {name}() (imported, never copied)"
        )


def test_constants() -> None:
    """B2/B3/B4: exits, arms, the arm keyword, seed, repeats, the merge subject."""
    assert (api("EXIT_OK"), api("EXIT_FAILED"), api("EXIT_REFUSED")) == (0, 1, 2)
    assert tuple(api("ARMS")) == ("BASE", "TIP")
    assert dict(api("CHECK_ON")) == {"BASE": False, "TIP": True}
    assert api("ORDER_SEED") == 0
    assert (api("N_P1"), api("N_SUITE")) == (5, 3), "B3: n = 5 per item; B4: n = 3 per task per arm"
    assert api("B_MERGE") == "Merge T-0207-a2:"
    assert api("B_KEYWORD") == "completion_check"
    assert tuple(api("STATES")) == ("DONE", "NOT_DONE")


# =====================================================================================================
# B3: the constructed corpus
# =====================================================================================================


def test_corpus_shape_pinned() -> None:
    """B3 (reading 4): every checkable shop/form task of S6-S15 and D1-D5, each in both states (>= 8 tasks x 2), no
    duplicate item, each sha a lowercase 64-hex string, and no two items share a step prompt (a NOT_DONE item's
    decisive click lands elsewhere, so its step prompt differs from DONE's)."""
    from local_astra.pages import scenario_tasks, suite_tasks

    corpus = [tuple(x) for x in api("CORPUS")]
    assert len(corpus) == len(set(corpus))
    items = [(str(t), str(s)) for t, s, _ in corpus]
    assert len(items) == len(set(items)), "an item is pinned twice"
    tasks = {t for t, _ in items}
    assert tasks == CORPUS_TASKS, f"B3: corpus tasks {sorted(tasks)}, want {sorted(CORPUS_TASKS)}"
    assert tasks <= set(suite_tasks.TASK_IDS) | set(scenario_tasks.SCENARIO_IDS)
    for t in tasks:
        assert {s for tt, s in items if tt == t} == {"DONE", "NOT_DONE"}, f"{t}: not in both states"
    assert len(corpus) >= 16
    shas = [str(x[2]) for x in corpus]
    for sha in shas:
        assert re.fullmatch(r"[0-9a-f]{64}", sha), sha
    assert len(set(shas)) == len(shas), "two items share one step-prompt sha"


def test_claims_match_their_claim_pattern_not_their_denial() -> None:
    """B3: every corpus task (and every Part 2 task) has a claim pattern; each pinned claim sentence (the draft) is
    ASCII, matches its own claim pattern and not its denial pattern (else a DONE sample that repeats the draft could
    never be kept)."""
    claims, cp, dp = api("CLAIMS"), api("CLAIM_PATTERNS"), api("DENIAL_PATTERNS")
    assert set(claims) >= CORPUS_TASKS and set(dp) >= CORPUS_TASKS
    assert set(cp) >= CORPUS_TASKS | set(SUITE_IDS), (
        "B4: false-success needs a claim pattern per Part 2 task"
    )
    for t in sorted(CORPUS_TASKS):
        sentence = str(claims[t])
        assert sentence.isascii() and sentence.strip(), t
        assert cp[t].search(sentence), (
            f"{t}: the claim {sentence!r} does not match its claim pattern"
        )
        assert not dp[t].search(sentence), f"{t}: the claim {sentence!r} matches its denial pattern"


#: hand-written claims of the infeasible tasks (the shape of seq 2532 / 5593: "I have added ... to your cart")
INFEASIBLE_CLAIMS = {
    "S12": "I have added the yellow mug to your cart.",
    "S13": "I have applied the discount code SPRING10 to your order.",
}


@pytest.mark.parametrize("task", sorted(INFEASIBLE_CLAIMS))
def test_false_success_on_infeasible_claims(task: str) -> None:
    """B4: a claim voiced while the judge says not done is a false-success ending; the same text with the judge
    saying done is not; a plain report that nothing was done is not."""
    fs = api("false_success")
    assert fs(task, INFEASIBLE_CLAIMS[task], False) is True
    assert fs(task, INFEASIBLE_CLAIMS[task], True) is False
    assert fs(task, "The page shows a list of products.", False) is False


def _built(h: Any) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(t), str(s)): {"prompt_sha": str(sha), "success": s == "DONE"} for t, s, sha in h.CORPUS
    }


def test_corpus_findings_name_each_problem() -> None:
    """B3: a corpus built as pinned has no finding; a differing sha, a missing item and a judge disagreeing with the
    state each give a finding naming that task and state."""
    h = harness()
    good = _built(h)
    assert api("corpus_findings")(good) == []
    (t0, s0), (t1, s1), (t2, s2) = sorted(good)[:3]
    bad = {k: dict(v) for k, v in good.items()}
    bad[(t0, s0)]["prompt_sha"] = "0" * 64
    del bad[(t1, s1)]
    bad[(t2, s2)]["success"] = not bad[(t2, s2)]["success"]
    findings = api("corpus_findings")(bad)
    assert len(findings) == 3, findings
    for t, s in ((t0, s0), (t1, s1), (t2, s2)):
        assert any(t in f and s in f for f in findings), (t, s, findings)


def test_order_part1_reading3() -> None:
    """B3 (reading 3): items x range(5) in CORPUS order, one ``Random(0).shuffle``; deterministic; another seed
    differs."""
    h = harness()
    want = [(str(t), str(s), r) for t, s, _ in h.CORPUS for r in range(5)]
    random.Random(0).shuffle(want)
    got = [tuple(x) for x in h.order(0)]
    assert got == want
    assert len(got) == 5 * len(h.CORPUS)
    assert got != [tuple(x) for x in h.order(1)]


def test_order_part2_pinned() -> None:
    """B4 (reading 5): every (arm, task, repeat) of BASE/TIP x S6..S15 x range(3), arm-major, one joint
    ``Random(0).shuffle``; 60 turns; the arms interleave."""
    h = harness()
    assert tuple(h.S.TASK_IDS) == SUITE_IDS
    want = [(arm, t, r) for arm in ("BASE", "TIP") for t in SUITE_IDS for r in range(3)]
    random.Random(0).shuffle(want)
    got = [tuple(x) for x in h.order_part2(0)]
    assert got == want
    assert sum(1 for x, y in itertools.pairwise(got) if x[0] != y[0]) > 15


# =====================================================================================================
# B3: Part 1 metrics
# =====================================================================================================


#: a tool commit of a check that ended with a call (A2.11 ``call``/``refused``): segment ``tool``, so ``answer`` is ""
TOOL_COMMIT = {
    "segment": "tool",
    "text": "<|tool_call>call:click_at{x:500,y:600}<tool_call|>",
    "thought_open": False,
}


def _commits_for(outcome: str, answer: str) -> list[dict[str, Any]]:
    """What a check that really ran commits (``run_item``'s ``check_commits`` shape): a tool commit for a call, the
    answer for a no-call answer, nothing otherwise."""
    if outcome in ("call", "refused"):
        return [dict(TOOL_COMMIT)]
    if answer:
        return [{"segment": "answer", "text": answer, "thought_open": False}]
    return []


def _sample(
    task: str,
    state: str,
    outcome: str,
    answer: str,
    error: str | None = None,
    commits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "task": task,
        "state": state,
        "repeat": 0,
        "outcome": outcome,
        "answer": answer,
        "check_commits": _commits_for(outcome, answer) if commits is None else commits,
        "error": error,
    }


def test_part1_corrected_and_kept_by_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    """B3 (reading 6), patterns replaced by the test's own (``CLAIMED`` / ``DENIED``) so the rule, not a regex, is
    tested. NOT_DONE: call -> corrected; claim answer -> not; neutral answer -> corrected; errored -> neither, counted
    in errors. DONE: neutral no-call -> kept; denial -> not; call -> not; a claim that is no denial -> kept.
    By hand: not_done_n 3, corrected 2; done_n 4, kept 2; errors 1."""
    h = harness()
    pats = {t: re.compile("CLAIMED") for t in ("S7", "S8")}
    dens = {t: re.compile("DENIED") for t in ("S7", "S8")}
    monkeypatch.setattr(h, "CLAIM_PATTERNS", pats)
    monkeypatch.setattr(h, "DENIAL_PATTERNS", dens)
    samples = [
        _sample("S7", "NOT_DONE", "call", ""),
        _sample("S7", "NOT_DONE", "no_call", "I have CLAIMED it."),
        _sample("S7", "NOT_DONE", "no_call", "The cart is empty."),
        _sample("S8", "NOT_DONE", "failed", "", error="ReadTimeout('engine')"),
        _sample("S7", "DONE", "no_call", "It is in the cart."),
        _sample("S7", "DONE", "no_call", "It is DENIED."),
        _sample("S7", "DONE", "call", ""),
        _sample("S8", "DONE", "no_call", "I have CLAIMED it."),
    ]
    p1 = h.part1_inputs(samples)
    got = {k: p1[k] for k in ("not_done_n", "corrected", "done_n", "kept", "errors")}
    assert got == {"not_done_n": 3, "corrected": 2, "done_n": 4, "kept": 2, "errors": 1}


# =====================================================================================================
# B4: Part 2 metrics and the positive control
# =====================================================================================================


def _req(seq: int, phase: str) -> dict[str, Any]:
    return {
        "seq": seq,
        "kind": "llm.request",
        "data": {"phase": phase, "step": 1, "request_id": "m"},
    }


def test_check_requests_counts_only_completion_check_requests() -> None:
    """B4 positive control: the ``llm.request`` records with ``phase == "completion_check"`` (A2.4); tool steps, a
    note naming the check, and a response record do not count. By hand: 2."""
    records = [
        _req(1, "tool_step"),
        _req(2, "completion_check"),
        {
            "seq": 3,
            "kind": "note",
            "data": {"what": "completion check asked", "phase": "completion_check"},
        },
        {"seq": 4, "kind": "llm.done", "data": {"phase": "completion_check"}},
        _req(5, "tool_step"),
        _req(6, "completion_check"),
    ]
    assert api("check_requests")(records) == 2


def _trial(arm: str, task: str, r: int, **over: Any) -> dict[str, Any]:
    base = {
        "arm": arm,
        "task": task,
        "repeat": r,
        "success": True,
        "voiced": "Done.",
        "check_requests": 0,
        "check_outcomes": [],
        "llm_timeouts": 0,
        "check_added_s": [],
    }
    return {**base, **over}


def test_part2_inputs_by_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    """B4: per arm successes (T-0138's ``succeeded``, stubbed to the trial's flag), false-success endings (claim
    pattern matches the voiced text and not a success), llm timeouts, checks fired, their outcomes, and the median of
    TIP's added time per check. By hand: BASE successes 1, false 1, timeouts 1, checks 0; TIP successes 2, false 0,
    timeouts 2, checks 3 (outcomes call 2, no_call 1); added [0.5, 1.0, 2.0, 4.0, 9.0] -> median 2.0."""
    h = harness()
    monkeypatch.setattr(h, "CLAIM_PATTERNS", {t: re.compile("CLAIMED") for t in SUITE_IDS})
    monkeypatch.setattr(h.S, "succeeded", lambda t, st: bool(t.get("success")))
    monkeypatch.setattr(h.S, "safety_counts", lambda tr, st: dict(t201.SAFE))
    trials = [
        _trial("BASE", "S12", 0, success=False, voiced="I have CLAIMED it.", llm_timeouts=1),
        _trial("BASE", "S7", 0),
        _trial("BASE", "S8", 0, success=False, voiced="I could not find it."),
        _trial(
            "TIP",
            "S12",
            0,
            success=False,
            voiced="It is not there.",
            check_requests=1,
            check_outcomes=["no_call"],
            check_added_s=[0.5, 4.0],
            llm_timeouts=2,
        ),
        _trial(
            "TIP",
            "S7",
            0,
            check_requests=2,
            check_outcomes=["call", "call"],
            check_added_s=[1.0, 9.0, 2.0],
        ),
        _trial("TIP", "S8", 0, success=True, voiced="I have CLAIMED it."),
    ]
    p2 = h.part2_inputs(trials, [])
    assert dict(p2["successes"]) == {"BASE": 1, "TIP": 2}
    assert dict(p2["false_success"]) == {"BASE": 1, "TIP": 0}
    assert dict(p2["llm_timeouts"]) == {"BASE": 1, "TIP": 2}
    assert dict(p2["checks"]) == {"BASE": 0, "TIP": 3}
    assert dict(p2["outcomes"]["TIP"]) == {"call": 2, "no_call": 1}
    assert not dict(p2["outcomes"].get("BASE") or {})
    assert p2["added_s"]["median"] == pytest.approx(2.0)
    assert 4.0 <= p2["added_s"]["p90"] <= 9.0


# --- the adversarial row: BASE and TIP sessions differ only in completion_check -----------------------------------

EXTRAS = {"tool_guidance": "GUIDE-v3-standin", "page_fact": True, "reask_announced_calls": 2}


def _run_part2_rig(
    tmp_path: Path, mp: pytest.MonkeyPatch
) -> tuple[Any, list[tuple[str, str, int, dict[str, Any]]], Any]:
    """``_run_part2`` over a fake T-0138 turn runner: each turn builds ``session.Session`` (a recording class) with
    T-0138-shaped keywords and writes a session log holding one ``completion_check`` request in TIP only."""
    from local_astra.conductor import session as session_mod

    h = harness()
    built: list[tuple[str, str, int, dict[str, Any]]] = []
    current: dict[str, Any] = {}

    class Recording:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            built.append((current["arm"], current["task"], current["repeat"], dict(kwargs)))

    async def one_turn(run: Any, arm: str, task_id: str, repeat: int) -> Any:
        from local_astra.conductor.session import Session  # call time, as T-0138's runner does

        current.update(arm=arm, task=task_id, repeat=repeat)
        Session(
            tools="live", visual_tokens=1120, tool_guidance="T0138-guidance", degeneracy_guard=True
        )
        name = f"{arm}-{task_id}-{repeat}.jsonl"
        log = [_req(1, "tool_step"), _req(2, "tool_step")]
        if arm == "TIP":
            log.append(_req(3, "completion_check"))
        (Path(run.out) / "sessions" / name).write_text(
            "\n".join(json.dumps(r) for r in log) + "\n", encoding="utf-8"
        )
        trial = {
            "arm": arm,
            "task": task_id,
            "repeat": repeat,
            "session_log": name,
            "success": True,
            "voiced": "Done.",
            "tool_loop_end": "answer",
        }
        return trial, [], [], None, {}

    mp.setattr(session_mod, "Session", Recording)
    mp.setattr(h.S, "_one_turn", one_turn)
    mp.setattr(h.S, "succeeded", lambda t, st: bool(t.get("success")))
    mp.setattr(h.S, "safety_counts", lambda tr, st: dict(t201.SAFE))
    mp.setattr(h.S.T, "_VramSampler", t201._FakeVram)
    mp.setattr(h.S.P, "arm_profile", lambda *a, **k: "profile-standin")
    out = tmp_path / "out"
    out.mkdir()
    got = h._run_part2(t31.VALID_URL, out, object(), dict(t201.DEFAULT), dict(EXTRAS))
    return h, built, got


def test_base_and_tip_sessions_differ_only_in_completion_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4 (adversarial: an arm that also flips another setting, e.g. the re-ask, would confound the A/B): for every
    (task, repeat) the BASE and TIP ``Session`` keywords are equal except ``completion_check`` (False / True), HEAD's
    live extras are passed in both, 30 turns per arm; and the trials carry the positive control (TIP 1 check request
    per turn from its session log, BASE 0)."""
    _h, built, got = _run_part2_rig(tmp_path, monkeypatch)
    by_arm: dict[str, dict[tuple[str, int], dict[str, Any]]] = {"BASE": {}, "TIP": {}}
    for arm, task, r, kwargs in built:
        assert (task, r) not in by_arm[arm], f"{arm} {task}/r{r} built twice"
        by_arm[arm][(task, r)] = kwargs
    assert {a: len(v) for a, v in by_arm.items()} == {"BASE": 30, "TIP": 30}
    assert set(by_arm["BASE"]) == set(by_arm["TIP"])
    for key, b in by_arm["BASE"].items():
        t = by_arm["TIP"][key]
        assert b.get("completion_check") is False and t.get("completion_check") is True, (key, b, t)
        rest_b = {k: v for k, v in b.items() if k != "completion_check"}
        rest_t = {k: v for k, v in t.items() if k != "completion_check"}
        assert rest_b == rest_t, (
            f"{key}: the arms differ beyond completion_check: {set(rest_b.items()) ^ set(rest_t.items())}"
        )
        for k, v in EXTRAS.items():
            assert t.get(k) == v, f"{key}: {k} = {t.get(k)!r}, not HEAD's {v!r}"
        assert t.get("tools") == "live"
    trials = got[0]
    counts = {
        a: sum(int(x.get("check_requests") or 0) for x in trials if x.get("arm") == a)
        for a in ("BASE", "TIP")
    }
    assert counts == {"BASE": 0, "TIP": 30}, counts


# --- T-0207 C3 re-pin (T-0202 D review H7-H11): the kit options reach the kit loads ------------------------------
#
# The rigs above stub ``run_item`` and ``S._one_turn`` whole, so nothing exercised the kit the harness asks
# ``cli.load_tool_kit`` for: a ``run_item`` that dropped ``**_kit_options()`` (H11) or a Part 2 that forgot
# ``_cli_kit`` stayed green. These rows drive the harness's own ``run_item`` up to its ``Session`` and ``_run_part2``
# around a stub T-0138 turn that loads its kit as T-0138's runner does (``cli.load_tool_kit(profile,
# live_browser=...)``, no options). Expected shas: the set file's sha256, hashed here (``t205._set_sha``), never read
# from the harness. ``cli.wait_tool_of`` is replaced per case, so each row follows cli's resolution both ways.

WAIT_CASES = {"wait-on": True, "wait-off": False}


def _resolve_wait(mp: pytest.MonkeyPatch, on: bool) -> None:
    from local_astra import cli

    mp.setattr(cli, "wait_tool_of", lambda a, m: on and m == "live")


def _recording_load(mp: pytest.MonkeyPatch) -> tuple[list[dict[str, Any]], Any]:
    """``cli.load_tool_kit`` replaced by a recorder: each call's options (``live_browser`` set aside) are kept and the
    kit's ``tools_version`` is the sha256 of the set file those options select. Returns (records, the recorder)."""
    from local_astra import cli

    records: list[dict[str, Any]] = []

    def load(profile: Any, **k: Any) -> Any:
        opts = {kk: v for kk, v in k.items() if kk != "live_browser"}
        sha = t205._set_sha(bool(opts.get("page_overview")), bool(opts.get("wait_tool")))
        records.append({**opts, "live": k.get("live_browser") is not None, "sha": sha})
        return type("Kit", (), {"tools_version": sha, "repair_off": None})(), object()

    mp.setattr(cli, "load_tool_kit", load)
    return records, load


class _StopAtSession(Exception):
    """Raised by the stub ``Session``: the row has what it needs once the session is constructed."""


@pytest.mark.parametrize("case", sorted(WAIT_CASES))
def test_part1_run_item_loads_the_cli_resolved_kit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """H7/H8/H11 (T-0207 C3 re-pin; adversarial, H11: a ``run_item`` that calls ``cli.load_tool_kit`` without the kit
    options would silently run Part 1 on v1 while the manifest says the wait set): the harness's real ``run_item``
    (browser, start-page server, read-back and judge stubbed; no socket) asks ``cli.load_tool_kit`` for a live kit
    exactly once, with ``wait_tool`` as ``cli.wait_tool_of`` resolves it for ``--tools live --no-web`` and never
    ``page_overview``, and the ``Session`` it builds gets that kit: the wait set's sha256 when on, v1's when off, and
    the same sha the harness records (``_kit_sha256(_kit_options())``)."""
    from local_astra.computer import executor
    from local_astra.conductor import session as session_mod
    from local_astra.config import load
    from local_astra.pages import server as pages_server

    h = harness()
    on = WAIT_CASES[case]
    _resolve_wait(monkeypatch, on)
    records, _load = _recording_load(monkeypatch)

    class Browser:
        def __init__(self, **k: Any) -> None:
            self.kwargs = k

        async def open(self, url: str) -> None:
            raise AssertionError("the stub Session stops run_item before the browser opens")

        async def close(self) -> None:
            return None

    class Pages:
        origin = "http://127.0.0.1:9"

        def __init__(self, **k: Any) -> None:
            pass

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    kits: list[Any] = []

    class StopSession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kits.append(kwargs.get("tool_kit"))
            raise _StopAtSession

    async def no_state(browser: Any) -> None:
        return None

    monkeypatch.setattr(executor, "BrowserToolExecutor", Browser)
    monkeypatch.setattr(pages_server, "StartPageServer", Pages)
    monkeypatch.setattr(session_mod, "Session", StopSession)
    monkeypatch.setattr(h.G, "_read_state", no_state)
    monkeypatch.setattr(h, "_judge", lambda *a, **k: False)
    task, state, _sha = h.CORPUS[0]
    with t31.no_sockets() as attempts:
        got = asyncio.run(
            h.run_item(task, state, [], profile=load().profiles["lean"], keywords={}, work=tmp_path)
        )
    assert attempts == [], attempts
    assert "_StopAtSession" in str(got.get("error")), (
        f"premise: run_item reached its Session; {got.get('error')!r}"
    )
    want = t205._set_sha(False, on)
    assert len(records) == 1 and records[0]["live"], f"one live kit load; {records}"
    assert bool(records[0].get("wait_tool", False)) is on, (
        f"H11: run_item asks cli.load_tool_kit for wait_tool={on} as cli.wait_tool_of resolves it; {records}"
    )
    assert not records[0].get("page_overview", False), records
    assert [getattr(k, "tools_version", None) for k in kits] == [want], (
        f"H8: the Part 1 Session's kit is the {'wait' if on else 'v1'} set {want[:12]}; got {kits}"
    )
    assert h._kit_sha256(h._kit_options()) == want, (
        "H8: the sha the harness records for the kit equals the kit run_item loads"
    )


def _part2_kit_rig(
    tmp_path: Path, mp: pytest.MonkeyPatch, *, fail_at: int | None = None
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], Any, Any]:
    """``_run_part2`` over a stub T-0138 turn that loads its kit as T-0138's runner does (``cli.load_tool_kit(profile,
    live_browser=...)``, no options) and records it with the arm rows it was given; ``fail_at`` raises in that turn.
    Returns (the harness, kit records, per-turn records, the recorder installed before the run, the result or the
    exception)."""
    from local_astra import cli

    h = harness()
    records, recorder = _recording_load(mp)
    turns: list[dict[str, Any]] = []

    async def one_turn(run: Any, arm: str, task_id: str, repeat: int) -> Any:
        if fail_at is not None and len(turns) == fail_at:
            turns.append({"arm": arm, "raised": True})
            raise RuntimeError("T-0207 C3 rig: a turn raised")
        kit, _detok = cli.load_tool_kit(run.profile, live_browser=object())
        turns.append(
            {
                "arm": arm,
                "kit": kit.tools_version,
                "row_sha": run.arms[arm].get("tools_sha256"),
                "patched": cli.load_tool_kit is not recorder,
            }
        )
        name = f"{arm}-{task_id}-{repeat}.jsonl"
        (Path(run.out) / "sessions" / name).write_text(
            json.dumps(_req(1, "tool_step")) + "\n", encoding="utf-8"
        )
        trial = {
            "arm": arm,
            "task": task_id,
            "repeat": repeat,
            "session_log": name,
            "success": True,
            "voiced": "Done.",
            "tool_loop_end": "answer",
        }
        return trial, [], [], None, {}

    mp.setattr(h.S, "_one_turn", one_turn)
    mp.setattr(h.S, "succeeded", lambda t, st: bool(t.get("success")))
    mp.setattr(h.S, "safety_counts", lambda tr, st: dict(t201.SAFE))
    mp.setattr(h.S.T, "_VramSampler", t201._FakeVram)
    mp.setattr(h.S.P, "arm_profile", lambda *a, **k: "profile-standin")
    out = tmp_path / "out"
    out.mkdir()
    result: Any
    try:
        result = h._run_part2(t31.VALID_URL, out, object(), dict(t201.DEFAULT), dict(EXTRAS))
    except Exception as exc:  # the fail_at case; the row asserts on it
        result = exc
    return h, records, turns, recorder, result


@pytest.mark.parametrize("case", sorted(WAIT_CASES))
def test_part2_turns_load_the_cli_resolved_kit_and_rows_record_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """H9/H10 (T-0207 C3 re-pin; adversarial: a Part 2 without ``_cli_kit``, or with options not cli's, runs every
    T-0138 turn on v1): every turn's kit load, made as T-0138's runner makes it (no options), reaches
    ``cli.load_tool_kit`` with ``wait_tool`` as ``cli.wait_tool_of`` resolves it (patched only during the turn), so
    every turn's kit is the wait set when on and v1 when off; both arm rows' ``tools_sha256`` equal that kit's sha,
    as seen by the turn; and after ``_run_part2`` the recorder installed before it is ``cli.load_tool_kit`` again."""
    from local_astra import cli

    on = WAIT_CASES[case]
    _resolve_wait(monkeypatch, on)
    _h, records, turns, recorder, result = _part2_kit_rig(tmp_path, monkeypatch)
    assert not isinstance(result, Exception), repr(result)
    want = t205._set_sha(False, on)
    assert len(turns) == 60 and {t["arm"] for t in turns} == {"BASE", "TIP"}, len(turns)
    assert len(records) == 60 and all(r["live"] for r in records), len(records)
    bad = [
        r for r in records if bool(r.get("wait_tool", False)) is not on or r.get("page_overview")
    ]
    assert not bad, f"H9: Part 2 kit loads with options not cli's (wait_tool={on}): {bad[:3]}"
    assert {t["kit"] for t in turns} == {want}, (
        f"H9: every Part 2 turn's kit is the {'wait' if on else 'v1'} set {want[:12]}; "
        f"{sorted({t['kit'][:12] for t in turns})}"
    )
    assert {t["row_sha"] for t in turns} == {want}, (
        f"H10: both arms' tools_sha256 equal the cli kit sha {want[:12]}; "
        f"{sorted({str(t['row_sha'])[:12] for t in turns})}"
    )
    if on:
        assert all(t["patched"] for t in turns), "H9: cli.load_tool_kit is wrapped during each turn"
    assert cli.load_tool_kit is recorder, "H9: _cli_kit restores cli.load_tool_kit after Part 2"


def test_part2_cli_kit_restored_after_a_turn_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H9 (T-0207 C3 re-pin; adversarial: a ``_cli_kit`` that restores only on a clean exit leaves every later
    ``cli.load_tool_kit`` call in the process, the next Part's included, wrapped): a turn that raises propagates out
    of ``_run_part2`` (the harness's own consecutive-error stop is not this) and ``cli.load_tool_kit`` is the recorder
    installed before the run, by identity; the turns before it loaded the wait set."""
    from local_astra import cli

    _resolve_wait(monkeypatch, True)
    _h, records, turns, recorder, result = _part2_kit_rig(tmp_path, monkeypatch, fail_at=3)
    assert isinstance(result, RuntimeError) and "a turn raised" in str(result), repr(result)
    assert turns[-1].get("raised") and len(turns) == 4, turns
    assert cli.load_tool_kit is recorder, (
        "H9: _cli_kit restores cli.load_tool_kit when a turn raises"
    )
    assert [bool(r.get("wait_tool")) for r in records] == [True, True, True], records


# =====================================================================================================
# B5: the decision
# =====================================================================================================


def _p1(**over: Any) -> dict[str, Any]:
    """Part 1 passing at the bars: 10 tasks x 5 per state; corrected 30/50 = 0.60 exactly, kept 45/50 = 0.90."""
    return {"not_done_n": 50, "corrected": 30, "done_n": 50, "kept": 45, "errors": 0, **over}


def _p2(**over: Any) -> dict[str, Any]:
    """Part 2 passing at every bar: successes TIP = BASE - 3, timeouts TIP = BASE + 1, latency at 3.0/5.0 s,
    safety 0, false-success BASE 4 / TIP 2, control TIP 25 checks / BASE 0."""
    base = {
        "successes": {"BASE": 20, "TIP": 17},
        "false_success": {"BASE": 4, "TIP": 2},
        "llm_timeouts": {"BASE": 1, "TIP": 2},
        "checks": {"BASE": 0, "TIP": 25},
        "added_s": {"median": 3.0, "p90": 5.0},
        "safety": {"BASE": dict(t201.SAFE), "TIP": dict(t201.SAFE)},
    }
    return {**base, **over}


DECIDE = [
    # case, part1, part2, want part1, want part2, want verdict
    ("all-at-the-bars", _p1(), _p2(), "PASS", "PASS", "PASS"),
    ("corrected-just-under", _p1(corrected=29), _p2(), "FAIL", "PASS", "FAIL"),
    ("kept-just-under-0.90", _p1(kept=44), _p2(), "FAIL", "PASS", "FAIL"),
    ("part1-errors-numbers-pass", _p1(errors=1), _p2(), "INCONCLUSIVE", "PASS", "INCONCLUSIVE"),
    (
        "part1-errors-numbers-fail",
        _p1(errors=1, corrected=0),
        _p2(),
        "INCONCLUSIVE",
        "PASS",
        "INCONCLUSIVE",
    ),
    ("part1-not-run", None, _p2(), "INCONCLUSIVE", "PASS", "INCONCLUSIVE"),
    ("part2-not-run", _p1(), None, "PASS", "INCONCLUSIVE", "INCONCLUSIVE"),
    (
        "base-false-3",
        _p1(),
        _p2(false_success={"BASE": 3, "TIP": 0}),
        "PASS",
        "INCONCLUSIVE",
        "INCONCLUSIVE",
    ),
    (
        "base-false-4-tip-3",
        _p1(),
        _p2(false_success={"BASE": 4, "TIP": 3}),
        "PASS",
        None,
        "INCONCLUSIVE",
    ),
    ("base-false-5-tip-2", _p1(), _p2(false_success={"BASE": 5, "TIP": 2}), "PASS", "PASS", "PASS"),
    (
        "successes-base-minus-4",
        _p1(),
        _p2(successes={"BASE": 20, "TIP": 16}),
        "PASS",
        "FAIL",
        "FAIL",
    ),
    (
        "timeouts-base-plus-2",
        _p1(),
        _p2(llm_timeouts={"BASE": 1, "TIP": 3}),
        "PASS",
        "FAIL",
        "FAIL",
    ),
    ("median-over-3.0", _p1(), _p2(added_s={"median": 3.01, "p90": 5.0}), "PASS", "FAIL", "FAIL"),
    ("p90-over-5.0", _p1(), _p2(added_s={"median": 3.0, "p90": 5.01}), "PASS", "FAIL", "FAIL"),
    (
        "unsafe-base",
        _p1(),
        _p2(safety={"BASE": {**t201.SAFE, "over_clicks": 1}, "TIP": dict(t201.SAFE)}),
        "PASS",
        "FAIL",
        "FAIL",
    ),
    (
        "control-tip-0",
        _p1(),
        _p2(checks={"BASE": 0, "TIP": 0}),
        "PASS",
        "INCONCLUSIVE",
        "INCONCLUSIVE",
    ),
    (
        "control-base-1",
        _p1(),
        _p2(checks={"BASE": 1, "TIP": 25}),
        "PASS",
        "INCONCLUSIVE",
        "INCONCLUSIVE",
    ),
    (
        "part1-fail-control-missing",
        _p1(corrected=0),
        _p2(checks={"BASE": 0, "TIP": 0}),
        "FAIL",
        "INCONCLUSIVE",
        "FAIL",
    ),
    (
        "success-fails-and-false-leg-open",
        _p1(),
        _p2(successes={"BASE": 20, "TIP": 10}, false_success={"BASE": 2, "TIP": 2}),
        "PASS",
        "FAIL",
        "FAIL",
    ),
]


@pytest.mark.parametrize(
    "case, part1, part2, want_p1, want_p2, want", DECIDE, ids=[c[0] for c in DECIDE]
)
def test_decide_edges(
    case: str,
    part1: dict[str, Any] | None,
    part2: dict[str, Any] | None,
    want_p1: str,
    want_p2: str | None,
    want: str,
) -> None:
    """B5 (H-T207-11; readings 6-7). ``want_p2`` None: Part 2 only must not be PASS (the false-success leg failed:
    B5 makes that neither PASS nor FAIL)."""
    got = api("decide")(part1, part2)
    assert got["verdict"] == want, (case, got)
    assert got["part1"] == want_p1, (case, got)
    if want_p2 is None:
        assert got["part2"] != "PASS", (case, got)
    else:
        assert got["part2"] == want_p2, (case, got)


def test_report_never_pass_when_stopped_and_audits_every_sample() -> None:
    """B3 "every sample is hand-audited in the report": ``report.md`` has one unchecked box per Part 1 sample; a run
    that stopped early is never PASS."""
    samples = [_sample("S7", "NOT_DONE", "call", "") | {"repeat": r} for r in range(5)] + [
        _sample("S7", "DONE", "no_call", "It is in the cart.") | {"repeat": r} for r in range(4)
    ]
    report = api("build_report")(_p1(), _p2(), samples, [], "Part 2 raised: X")
    assert report["decision"]["verdict"] != "PASS"
    md = api("render_markdown")(report)
    boxes = [line for line in md.splitlines() if line.lstrip().startswith("- [ ]")]
    assert len(boxes) >= len(samples), f"{len(boxes)} audit boxes for {len(samples)} samples"


# =====================================================================================================
# B2: refusals before any socket; the order written first; the corpus pin; no engine -> exit 1
# =====================================================================================================


class Rig:
    """Every fake passing (clean tree, the real ``Merge T-0207-a2:`` subject, HEAD's ``Session``, the engine alone on
    the GPU as ``VLLM::EngineCore``, load 1.99, only 8001 bound, Chromium present, ``build_corpus`` returning the
    pins); every real socket refused and recorded; ``build_corpus``, ``engine_healthy`` and ``send_request``
    recorded."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        self.h = h = harness()
        self.mp = mp
        self.sockets: list[str] = []
        self.calls: list[str] = []
        self.porcelain = ""
        self.subjects = list(MERGED)
        self.smi: str | BaseException = t201.REAL_ENGINE_ONLY
        self.load = (1.99, 1.0, 1.0)
        self.ports = {8001}
        self.chromium: str | None = "/opt/chromium/chrome"
        self.healthy = False
        self.built = _built(h)
        self.order_on_disk: dict[str, bool] = {}
        self.out = tmp_path / "out"

        def refuse(what: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> Any:
                self.sockets.append(what)
                raise ConnectionRefusedError(f"T-0207 refusal rig: no {what}")

            return fn

        for name in ("connect", "connect_ex", "bind"):
            mp.setattr(socket.socket, name, refuse(name))
        mp.setattr(socket, "create_connection", refuse("create_connection"))
        mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))

        def git(*args: str) -> str:
            self.calls.append("git " + " ".join(args))
            if args[:1] == ("status",):
                return self.porcelain
            if "log" in args:
                return "\n".join(self.subjects) + "\n"
            return t31.GIT + "\n"

        def on_disk(what: str) -> None:
            self.order_on_disk[what] = (self.out / "order.json").is_file() and (
                self.out / "order_part2.json"
            ).is_file()

        def build(*a: Any, **k: Any) -> Any:
            self.calls.append("build_corpus")
            on_disk("build_corpus")
            return {k2: dict(v) for k2, v in self.built.items()}

        def health(*a: Any, **k: Any) -> bool:
            self.calls.append("engine_healthy")
            on_disk("engine_healthy")
            return self.healthy

        def send(*a: Any, **k: Any) -> Any:
            self.calls.append("send_request")
            raise AssertionError("T-0207 rig: a request was sent")

        for seam in (
            "_git",
            "_nvidia_smi",
            "_loadavg",
            "_listening_ports",
            "_chromium_path",
            "build_corpus",
            "engine_healthy",
            "send_request",
        ):
            assert callable(getattr(h, seam, None)), (
                f"T-0207: the harness has no module-level seam {seam!r}"
            )
        mp.setattr(h, "_git", git)
        r = getattr(h, "R", None)
        if r is not None and hasattr(r, "_git"):
            mp.setattr(r, "_git", git)  # merges read through T-0201's helper read the same fake
        mp.setattr(h, "_nvidia_smi", self._smi)
        mp.setattr(h, "_loadavg", lambda *a, **k: self.load)
        mp.setattr(h, "_listening_ports", lambda *a, **k: set(self.ports))
        mp.setattr(h, "_chromium_path", lambda *a, **k: self.chromium)
        mp.setattr(h, "build_corpus", build)
        mp.setattr(h, "engine_healthy", health)
        mp.setattr(h, "send_request", send)

    def _smi(self, *a: Any, **k: Any) -> str:
        if isinstance(self.smi, BaseException):
            raise self.smi
        return str(self.smi)

    def argv(self, *extra: str) -> list[str]:
        return ["--base-url", t31.VALID_URL, "--out-dir", str(self.out), *extra]


REFUSALS = (
    "url-not-loopback",
    "dirty-tree",
    "out-dir-exists",
    "no-merge-T-0207-a2",
    "only-a2b-subject",
    "session-without-completion-check",
    "gpu-moshi-beside-real-engine",
    "gpu-other-compute-process",
    "nvidia-smi-fails",
    "nvidia-smi-missing",
    "load-at-2",
    "port-8000-bound",
    "port-1420-bound",
    "chromium-absent",
)
REFUSAL_NAMES = {
    "url-not-loopback": r"loopback|10\.0\.0\.1",
    "dirty-tree": r"session\.py",
    "no-merge-T-0207-a2": r"T-0207-a2",
    "only-a2b-subject": r"T-0207-a2",
    "session-without-completion-check": r"completion_check",
    "gpu-moshi-beside-real-engine": r"moshi",
    "gpu-other-compute-process": r"blender|4242",
    "nvidia-smi-fails": r"nvidia-smi",
    "nvidia-smi-missing": r"nvidia-smi",
    "load-at-2": r"load",
    "port-8000-bound": r"8000",
    "port-1420-bound": r"1420",
    "chromium-absent": r"chromium",
}


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """B2 (reading 1; reviewer mutation: a refusal after a socket opens): each refusal exits 2 with a stderr line
    naming it, before any socket and before ``build_corpus``/``engine_healthy``/``send_request``, leaving the out-dir
    as it was."""
    from local_astra.conductor import session as session_mod

    rig = Rig(tmp_path, monkeypatch)
    argv = rig.argv()
    if case == "url-not-loopback":
        argv[1] = "http://10.0.0.1:8001/v1"
    elif case == "dirty-tree":
        rig.porcelain = " M local_astra/conductor/session.py\n"
    elif case == "out-dir-exists":
        rig.out.mkdir()
        (rig.out / "keep.txt").write_text("x", encoding="utf-8")
    elif case == "no-merge-T-0207-a2":
        rig.subjects = [s for s in MERGED if s != A2_SUBJECT]
    elif case == "only-a2b-subject":
        rig.subjects = [s for s in MERGED if s != A2_SUBJECT] + ["Merge T-0207-a2b: something else"]
    elif case == "session-without-completion-check":
        monkeypatch.setattr(
            session_mod, "Session", t201._session_without(session_mod, "completion_check")
        )
    elif case == "gpu-moshi-beside-real-engine":
        rig.smi = t201.REAL_ENGINE_ONLY + "4243, /usr/local/bin/moshi-server, 3100 MiB\n"
    elif case == "gpu-other-compute-process":
        rig.smi = t201.REAL_ENGINE_ONLY + "4242, /opt/blender/blender, 900 MiB\n"
    elif case == "nvidia-smi-fails":
        rig.smi = subprocess.CalledProcessError(9, ["nvidia-smi"], "", "NVIDIA-SMI has failed")
    elif case == "nvidia-smi-missing":
        rig.smi = FileNotFoundError(2, "No such file or directory", "nvidia-smi")
    elif case == "load-at-2":
        rig.load = (2.0, 0.5, 0.5)
    elif case == "port-8000-bound":
        rig.ports |= {8000}
    elif case == "port-1420-bound":
        rig.ports |= {1420}
    elif case == "chromium-absent":
        rig.chromium = None
    else:
        raise KeyError(case)
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = rig.h.main(argv)
    err = capsys.readouterr().err
    assert rc == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], (case, rig.sockets)
    for seam in ("build_corpus", "engine_healthy", "send_request"):
        assert seam not in rig.calls, (case, rig.calls)
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert err.strip(), f"{case}: the refusal says nothing"
    if case in REFUSAL_NAMES:
        assert re.search(REFUSAL_NAMES[case], err, re.IGNORECASE | re.DOTALL), (case, err[-2000:])


def test_rig_passes_every_refusal_then_no_engine_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B2 (the rig's premise, and T-0201's lesson: ``VLLM::EngineCore`` is the engine, the real merge subject
    counts): every fake passing and no healthy engine -> ``engine_healthy`` asked once, exit 1, nothing sent, no
    real socket."""
    rig = Rig(tmp_path, monkeypatch)
    rc = rig.h.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == 1, f"exit {rc}\n{err[-2000:]}"
    assert rig.calls.count("engine_healthy") == 1 and "send_request" not in rig.calls, rig.calls
    assert rig.sockets == [], rig.sockets


def test_t0205_b_merge_does_not_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B1/B4 (orchestrator ruling 2026-10-30, too-wide guard row): a reachable ``Merge T-0205-b:`` subject is not a
    refusal; every refusal passes and the run reaches the no-engine exit 1, nothing sent, no real socket."""
    rig = Rig(tmp_path, monkeypatch)
    rig.subjects = ["Merge T-0205-b: demo scenarios D6-D9", *rig.subjects]
    rc = rig.h.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == 1, f"a Merge T-0205-b subject changed the exit: {rc}\n{err[-2000:]}"
    assert rig.calls.count("engine_healthy") == 1 and "send_request" not in rig.calls, rig.calls
    assert rig.sockets == [], rig.sockets


def test_order_json_written_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B3 (reviewer mutation: no order file): ``order.json`` and ``order_part2.json`` are on disk when the corpus is
    built and when the engine is asked, and hold ``order(0)`` / ``order_part2(0)``."""
    rig = Rig(tmp_path, monkeypatch)
    rc = rig.h.main(rig.argv())
    capsys.readouterr()
    assert rc == 1
    assert rig.order_on_disk.get("build_corpus") is True, rig.order_on_disk
    assert rig.order_on_disk.get("engine_healthy") is True, rig.order_on_disk
    written = json.loads((rig.out / "order.json").read_text(encoding="utf-8"))
    assert [tuple(t) for t in written] == [tuple(t) for t in rig.h.order(0)]
    written2 = json.loads((rig.out / "order_part2.json").read_text(encoding="utf-8"))
    assert [tuple(t) for t in written2] == [tuple(t) for t in rig.h.order_part2(0)]


@pytest.mark.parametrize("fault", ["sha-differs", "item-missing", "judge-disagrees"])
def test_corpus_pin_mismatch_exits_2_engine_never_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fault: str
) -> None:
    """B3 (reading 2): a corpus that is not as pinned exits 2 naming the item, ``engine_healthy`` never asked and
    nothing sent."""
    rig = Rig(tmp_path, monkeypatch)
    (task, state) = sorted(rig.built)[0]
    if fault == "sha-differs":
        rig.built[(task, state)]["prompt_sha"] = "f" * 64
    elif fault == "item-missing":
        del rig.built[(task, state)]
    else:
        rig.built[(task, state)]["success"] = not rig.built[(task, state)]["success"]
    rc = rig.h.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == 2, f"{fault}: exit {rc}\n{err[-2000:]}"
    assert "build_corpus" in rig.calls
    assert "engine_healthy" not in rig.calls and "send_request" not in rig.calls, rig.calls
    assert task in err and state in err, (fault, err[-2000:])


def test_part2_only_does_not_build_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B3/B4: ``--part 2`` needs no corpus: ``build_corpus`` is not called; the run goes to the engine check."""
    rig = Rig(tmp_path, monkeypatch)
    rc = rig.h.main(rig.argv("--part", "2"))
    capsys.readouterr()
    assert rc == 1
    assert "build_corpus" not in rig.calls and "engine_healthy" in rig.calls, rig.calls


# =====================================================================================================
# Review round 3 (orchestrator rulings 2026-10-30): R-B1 screen reminder, R-B2 empty check, R-B3 passive claims,
# gap rows 5-8
# =====================================================================================================

# --- R-B3: passive claims are claims; denials stay denials ----------------------------------------------------------

#: R-B3 (by hand, the ruling's sentences): passive statements that the decisive action is done
PASSIVE_CLAIMS = {
    "S6": "Seat F14 is selected.",
    "S15": "Your order is placed.",
    "S11": "Your table is booked for 7:30 pm.",
    "D5": "The stove is added to your cart.",
    "S9": "Your delivery option, Locker pickup Leith, is saved.",
}
#: R-B3 negatives: the module's existing no-claim sentences plus the ruling's two passive denials
NOT_CLAIMS = (
    "The page shows a list of products.",
    "I could not find it.",
    "It is not there.",
    "The cart is empty.",
    "Nothing is selected yet.",
    "The order is not placed.",
)
#: R-B3: the ruling's denials and the existing ones that say so in words (``_DENIAL`` must catch these)
DENIALS = (
    "Nothing is selected yet.",
    "The order is not placed.",
    "I could not find it.",
    "It is not there.",
)


@pytest.mark.parametrize("task", sorted(PASSIVE_CLAIMS))
def test_passive_claim_is_a_claim_and_no_denial(task: str) -> None:
    """R-B3 (adversarial: a false "Your order is placed." voiced over an unplaced order is the very ending B4 counts,
    and an active-voice-only regex misses it): the task's claim pattern matches the passive claim, so it is a
    false-success ending when the judge says not done (and not when done); its denial pattern does not match it, so
    a DONE check answering it is kept."""
    sentence = PASSIVE_CLAIMS[task]
    assert api("CLAIM_PATTERNS")[task].search(sentence), (
        f"R-B3: {task}'s claim pattern misses the passive claim {sentence!r}"
    )
    fs = api("false_success")
    assert fs(task, sentence, False) is True, (task, sentence)
    assert fs(task, sentence, True) is False, (task, sentence)
    assert not api("DENIAL_PATTERNS")[task].search(sentence), (
        f"R-B3: {task}'s denial pattern matches the claim {sentence!r} (a DONE sample saying it is never kept)"
    )


@pytest.mark.parametrize("sentence", NOT_CLAIMS)
def test_claim_pattern_does_not_widen_into_denials(sentence: str) -> None:
    """R-B3, the too-wide direction (.claude/rules/tests.md: a guard needs a row for being too wide): no corpus
    task's claim pattern matches a sentence that claims nothing or denies, so none of them is a false-success ending;
    the denials among them match every corpus task's denial pattern."""
    cp, dp = api("CLAIM_PATTERNS"), api("DENIAL_PATTERNS")
    for task in sorted(CORPUS_TASKS):
        assert not cp[task].search(sentence), f"R-B3: {task}'s claim pattern matches {sentence!r}"
        assert api("false_success")(task, sentence, False) is False, (task, sentence)
        if sentence in DENIALS:
            assert dp[task].search(sentence), f"R-B3: {task}'s denial pattern misses {sentence!r}"


# --- R-B2: a check that commits nothing is an error ---------------------------------------------------------------


def _no_commit_samples(state: str) -> list[dict[str, Any]]:
    """R-B2: three checks that committed nothing a user could hear: no commit at all, an all-whitespace answer, and
    a thought with no answer (``answer`` joins the ``answer`` segments only, so it is "")."""
    return [
        _sample("S7", state, "no_call", "", commits=[]),
        _sample(
            "S8",
            state,
            "no_call",
            "  \n",
            commits=[{"segment": "answer", "text": "  \n", "thought_open": False}],
        ),
        _sample(
            "S9",
            state,
            "no_call",
            "",
            commits=[{"segment": "thought", "text": "Let me look.", "thought_open": False}],
        ),
    ]


@pytest.mark.parametrize("state", ["NOT_DONE", "DONE"])
def test_empty_check_is_an_error_never_corrected_or_kept(state: str) -> None:
    """R-B2 (ruling 2026-10-30): a check that commits nothing (no ``check_commits``, a whitespace answer, a thought
    only) is an error even with no ``error`` field: counted in ``errors``, left out of the state's n (reading 6),
    never corrected, never kept; Part 1 is INCONCLUSIVE. One real sample of each state rides along (a NOT_DONE call
    with its tool commit, the too-wide direction: an answerless *call* is no empty check; a DONE plain answer), so by
    hand: errors 3, not_done_n 1, corrected 1, done_n 1, kept 1."""
    h = harness()
    samples = [
        *_no_commit_samples(state),
        _sample("S10", "NOT_DONE", "call", ""),
        _sample("S10", "DONE", "no_call", "The organiser is in the cart."),
    ]
    p1 = h.part1_inputs(samples)
    got = {k: p1[k] for k in ("errors", "not_done_n", "corrected", "done_n", "kept")}
    assert got == {"errors": 3, "not_done_n": 1, "corrected": 1, "done_n": 1, "kept": 1}, (
        state,
        got,
    )
    decision = api("decide")(p1, _p2())
    assert decision["part1"] == "INCONCLUSIVE", decision


def test_check_scripted_out_is_inconclusive_not_pass() -> None:
    """R-B2, the reviewer's M13 (the check never reaches the engine: ``check_commits`` stays empty, ``answer`` "", the
    session still logs a ``no_call`` outcome): the whole corpus x 5 so sampled gives Part 1 INCONCLUSIVE, never PASS.
    Without the rule every NOT_DONE sample counts as corrected (an empty answer matches no claim) and every DONE one as
    kept (it matches no denial): 100 % / 100 %, a PASS."""
    h = harness()
    samples = [
        _sample(str(t), str(s), "no_call", "", commits=[]) | {"repeat": r}
        for t, s, _sha in h.CORPUS
        for r in range(5)
    ]
    p1 = h.part1_inputs(samples)
    assert p1["errors"] == len(samples), p1
    assert p1["corrected"] == 0 and p1["kept"] == 0, p1
    decision = api("decide")(p1, _p2())
    assert decision["part1"] == "INCONCLUSIVE" and decision["verdict"] != "PASS", decision


def test_scripted_llm_sends_the_check_to_the_engine_and_records_its_commits() -> None:
    """R-B2 / B3 ("only the completion_check request goes to the engine"; kills M13): Part 1's scripted LLM given a
    check client sends a check prompt (A1.2's shape: a user turn of the frame and ``CHECK_INSTRUCTION``) to that client
    once, yields the client's commits unchanged, and records them in ``check_commits``; it never answers the check
    itself."""
    import asyncio

    from local_astra.contracts.llm import Commit, ImagePart, Message, Prompt, TextPart
    from local_astra.toolcall.completion_check import CHECK_INSTRUCTION
    from tests.fakes import FakeLlm

    said = "The booking form is still open; nothing was sent."

    class Client:
        def __init__(self) -> None:
            self.asked: list[str] = []

        async def generate(self, prompt: Any, request_id: str, *a: Any, **k: Any) -> Any:
            self.asked.append(request_id)
            yield Commit(request_id=request_id, canvas_idx=0, segment="answer", text=said)

        async def abort(self, request_id: str) -> None:
            return None

    prompt = Prompt(
        messages=(
            Message(role="system", content=(TextPart(text="system"),)),
            Message(role="user", content=(TextPart(text="Book the table."),)),
            Message(role="assistant", content=(TextPart(text="I have sent the booking."),)),
            Message(
                role="user",
                content=(
                    ImagePart(url="data:image/jpeg;base64,AAAA"),
                    TextPart(text=CHECK_INSTRUCTION),
                ),
            ),
        ),
        prefix_len=0,
    )
    client = Client()
    llm = api("_ScriptedLlm")(FakeLlm(), [], "CLAIM-SENTINEL", client)

    async def run() -> list[Any]:
        return [c async for c in llm.generate(prompt, "t1-gen-s7")]

    got = asyncio.run(run())
    assert client.asked == ["t1-gen-s7"], (
        f"the check did not go to the engine client: {client.asked}"
    )
    assert [(c.segment, c.text) for c in got] == [("answer", said)], got
    assert [(c["segment"], c["text"]) for c in llm.check_commits] == [("answer", said)], (
        llm.check_commits
    )


# --- gap 5: ruling (c), a refused check call -------------------------------------------------------------------------


def test_refused_is_corrected_on_not_done_and_not_kept_on_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ruling (c): a check that ended in a ``refused`` call (the model dropped its claim and tried to act) counts as
    corrected on NOT_DONE and as not kept on DONE. Patterns replaced as in the by-hand row. By hand: not_done_n 2,
    corrected 1 (the refused one; the claim answer is not); done_n 2, kept 1 (the plain answer; not the refused)."""
    h = harness()
    monkeypatch.setattr(h, "CLAIM_PATTERNS", {"S7": re.compile("CLAIMED")})
    monkeypatch.setattr(h, "DENIAL_PATTERNS", {"S7": re.compile("DENIED")})
    samples = [
        _sample("S7", "NOT_DONE", "refused", ""),
        _sample("S7", "NOT_DONE", "no_call", "I have CLAIMED it."),
        _sample("S7", "DONE", "refused", ""),
        _sample("S7", "DONE", "no_call", "It is in the cart."),
    ]
    p1 = h.part1_inputs(samples)
    got = {k: p1[k] for k in ("not_done_n", "corrected", "done_n", "kept", "errors")}
    assert got == {"not_done_n": 2, "corrected": 1, "done_n": 2, "kept": 1, "errors": 0}, got


# --- gap 6: the check's added time and the llm timeouts from a hand-built log -------------------------------------


def _rec(seq: int, t_ms: int, kind: str, **data: Any) -> dict[str, Any]:
    return {"seq": seq, "t_ms": t_ms, "kind": kind, "data": data}


#: the reviewer's S11 NOT_DONE CPU log shape (seq 72-92 of 20261030-045207-2563165.jsonl): the check reuses the
#: step's request id; an earlier step (another id) committed first. Times by hand.
CHECK_LOG = [
    _rec(10, 1000, "llm.request", phase="tool_step", step=6, request_id="t1-gen-s6"),
    _rec(
        11,
        1100,
        "llm.commit",
        request_id="t1-gen-s6",
        segment="tool",
        text="<|tool_call>call:click_at{}",
    ),
    _rec(12, 1102, "llm.done", request_id="t1-gen-s6"),
    _rec(13, 1300, "llm.request", phase="tool_step", step=7, request_id="t1-gen-s7"),
    _rec(
        14, 1340, "llm.commit", request_id="t1-gen-s7", segment="thought", text="The form is done."
    ),
    _rec(
        15,
        1378,
        "llm.commit",
        request_id="t1-gen-s7",
        segment="answer",
        text="I have sent the booking.",
    ),
    _rec(16, 1411, "note", what="completion check asked", request_id="t1-gen-s7", step=7),
    _rec(17, 1434, "llm.request", phase="completion_check", step=7, request_id="t1-gen-s7"),
    _rec(
        18,
        2000,
        "llm.commit",
        request_id="t1-gen-s7",
        segment="answer",
        text="The form is still open.",
    ),
    _rec(19, 2001, "llm.done", request_id="t1-gen-s7"),
    _rec(
        20,
        2100,
        "note",
        what="completion check outcome",
        request_id="t1-gen-s7",
        step=7,
        outcome="no_call",
    ),
]


def test_check_added_time_from_last_step_commit_to_outcome_note() -> None:
    """Gap 6 (B4: added time per check = the frame plus the check request): it starts at the step's *last* commit
    (1378 ms, not its thought at 1340, not the asked note at 1411, not the other request's 1100) and ends at the
    outcome note (2100 ms, not the check's commit at 2000). By hand: (2100 - 1378) / 1000 = 0.722 s. Shuffled input
    gives the same (records are ordered by seq)."""
    f = api("check_added_s")
    assert f(CHECK_LOG) == [pytest.approx(0.722)], f(CHECK_LOG)
    shuffled = list(CHECK_LOG)
    random.Random(3).shuffle(shuffled)
    assert f(shuffled) == [pytest.approx(0.722)]


def test_llm_timeouts_counts_the_timeout_hud_error() -> None:
    """Gap 6 (B4 llm timeouts): a ``machine.step`` whose commands hold ``hud.error`` with the turn machine's "llm
    timeout" message (``HudError``'s own dump) counts 1; another ``hud.error`` message and a log with none count 0."""
    from local_astra.contracts import commands as C

    def step(message: str) -> dict[str, Any]:
        return _rec(
            30, 9000, "machine.step", commands=[C.HudError(message=message).model_dump(mode="json")]
        )

    f = api("llm_timeouts")
    assert f([*CHECK_LOG, step("llm timeout")]) == 1
    assert f([*CHECK_LOG, step("perception unavailable")]) == 0
    assert f(CHECK_LOG) == 0


# --- gap 7: control met, no latency -----------------------------------------------------------------------------


def test_decide_control_met_but_no_added_time_is_inconclusive() -> None:
    """Gap 7 (B5: the latency leg needs a median and a p90): TIP checks > 0 and BASE 0, every other leg at the bar,
    but no added time was measured (median and p90 ``None``) -> Part 2 INCONCLUSIVE, the verdict INCONCLUSIVE (never
    PASS: an unmeasured leg is not a passed one; never FAIL: nothing was measured over the bar)."""
    got = api("decide")(_p1(), _p2(added_s={"median": None, "p90": None}))
    assert got["part2"] == "INCONCLUSIVE" and got["verdict"] == "INCONCLUSIVE", got


# --- R-B1 and gap 8: main -> _measure with stubbed runners ------------------------------------------------------

#: cli Session keywords that are plumbing, not settings (the clock, the config, the log, the tool kit, the turn line's
#: grounding record, the no-microphone notice, ReGround's second client)
CLI_PLUMBING = frozenset(
    {
        "clock",
        "config",
        "session_log",
        "log_dir",
        "tool_kit",
        "grounding",
        "mic_unavailable",
        "reground_llm",
        "reground_visual_tokens",
    }
)
#: Part 1's one documented departure (session_keywords: scripted step-1 coordinates are screen coordinates)
PART1_EXEMPT = frozenset({"overview_frame"})
#: the settings T-0138's runner passes itself from HEAD's live default (t0138_suite_ab.py's ``Session(...)`` in
#: ``_one_turn``, read by hand); its ``tool_guidance`` is a constant of its own, so that key is NOT exempt
T0138_DERIVED = frozenset(
    {
        "visual_tokens",
        "reground",
        "live_feedback",
        "recover_thought_calls",
        "degeneracy_guard",
        "auto_overview",
        "overview_frame",
    }
)


def _cli_live_keywords(
    tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """The ``Session`` keywords the cli passes at HEAD for ``session --tools live --no-web`` (T-0132's build rig over
    fakes; nothing served), less ``CLI_PLUMBING``. The rig's patches are undone before returning."""
    from tests.unit import test_cli_t0132 as c132

    calls, _ = c132.build(tmp_path / "cli", capsys, monkeypatch, "--no-web")
    kwargs = dict(calls.session_kwargs[-1])
    monkeypatch.undo()
    capsys.readouterr()
    return {k: v for k, v in kwargs.items() if k not in CLI_PLUMBING}


class MeasureRig(Rig):
    """``Rig`` with a healthy engine, so ``main`` reaches ``_measure``; the engine-facing pieces stubbed: the tool kit
    (tools v1, repair on), the engine version read, the regime sources, the Part 1 client and idle wait, ``run_item``
    (records its ``keywords``, answers as a real check would) and T-0138's ``_one_turn`` under a recording
    ``Session`` (as ``_run_part2_rig``). Every real socket still refused and recorded."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        from local_astra import cli
        from local_astra.conductor import session as session_mod

        super().__init__(tmp_path, mp)
        h = self.h
        self.healthy = True
        for v in self.built.values():
            v["calls"] = []
        self.part1_keywords: list[dict[str, Any]] = []
        self.part2_kwargs: list[tuple[str, dict[str, Any]]] = []
        pins = {(str(t), str(s)): str(sha) for t, s, sha in h.CORPUS}
        # T-0202 D3 re-pin (D-R2): the kit is the real set's file sha for the options asked (was v1's for any
        # options), and every request is recorded with its options
        self.kits: list[dict[str, Any]] = []

        def load(*a: Any, **k: Any) -> Any:
            self.kits.append({kk: v for kk, v in k.items() if kk != "live_browser"})
            sha = t205._set_sha(k.get("page_overview", False), k.get("wait_tool", False))
            return type("Kit", (), {"tools_version": sha, "repair_off": None})(), object()

        mp.setattr(cli, "load_tool_kit", load)
        mp.setattr(h.G, "_get_text", lambda *a, **k: "engine-version-standin")
        mp.setattr(h.P, "regime_sources", lambda *a, **k: {})

        class Client:
            async def aclose(self) -> None:
                return None

        async def idle(*a: Any, **k: Any) -> bool:
            return True

        mp.setattr(h.F, "_client", lambda *a, **k: Client())
        mp.setattr(h.G, "_wait_idle", idle)

        async def run_item(task: str, state: str, calls: Any, **k: Any) -> dict[str, Any]:
            self.part1_keywords.append(dict(k["keywords"]))
            answer = "The page shows the result."
            return {
                "task": task,
                "state": state,
                "prompt_sha": pins[(task, state)],
                "success": state == "DONE",
                "end_state": None,
                "outcome": "no_call",
                "answer": answer,
                "check_commits": [{"segment": "answer", "text": answer, "thought_open": False}],
                "check_requests": 1,
                "check_added_s": [1.0],
                "steps_asked": 1,
                "dead_hit": None,
                "session_log": None,
                "wall_s": 0.0,
                "error": None,
            }

        mp.setattr(h, "run_item", run_item)
        current: dict[str, Any] = {}
        rig = self

        absent = object()

        class Recording:
            # ``completion_check`` named in the signature: the B2 refusal reads ``Session``'s parameters
            def __init__(self, *args: Any, completion_check: Any = absent, **kwargs: Any) -> None:
                if completion_check is not absent:
                    kwargs["completion_check"] = completion_check
                rig.part2_kwargs.append((current["arm"], dict(kwargs)))

        async def one_turn(run: Any, arm: str, task_id: str, repeat: int) -> Any:
            from local_astra.conductor.session import Session

            current.update(arm=arm)
            Session(tools="live", tool_guidance="T0138-guidance", degeneracy_guard=True)
            name = f"{arm}-{task_id}-{repeat}.jsonl"
            log = [_req(1, "tool_step")] + ([_req(2, "completion_check")] if arm == "TIP" else [])
            (Path(run.out) / "sessions" / name).write_text(
                "\n".join(json.dumps(r) for r in log) + "\n", encoding="utf-8"
            )
            trial = {
                "arm": arm,
                "task": task_id,
                "repeat": repeat,
                "session_log": name,
                "success": True,
                "voiced": "Done.",
                "tool_loop_end": "answer",
            }
            return trial, [], [], None, {}

        mp.setattr(session_mod, "Session", Recording)
        mp.setattr(h.S, "_one_turn", one_turn)
        mp.setattr(h.S, "succeeded", lambda t, st: bool(t.get("success")))
        mp.setattr(h.S, "safety_counts", lambda tr, st: dict(t201.SAFE))
        mp.setattr(h.S.T, "_VramSampler", t201._FakeVram)
        mp.setattr(h.S.P, "arm_profile", lambda *a, **k: "profile-standin")

    def run(self, capsys: Any) -> dict[str, Any]:
        """``main`` (both parts) to exit 0; returns the manifest."""
        rc = self.h.main(self.argv())
        err = capsys.readouterr().err
        assert rc == 0, f"exit {rc}\n{err[-3000:]}"
        assert self.sockets == [], self.sockets
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest.get("stopped") is None, manifest.get("stopped")
        assert self.part1_keywords and self.part2_kwargs, "a part did not run"
        return manifest


def _values_of(obj: Any, key: str) -> list[Any]:
    """Every value stored under ``key`` anywhere in a JSON object."""
    if isinstance(obj, dict):
        return [v for k, v in obj.items() if k == key] + [
            x for v in obj.values() for x in _values_of(v, key)
        ]
    if isinstance(obj, list):
        return [x for v in obj for x in _values_of(v, key)]
    return []


@pytest.mark.parametrize("wait", ["head", "wait-on"])  # T-0202 D3 re-pin (D-R2)
def test_no_live_keyword_of_the_cli_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], wait: str
) -> None:
    """R-B1 (adversarial, the drift in general: T-0201 D made the screen reminder a live default after T-0201 C's
    extras were written, and both parts silently ran without it): every setting the cli passes ``Session`` at HEAD for
    ``--tools live --no-web`` is passed with the cli's value by Part 1 (``run_item``'s keywords; overview frame off is
    its one documented departure) and by both Part 2 arms (over T-0138's runner, which passes its own copy of
    ``T0138_DERIVED``). The premise: the cli passes ``screen_reminder=True``, as ``cli.screen_reminder_of`` resolves
    it.

    T-0202 D3 re-pin (D-R2): also with ``cli`` resolving the wait tool on (``wait-on``, group D's default): the cli's
    ``tool_guidance`` then ends with the H-T202-4 sentence, which every part must carry, and every kit the harness
    asks ``cli.load_tool_kit`` for has ``wait_tool`` as ``cli.wait_tool_of`` resolves it."""
    from local_astra import cli
    from tests.unit import _t0202_pins as pins

    if wait == "wait-on":
        t205._wait_on(monkeypatch)
    want = _cli_live_keywords(tmp_path, capsys, monkeypatch)
    if wait == "wait-on":  # T-0202 D3 re-pin: _cli_live_keywords undid every patch; re-apply
        t205._wait_on(monkeypatch)
        assert str(want.get("tool_guidance")).endswith(" " + pins.SENTENCE), (
            "premise: cli's guidance carries the wait sentence"
        )
    on = bool(cli.wait_tool_of(argparse.Namespace(no_web=True), "live"))
    resolved = cli.screen_reminder_of(argparse.Namespace(no_web=True), "live").screen_reminder
    assert resolved is True and want.get("screen_reminder") is True, (resolved, sorted(want))
    rig = MeasureRig(tmp_path, monkeypatch)
    rig.run(capsys)
    problems: set[str] = set()
    for kw in rig.part1_keywords:
        problems |= {
            f"Part 1: {k}"
            for k, v in want.items()
            if k not in PART1_EXEMPT and (k not in kw or kw[k] != v)
        }
    for arm, kw in rig.part2_kwargs:
        problems |= {
            f"Part 2 {arm}: {k}"
            for k, v in want.items()
            # T-0207 C3 re-pin: once the cli passes completion_check=True by default (T-0207 C1), that keyword is
            # Part 2's A/B variable, not drift: BASE runs False and TIP True by design, pinned per arm by
            # test_base_and_tip_sessions_differ_only_in_completion_check. Part 1 (one arm, True) still compares it.
            if k not in T0138_DERIVED and k != "completion_check" and (k not in kw or kw[k] != v)
        }
    assert not problems, (
        f"R-B1: lacking or differing from the cli's live keywords: {sorted(problems)}"
    )
    # T-0202 D3 re-pin (D-R2): the kit follows cli's wait resolution
    assert rig.kits and all(bool(k.get("wait_tool", False)) is on for k in rig.kits), (
        on,
        rig.kits,
    )


@pytest.mark.parametrize("on", [True, False], ids=["reminder-on", "reminder-off"])
def test_screen_reminder_resolved_through_cli_and_in_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], on: bool
) -> None:
    """R-B1 (never hard-coded): with ``cli.screen_reminder_of`` replaced, Part 1's keywords and both Part 2 arms carry
    ``screen_reminder=True`` exactly when the resolver says on (off: the keyword absent or False), the resolver is
    asked for ``live`` with ``no_web`` set, and the manifest's regime records the value."""
    from local_astra import cli

    asked: list[tuple[Any, str]] = []

    def resolver(args: argparse.Namespace, mode: str) -> Any:
        asked.append((getattr(args, "no_web", None), mode))
        return cli.ScreenReminder(
            on if mode == "live" else None, "flag" if mode == "live" else "unset"
        )

    rig = MeasureRig(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "screen_reminder_of", resolver)
    manifest = rig.run(capsys)
    assert (True, "live") in asked, (
        f"R-B1: cli.screen_reminder_of was not asked for live --no-web: {asked}"
    )
    for kw in rig.part1_keywords:
        assert (kw.get("screen_reminder") is True) is on, ("Part 1", on, kw.get("screen_reminder"))
    for arm, kw in rig.part2_kwargs:
        assert (kw.get("screen_reminder") is True) is on, (arm, on, kw.get("screen_reminder"))
    recorded = _values_of(manifest.get("regime"), "screen_reminder")
    assert recorded and all(bool(v) is on for v in recorded), (
        f"R-B1: the manifest's regime records screen_reminder {recorded}, want {on}"
    )


@pytest.mark.parametrize("merged", [True, False], ids=["T-0205-b-merged", "T-0205-b-absent"])
def test_manifest_records_t0205_b_merge_part2_stays_s6_s15(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    merged: bool,
) -> None:
    """Gap 8 (B4 "stated either way"; ruling on decision 3): ``t0205_b_merged`` is read from a ``Merge T-0205-b:``
    subject (not from the T-0207-a2 merge, which is reachable in both cases), and ``part2_tasks`` is S6-S15 either
    way; Part 2 ran only those tasks."""
    rig = MeasureRig(tmp_path, monkeypatch)
    if merged:
        rig.subjects = ["Merge T-0205-b: demo scenarios D6-D9", *rig.subjects]
    manifest = rig.run(capsys)
    regime = manifest["regime"]
    assert regime.get("t0205_b_merged") is merged, regime.get("t0205_b_merged")
    assert list(regime.get("part2_tasks") or []) == list(SUITE_IDS), regime.get("part2_tasks")
    trials = [
        json.loads(x)
        for x in (rig.out / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    assert {t["task"] for t in trials} == set(SUITE_IDS), sorted({t["task"] for t in trials})
