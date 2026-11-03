"""T-0115: the GPU probe of the multi-step tool loop -- the harness's pure parts (unit; CPU; no engine).

Rows that need the harness are red on the named assertion in ``harness()`` / ``api()``
(the file or a name is absent) or on an assertion of their own, never on an import error: the harness is loaded
lazily, by path, as ``tests/unit/test_t0093_report.py`` loads T-0093's. Bash may not name the GPU directory, so it is
never run from a shell here; ``main(argv)`` is called in-process with every connect refused.

**The assumed harness surface** (``scripts/gpu/t0115_loop_probe.py``). The task names the behaviour; these names and
shapes are the test-author's, stated here once so the implementer can implement to them (or say why not):

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2``; ``TOOLS_SHA12 = "bbfc53ccdd80"``; ``MAX_TOOL_STEPS = 8``;
    ``MAX_PROMPT_TOKENS = 14_336``; ``CASES_SHA256`` (the pin of ``tests/eval/toolcall/cases.jsonl``, T-0087's
    ``460cd1c3...``); ``SCENARIO_SHA256`` (the harness's copy of the scenario-table pin, equal to this module's
    ``SCENARIO_TABLE_SHA256``); ``METRIC_QUERIES = "vllm:prefix_cache_queries_total"`` and ``METRIC_HITS =
    "vllm:prefix_cache_hits_total"`` (as T-0093); ``OUTCOMES = ("final", "call", "refused", "repairable-refused",
    "thought-open", "empty")``.

The scenario table (criterion 4)
    ``ScenarioPrompt``, a frozen dataclass: ``prompt_id: str``, ``text: str`` (the typed turn), ``screen: str`` (a
    path relative to ``tests/eval/toolcall``, the turn's frame), ``case_id: str | None`` (the dev case the prompt or
    screen comes from), ``rows: tuple[SearchResults, ...] = ()`` (``ScenarioSearch``'s row sets) and ``fact: str |
    None = None`` (the invented fact a correct S1/S2 report contains). ``Scenario``, a frozen dataclass:
    ``scenario_id: str``, ``n: int`` (repeats per prompt), ``prompts: tuple[ScenarioPrompt, ...]``.
    ``SCENARIOS: tuple[Scenario, ...]``, S0..S6 in that order, read by ``main`` at call time (a module global).
    ``schedule() -> list[tuple[str, str, int]]``: ``(scenario_id, prompt_id, repeat)`` in run order, 118 entries.
    ``scenario_table_sha(scenarios=None) -> str`` (``None`` means ``SCENARIOS``): the sha256 hex of
    ``json.dumps([{"scenario_id", "n", "prompts": [{"prompt_id", "text", "screen", "case_id", "fact",
    "rows": [r.model_dump(mode="json") for r in rows]}]}], sort_keys=True, separators=(",", ":"),
    ensure_ascii=False).encode("utf-8")`` -- ``canonical_sha`` below is that formula.

``ScenarioSearch(rows: Sequence[SearchResults])`` (criterion 3)
    Satisfies ``SearchBackend``; ``async search(query) -> SearchResults`` whose ``results`` are those of
    ``rows[min(i, len(rows) - 1)]`` for the i-th call (0-based) on that instance, whatever the query; ``queries:
    list[str]`` holds every query in call order (the harness writes them to ``queries.jsonl``). One instance per
    turn. Opens no socket.

``repeat_counts(calls) -> Mapping[str, int]``
    ``calls`` are the executed calls of one turn in order, ``{"step": int, "name": str, "arguments": dict}``. Keys:
    ``exact`` (calls whose ``(name, arguments)`` equals that of ANY earlier call of the turn, dict equality),
    ``normalised_query`` (``google_search`` calls whose query, casefolded with whitespace runs collapsed and the ends
    stripped, equals an earlier search's), ``click_near`` (``click_at`` calls within 16 of an earlier click on each
    axis, in the call's own 0-999 units).

``classify_step(*, answer: str, thought_open: bool, call: str | None) -> str``
    One of ``OUTCOMES``. ``call`` is ``None`` (no call), ``"valid"``, ``"refused"`` or ``"repairable"`` (refused at
    step >= 2 as "repair: step 1 only"). A call decides first (``call``, ``refused``, ``repairable-refused``); else
    a canvas ending inside an open thought is ``thought-open`` whatever text it carries; else non-blank ``answer`` is
    ``final``; else ``empty``.

``LoopMeter(inner, *, metrics, clock)``
    The session's ``LlmBackend`` wrapper (T-0093's ``_MeteredLlm`` reshaped): ``metrics()`` returns ``/metrics``
    counters keyed by ``METRIC_QUERIES``/``METRIC_HITS`` (sync; the harness passes ``read_metrics``), ``clock()`` is
    monotonic seconds. ``generate(prompt, request_id, options=None, *, xargs=None, cd=None)`` reads ``metrics``
    before the request and after it is done, and appends to ``generates: list[dict]`` one row per generate:
    ``request_id``, ``prefix_query_tokens`` and ``prefix_hit_tokens`` (that generate's own delta), ``first_commit_ms``
    and ``done_ms`` (from the moment the request is sent, after the before-read, to the first commit and to the end).

The rule (criteria 5-7), on the records of ``trials.jsonl`` and ``steps.jsonl`` (lists of dicts, joined on
``(scenario, prompt_id, repeat)``; only the keys below are read, so prompt ids are not checked against the table)
    A trial: ``scenario``, ``prompt_id``, ``repeat``, ``request_ids``, ``tool_steps``, ``tool_steps_executed``,
    ``tool_loop_end`` (``None`` when absent), ``executed`` (the executed calls as for ``repeat_counts``),
    ``repeats`` (``repeat_counts`` of them), ``played_runs`` (the text of each playback run, in order),
    ``history`` (the assistant entry as stored, or ``None``), ``thought_commits`` (every thought commit text of the
    turn), ``report`` (the final voiced report), ``fact_hit``, ``over_budget_step`` (the step whose request was not
    sent, from the session's note) and ``over_budget_chars``, ``exception`` (``None`` or a repr), ``timed_out``,
    ``silent_gap_ms``.
    A step (one per generate): ``scenario``, ``prompt_id``, ``repeat``, ``step``, ``request_id``,
    ``continuation_chars`` (``None`` at step 1), ``prefix_query_tokens`` (the engine's prompt tokens), ``prefix_hit_tokens``,
    ``tokenize`` (ids, or ``"unsupported"``), ``first_commit_ms``, ``done_ms``, ``outcome`` (``classify_step``),
    ``call_name``, ``step_text`` (the step's committed, thought-stripped answer text), ``opened_with_thought``,
    ``executed`` (this step's call was executed), ``voiced`` (the ``TtsSpeak`` texts sourced at this step),
    ``voiced_sources`` (the same commands' ``TtsSpeak.source`` texts, raw committed text, in order; A2') and
    ``http_status``.
    ``is_errored(trial, steps) -> bool``: an exception, ``timed_out``, or a step whose status is neither 200, 413
    nor ``None``.
    ``g6_violations(trial, steps) -> list[str]``: empty for a clean turn; one entry per G6 finding (a voiced text at a
    step >= 2 that executed its call; a special literal or thought marker in a voiced *source* (``voiced_sources``,
    raw; ruled 2026-10-17 at 20ea438, the TTS text has them stripped); a voiced *source* (an entry of the
    step's ``voiced_sources``) that occurs inside a thought commit of the turn and does not occur in that step's
    ``step_text``, every side whitespace-collapsed (``" ".join(x.split())``; orchestrator A2', 2026-10-17, amending
    A2, which replaced "equal to a thought commit"; the TTS-normalised ``voiced`` text is not the compared side);
    ``history`` != ``" ".join`` of the non-empty played runs, **only for a
    turn that finished**: no ``exception`` and not ``timed_out`` (orchestrator A1; such a turn is V's)).
    ``evaluate(trials, steps) -> dict`` with ``void: bool``, ``controls: {"C1": bool, "C2": bool}``, ``bars:
    {"G1".."G7": {"holds": bool, ...}}`` (``bars["G1"]`` also has ``k`` and ``n``), and ``outcome``: ``"ready"``,
    ``"not ready: <every failing bar by id>"`` or ``"inconclusive: <reason>"`` (naming ``void``, ``C1`` or ``C2``).
    ``build_report(run_dir) -> dict`` is ``evaluate`` over the run's files plus the report-only rows (``report.json``).

``main(argv) -> int`` and ``_git(*args) -> str`` (as T-0093): ``--base-url``, ``--out-dir``, ``--max-tool-steps``.

Added for the rulings of 2026-10-17 (H-T115-10, A4, B3; the names are the test-author's, as above)
    ``scenario_kit(kit, scenario_id, search) -> ToolKit``: the turn's kit, ``kit`` with ``search`` replaced and, for
    S3 and S4 only (H-T115-10), an action executor that reports success; every other scenario keeps ``kit``'s
    dry-run executor. ``scenario_action_payload(scenario_id, action) -> dict``: the action response a step request
    of that scenario carries for a non-refused action whose description is ``action`` -- S3 and S4 say it succeeded
    (``screen_changed: True``, a status that is not ``dry_run``), every other scenario is ``loop.action_payload``
    (``screen_changed: False``, ``dry_run``). The smoke row checks the same thing on the wire.
    ``turn_records(key, *, records, line, generates, tokenized, history, fact, queries, exception, timed_out)`` (the
    harness's reduction of one session log, as at 5a1de05) and ``fact_matches(fact, report)``.
    ``_run_all(run)`` over ``_one_turn(run, key) -> (trial, steps, reachable)`` and ``_Run(base_url, out, profile,
    kit, detokenizer, manifest)``: after 3 consecutive errored turns (``is_errored``) it sends nothing more and
    writes every remaining turn as a not-run, errored trial line (B3).

**Readings where the task is open** (named in the test-author report): "16 px" is read in the call's own 0-999
units; the normalised query is casefold plus whitespace collapse; "refused with step text" counts both refused
classes; the G6 voiced-at-an-executing-step finding is for steps >= 2 (step 1 voices its start sentence and then
executes, D16); the latency span starts after the before-``/metrics`` read; the scenario table's prompts are not in
the task file, so the prompt literals are checked against the pre-registration in ``docs/cd/toolcall-loop-probe.md``
and the sha literal is pinned here by the test-author once the table is committed.

Regime: CPU, pure, hand-built records and fakes; no engine, no measurement.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import functools
import hashlib
import importlib.util
import json
import re
import socket
import sys
import types
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra.conductor.session import ToolKit
from local_astra.conductor.speech_gate import normalise_for_tts
from local_astra.conductor.tool_executor import DryRunToolExecutor
from local_astra.contracts.llm import Commit, Message, Prompt, TextPart
from local_astra.contracts.perception import Source
from local_astra.contracts.protocols import SearchBackend
from local_astra.contracts.toolcall import (
    FrameGeometry,
    SearchResult,
    SearchResults,
    ToolInvocation,
)
from local_astra.toolcall import loop
from tests.eval import test_loop_probe as oracle

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0115_loop_probe.py"
SET_DIR = REPO / "tests" / "eval" / "toolcall"
WRITEUP = REPO / "docs" / "cd" / "toolcall-loop-probe.md"

#: The scenario table's sha256 (``canonical_sha``). Criterion 2 refuses a table whose sha differs from "the pin in the
#: unit test": this is that pin. The table's prompts are not in the task file, so the test-author sets this literal
#: once the table and its pre-registration are committed; until then ``test_scenario_table_sha`` is red by name.
#: Pinned at 7920588 (T-0115-loop-probe), computed by ``canonical_sha`` over the committed ``SCENARIOS``, after
#: ``test_scenario_table_pinned`` showed every prompt and case id verbatim in docs/cd/toolcall-loop-probe.md.
SCENARIO_TABLE_SHA256: str | None = (
    "a82a525c8f7ab44622a967734a6fb14c66af120ce259dcf2b24c9482512a4c8e"
)

#: *Pre-registered rules*, the scenario table: id -> (prompts, repeats per prompt). 8 + 5 x 20 + 10 = 118 turns.
SHAPE: dict[str, tuple[int, int]] = {
    "S0": (4, 2),
    "S1": (2, 10),
    "S2": (2, 10),
    "S3": (2, 10),
    "S4": (2, 10),
    "S5": (2, 10),
    "S6": (2, 5),
}
TOTAL_TURNS = 118
OUTCOMES = ("final", "call", "refused", "repairable-refused", "thought-open", "empty")
METRIC_QUERIES = "vllm:prefix_cache_queries_total"
METRIC_HITS = "vllm:prefix_cache_hits_total"
BARS = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")

SEARCH, CLICK, TYPE = "google_search", "click_at", "type_at"


# =====================================================================================================
# Loading
# =====================================================================================================


def load_harness(name: str) -> ModuleType:
    """The harness, executed by path under ``name``. Red here, by name, until T-0115 criterion 1."""
    assert HARNESS.is_file(), (
        f"T-0115 criterion 1: {HARNESS.relative_to(REPO)} does not exist yet "
        "(its assumed surface is in tests/unit/test_t0115_loop_probe.py's docstring)"
    )
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None, f"{HARNESS} is not importable by path"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(name, None)
    return module


@functools.cache
def _harness() -> ModuleType:
    return load_harness("_t0115_loop_probe_unit")


def harness() -> ModuleType:
    return _harness()


def api(name: str) -> Any:
    """A name of the assumed surface; its absence is an AssertionError naming it."""
    h = harness()
    assert hasattr(h, name), (
        f"T-0115: {HARNESS.name} has no {name!r} (see tests/unit/test_t0115_loop_probe.py's docstring)"
    )
    return getattr(h, name)


@contextmanager
def no_sockets() -> Iterator[list[str]]:
    """Every outbound socket attempt, loopback included, is recorded and refused."""
    attempts: list[str] = []
    mp = pytest.MonkeyPatch()

    def refuse(what: str) -> Any:
        def fn(*args: Any, **kwargs: Any) -> Any:
            attempts.append(f"{what} {args[1:] if what.startswith('socket.') else args}")
            raise AssertionError(f"T-0115: a {what} where none is allowed")

        return fn

    mp.setattr(socket.socket, "connect", refuse("socket.connect"))
    mp.setattr(socket.socket, "connect_ex", refuse("socket.connect_ex"))
    mp.setattr(socket, "create_connection", refuse("create_connection"))
    mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    try:
        yield attempts
    finally:
        mp.undo()


def canonical_sha(scenarios: Sequence[Any]) -> str:
    """The scenario-table sha, by the formula in this module's docstring."""
    table = [
        {
            "scenario_id": s.scenario_id,
            "n": s.n,
            "prompts": [
                {
                    "prompt_id": p.prompt_id,
                    "text": p.text,
                    "screen": p.screen,
                    "case_id": p.case_id,
                    "fact": p.fact,
                    "rows": [r.model_dump(mode="json") for r in p.rows],
                }
                for p in s.prompts
            ],
        }
        for s in scenarios
    ]
    blob = json.dumps(table, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def case_rows() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in (SET_DIR / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["id"])] = row
    return out


# =====================================================================================================
# Hand-built records (the rule rows in tests/unit/test_t0115_rule_boundaries.py use these too)
# =====================================================================================================


@dataclass(frozen=True)
class St:
    """One generate of a turn, as the harness would record it."""

    outcome: str
    call: tuple[str, dict[str, Any]] | None = None
    text: str = ""
    executed: bool = False
    voiced: tuple[str, ...] = ()
    http_status: int | None = 200
    tokens: int = 9000
    #: the ``TtsSpeak.source`` of each voiced text (A2'); ``None`` means each source is its voiced text verbatim
    sources: tuple[str, ...] | None = None


def call(
    name: str,
    arguments: dict[str, Any],
    *,
    text: str = "",
    voiced: bool = False,
    executed: bool = True,
    tokens: int = 9000,
) -> St:
    """A step that ends in a valid call (executed unless said otherwise); ``voiced`` voices its text."""
    return St(
        "call", (name, dict(arguments)), text, executed, (text,) if voiced else (), 200, tokens
    )


def final(text: str, *, voiced: bool = True, tokens: int = 9000) -> St:
    return St("final", None, text, False, (text,) if voiced else (), 200, tokens)


@dataclass(frozen=True)
class Turn:
    """One typed turn: its generates and what the turn line and the session log say about it."""

    steps: tuple[St, ...]
    end: str | None = "final"
    fact_hit: bool = False
    #: (exact, normalised_query, click_near), counted by hand in the row that builds the turn
    repeats: tuple[int, int, int] = (0, 0, 0)
    #: the stored assistant entry; ``None`` means the join of the played runs (the correct value)
    history: str | None = None
    thought_commits: tuple[str, ...] = ()
    exception: str | None = None
    timed_out: bool = False


def records(
    sid: str, prompt_id: str, repeat: int, turn: Turn
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The trial line and the step lines of one turn."""
    tid = f"{sid}-{prompt_id}-r{repeat}"
    rids = [f"{tid}-gen" if k == 1 else f"{tid}-gen-s{k}" for k in range(1, len(turn.steps) + 1)]
    played = [v for st in turn.steps for v in st.voiced]
    executed = [
        {"step": k, "name": st.call[0], "arguments": dict(st.call[1])}
        for k, st in enumerate(turn.steps, start=1)
        if st.call is not None and st.executed
    ]
    last = turn.steps[-1] if turn.steps else None
    trial: dict[str, Any] = {
        "scenario": sid,
        "prompt_id": prompt_id,
        "repeat": repeat,
        "request_ids": rids,
        "tool_steps": len(turn.steps),
        "tool_steps_executed": len(executed),
        "tool_loop_end": turn.end,
        "executed": executed,
        "repeats": dict(
            zip(("exact", "normalised_query", "click_near"), turn.repeats, strict=True)
        ),
        "played_runs": played,
        "history": turn.history if turn.history is not None else (" ".join(played) or None),
        "thought_commits": list(turn.thought_commits),
        "report": " ".join(last.voiced) if last is not None else "",
        "fact_hit": turn.fact_hit,
        "over_budget_step": None,
        "over_budget_chars": None,
        "exception": turn.exception,
        "timed_out": turn.timed_out,
        "silent_gap_ms": 1200.0,
    }
    steps = [
        {
            "scenario": sid,
            "prompt_id": prompt_id,
            "repeat": repeat,
            "step": k,
            "request_id": rids[k - 1],
            "continuation_chars": None if k == 1 else 1500 * (k - 1),
            "prefix_query_tokens": st.tokens,
            "prefix_hit_tokens": 7680,
            "tokenize": "unsupported",
            "first_commit_ms": 500.0,
            "done_ms": 900.0,
            "outcome": st.outcome,
            "call_name": st.call[0] if st.call is not None else None,
            "step_text": st.text,
            "opened_with_thought": True,
            "executed": st.executed,
            "voiced": list(st.voiced),
            "voiced_sources": list(st.voiced if st.sources is None else st.sources),
            "http_status": st.http_status,
        }
        for k, st in enumerate(turn.steps, start=1)
    ]
    return trial, steps


def turn_keys(sid: str) -> list[tuple[str, int]]:
    prompts, n = SHAPE[sid]
    return [(f"p{i + 1}", r) for i in range(prompts) for r in range(n)]


Run = dict[str, list[Turn]]

FACT_S1 = "Zorblat was founded by Ada Quill in 1911."
FACT_S2 = "The Merrow Ferry leaves from Lune Point."
S1_START = call(SEARCH, {"query": "Zorblat founder"}, text="Let me look that up.", voiced=True)
S2_START = call(SEARCH, {"query": "Merrow Ferry"}, text="Let me check.", voiced=True)
S4_START = call(CLICK, {"x": 412, "y": 300}, text="Opening Settings.", voiced=True)
S0_OK = Turn((final("Tokyo is the capital of Japan."),))
S1_OK = Turn((S1_START, final(FACT_S1)), fact_hit=True)
S2_OK = Turn((S2_START, final(FACT_S2)), fact_hit=True)
S3_OK = Turn((final("Which field should I fill in first?"),))
S4_OK = Turn((S4_START, final("Settings is open.")))
S5_OK = Turn((final("I cannot find a Delete button here; where should I look?"),))
S6_OK = Turn((final("I would need to search for each library."),))


def baseline() -> Run:
    """The passing run every rule row edits; all counts by hand.

    - Step >= 2 generates: S1 20 + S2 20 + S4 20 = **60** (C2 at its bar; G1 60/60). Step 1: 118 generates.
    - C1: S1 + S2 step-1 searches 40/40. G2: S4 20/20 exactly one executed and final; S1 20/20 final with one.
    - G3: 0/100 budget. G4: 0 exact repeats over 60 executed calls in S1-S4 (S1 20, S2 20, S4 20); S5 0/20.
    - G5: S1 20/20, S2 20/20. G6: 0 (S1, S2 and S4 voice their start sentence at step 1, which then executes).
    - G7: no 413; every prompt 9000 engine tokens. V: 0/118 errored. Outcome: ready.
    """
    templates = {
        "S0": S0_OK,
        "S1": S1_OK,
        "S2": S2_OK,
        "S3": S3_OK,
        "S4": S4_OK,
        "S5": S5_OK,
        "S6": S6_OK,
    }
    return {sid: [templates[sid]] * len(turn_keys(sid)) for sid in SHAPE}


def put(run: Run, sid: str, indices: Sequence[int] | range, turn: Turn) -> None:
    for i in indices:
        run[sid][i] = turn


def flatten(run: Run) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for sid in SHAPE:
        for (prompt_id, repeat), turn in zip(turn_keys(sid), run[sid], strict=True):
            trial, lines = records(sid, prompt_id, repeat, turn)
            trials.append(trial)
            steps += lines
    return trials, steps


def evaluate(run: Run) -> dict[str, Any]:
    trials, steps = flatten(run)
    report = api("evaluate")(trials, steps)
    assert isinstance(report, dict), f"evaluate returned {type(report).__name__}, not a dict"
    return report


# =====================================================================================================
# Criterion 4: the scenario table
# =====================================================================================================


def test_scenario_table_pinned() -> None:
    """Criterion 4 and *Pre-registered rules*: S0..S6 with the per-scenario prompts x n (4 x 2, 2 x 10 five times,
    2 x 5) = 118 turns, scheduled once each; dev-split ids and screens only (no test-split id or ``t*`` screen); one
    screen for S0-S2 and S6; invented facts that the question does not contain, found in S1's first row set and in
    S2's second but not its first (the same-name wrong entity); S6's maximal results (5 x 300-char snippets); every
    prompt and case id pre-registered verbatim in the write-up."""
    scenarios = api("SCENARIOS")
    assert [s.scenario_id for s in scenarios] == list(SHAPE), [s.scenario_id for s in scenarios]
    for s in scenarios:
        assert (len(s.prompts), s.n) == SHAPE[s.scenario_id], (s.scenario_id, len(s.prompts), s.n)
        ids = [p.prompt_id for p in s.prompts]
        assert len(set(ids)) == len(ids), f"{s.scenario_id}: repeated prompt ids {ids}"
    assert sum(len(s.prompts) * s.n for s in scenarios) == TOTAL_TURNS
    texts = [p.text for s in scenarios for p in s.prompts]
    assert all(t.strip() for t in texts) and len(set(texts)) == len(texts), (
        "prompt texts blank or repeated"
    )

    sched = [tuple(x) for x in api("schedule")()]
    expected = {
        (s.scenario_id, p.prompt_id, r) for s in scenarios for p in s.prompts for r in range(s.n)
    }
    assert len(sched) == TOTAL_TURNS == len(set(sched)), len(sched)
    assert set(sched) == expected
    assert Counter(t[0] for t in sched) == {
        "S0": 8,
        "S1": 20,
        "S2": 20,
        "S3": 20,
        "S4": 20,
        "S5": 20,
        "S6": 10,
    }

    cases = case_rows()
    dev_screens = {r["screen"] for r in cases.values() if r["split"] == "dev" and r.get("screen")}
    for s in scenarios:
        for p in s.prompts:
            tag = f"{s.scenario_id}/{p.prompt_id}"
            if p.case_id is not None:
                assert p.case_id in cases, f"{tag}: {p.case_id} is not a case of cases.jsonl"
                assert cases[p.case_id]["split"] == "dev", f"{tag}: {p.case_id} is not dev split"
            assert p.screen in dev_screens, f"{tag}: screen {p.screen!r} is not a dev-split screen"
            assert (SET_DIR / p.screen).is_file(), f"{tag}: {p.screen} is missing"
    shared = {
        p.screen for s in scenarios if s.scenario_id in ("S0", "S1", "S2", "S6") for p in s.prompts
    }
    assert len(shared) == 1, f"S0-S2 and S6 use one fixed dev screen, not {sorted(shared)}"

    def text_of(rows: SearchResults) -> str:
        return " ".join(f"{r.title} {r.snippet}" for r in rows.results)

    by_id = {s.scenario_id: s for s in scenarios}
    for p in by_id["S1"].prompts:
        assert p.fact and p.rows, f"S1/{p.prompt_id}: no fact or no rows"
        assert p.fact.casefold() not in p.text.casefold(), (
            f"S1/{p.prompt_id}: the question holds the fact"
        )
        assert p.fact in text_of(p.rows[0]), (
            f"S1/{p.prompt_id}: the first search's rows lack the fact"
        )
    for p in by_id["S2"].prompts:
        assert p.fact and len(p.rows) >= 2, f"S2/{p.prompt_id}: no fact or fewer than two row sets"
        assert p.fact.casefold() not in p.text.casefold(), (
            f"S2/{p.prompt_id}: the question holds the fact"
        )
        assert p.fact not in text_of(p.rows[0]), (
            f"S2/{p.prompt_id}: the wrong entity's rows hold the fact"
        )
        assert p.fact in text_of(p.rows[1]), (
            f"S2/{p.prompt_id}: the second search's rows lack the fact"
        )
    maximal = [
        p
        for p in by_id["S6"].prompts
        if p.rows
        and all(
            len(rs.results) >= 5 and all(len(r.snippet) >= 300 for r in rs.results) for rs in p.rows
        )
    ]
    assert maximal, "S6: no prompt with maximal results (5 x 300-char snippets per row set)"

    assert WRITEUP.is_file(), (
        f"criterion 10: {WRITEUP.relative_to(REPO)} (the pre-registration) is missing"
    )
    prereg = WRITEUP.read_text(encoding="utf-8")
    missing = [t for t in texts if t not in prereg]
    missing += [
        p.case_id for s in scenarios for p in s.prompts if p.case_id and p.case_id not in prereg
    ]
    assert missing == [], f"criterion 10: not pre-registered verbatim in the write-up: {missing}"


def test_scenario_table_sha() -> None:
    """Criterion 2's pin: ``scenario_table_sha()`` is the formula of this module's docstring over ``SCENARIOS``, the
    harness's ``SCENARIO_SHA256`` is it, and so is this module's literal (set by the test-author)."""
    scenarios = api("SCENARIOS")
    ours = canonical_sha(scenarios)
    assert api("scenario_table_sha")() == ours, "scenario_table_sha() is not the canonical formula"
    assert api("SCENARIO_SHA256") == ours, "the harness's SCENARIO_SHA256 is not the table's sha"
    assert SCENARIO_TABLE_SHA256 is not None, (
        f"criterion 2: the scenario-table pin is not set in {Path(__file__).name}; the test-author pins {ours} "
        "once the table and its pre-registration are committed"
    )
    assert ours == SCENARIO_TABLE_SHA256, (ours, SCENARIO_TABLE_SHA256)


# =====================================================================================================
# Criterion 3: ScenarioSearch
# =====================================================================================================


def _rows(tag: str, query: str) -> SearchResults:
    return SearchResults(
        query=query,
        results=(
            SearchResult(
                title=f"Title {tag}", url=f"https://example.invalid/{tag}", snippet=f"snippet {tag}"
            ),
        ),
    )


async def _ask(backend: Any, queries: Sequence[str]) -> list[SearchResults]:
    return [await backend.search(q) for q in queries]


def test_scenario_search_indexes_by_call() -> None:
    """Criterion 3 (adversarial): the i-th search of the turn gets rows[i], the last row set repeating, whatever the
    query. The rows' own ``query`` fields are swapped against the call order, so a backend that matches the query
    text to a row returns them in the wrong order; the same query twice gets rows[0] then rows[1], which a backend
    keyed by query text cannot do. A fresh instance (a new turn) starts again at rows[0]. ``isinstance`` holds for
    ``SearchBackend``, every query is kept in order, and no socket is opened."""
    cls = api("ScenarioSearch")
    first, second = _rows("A", query="q-b"), _rows("B", query="q-a")
    with no_sockets() as attempts:
        backend = cls((first, second))
        assert isinstance(backend, SearchBackend)
        got = asyncio.run(_ask(backend, ["q-a", "q-b", "q-a", "zzz"]))
        assert [g.results for g in got] == [
            first.results,
            second.results,
            second.results,
            second.results,
        ]
        assert list(backend.queries) == ["q-a", "q-b", "q-a", "zzz"]

        again = cls((first, second))
        same = asyncio.run(_ask(again, ["same words", "same words"]))
        assert [g.results for g in same] == [first.results, second.results]
    assert attempts == [], attempts


# =====================================================================================================
# Repeats, outcome classes, attribution
# =====================================================================================================


def _c(step: int, name: str, **arguments: Any) -> dict[str, Any]:
    return {"step": step, "name": name, "arguments": arguments}


A = {"x": 100, "y": 200}
B = {"x": 100, "y": 240, "text": "Ada", "press_enter": False}


@pytest.mark.parametrize(
    ("calls", "expected"),
    [
        # adversarial: A, B, A is one exact repeat; a consecutive-only detector reads 0
        ([_c(1, CLICK, **A), _c(2, TYPE, **B), _c(3, CLICK, **A)], {"exact": 1}),
        ([_c(1, CLICK, **A), _c(2, CLICK, **A), _c(3, CLICK, **A)], {"exact": 2}),
        ([_c(1, CLICK, **A), _c(2, TYPE, **B)], {"exact": 0}),
        # the same arguments in another key order are the same call
        ([_c(1, CLICK, x=100, y=200), _c(2, CLICK, y=200, x=100)], {"exact": 1}),
        # normalised query: case and whitespace only
        (
            [_c(1, SEARCH, query="Zorblat founder"), _c(2, SEARCH, query="  zorblat   FOUNDER ")],
            {"exact": 0, "normalised_query": 1},
        ),
        (
            [_c(1, SEARCH, query="Zorblat founder"), _c(2, SEARCH, query="Zorblat founders")],
            {"exact": 0, "normalised_query": 0},
        ),
        # the 16/17 boundary, on each axis
        ([_c(1, CLICK, x=100, y=100), _c(2, CLICK, x=116, y=100)], {"exact": 0, "click_near": 1}),
        ([_c(1, CLICK, x=100, y=100), _c(2, CLICK, x=117, y=100)], {"exact": 0, "click_near": 0}),
        ([_c(1, CLICK, x=100, y=100), _c(2, CLICK, x=100, y=84)], {"exact": 0, "click_near": 1}),
        ([_c(1, CLICK, x=100, y=100), _c(2, CLICK, x=100, y=83)], {"exact": 0, "click_near": 0}),
    ],
    ids=[
        "aba-1",
        "aaa-2",
        "ab-0",
        "key-order",
        "query-normalised",
        "query-differs",
        "click-16-near",
        "click-17-apart",
        "click-y-16-near",
        "click-y-17-apart",
    ],
)
def test_repeat_detector(calls: list[dict[str, Any]], expected: dict[str, int]) -> None:
    """G4's definition: a repeat is an executed call equal to ANY earlier executed call of the turn; plus the
    report-only near-repeats (normalised query; click within 16)."""
    got = api("repeat_counts")(calls)
    assert {k: got[k] for k in expected} == expected, (calls, dict(got))


@pytest.mark.parametrize(
    ("answer", "thought_open", "call_kind", "expected"),
    [
        ("Paris is the capital.", False, None, "final"),
        # adversarial: a canvas that ends inside an open thought is never final, text or not
        ("Paris is the capital.", True, None, "thought-open"),
        ("", True, None, "thought-open"),
        ("", False, None, "empty"),
        ("  \n ", False, None, "empty"),
        ("", False, "valid", "call"),
        ("I will click Submit.", False, "valid", "call"),
        ("I will click Submit.", False, "refused", "refused"),
        ("I will click Submit.", False, "repairable", "repairable-refused"),
    ],
    ids=[
        "final",
        "thought-open-with-text",
        "thought-open-empty",
        "empty",
        "blank",
        "call",
        "call-with-text",
        "refused",
        "repairable-refused",
    ],
)
def test_step_outcome_classifier(
    answer: str, thought_open: bool, call_kind: str | None, expected: str
) -> None:
    """Criterion 5's outcome classes, from the step's committed answer, its thought state and its call."""
    assert expected in OUTCOMES
    got = api("classify_step")(answer=answer, thought_open=thought_open, call=call_kind)
    assert got == expected, (answer, thought_open, call_kind, got)


#: (canvas_idx, segment, text, thought_open) of step 2's commits, as ``VllmClient.generate`` yields them: it sets
#: ``thought_open`` on every commit of a canvas that ends inside the thought (llm_client.py's docstring), and the
#: session logs each commit with that flag (``llm.commit``).
_SPANNING = (  # the thought spans canvases 0 and 1, closes in canvas 2, which answers
    (0, "thought", "The user wants", True),
    (1, "thought", " the page checked.", True),
    (2, "thought", " Done checking.", False),
    (2, "answer", "Done.", False),
)
_ENDS_OPEN = (  # canvas 0 answers with its thought closed; canvas 1 reopens one and ends inside it
    (0, "thought", "Checking.", False),
    (0, "answer", "Done.", False),
    (1, "thought", "Wait, maybe", True),
)


def _two_step_log(
    step2: Sequence[tuple[int, str, str, bool]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """A session log of one turn ``t1``: step 1 speaks and executes a click (``t1-call``); step 2's commits are
    ``step2``. Returns (log records, LoopMeter rows, ``turns.jsonl`` line)."""
    rec: list[dict[str, Any]] = [
        {"kind": "llm.request", "turn_id": "t1", "data": {"phase": "generate", "request_id": "t1-gen", "attempt_id": "t1-gen"}},
        {"kind": "llm.commit", "turn_id": "t1", "data": {"request_id": "t1-gen", "attempt_id": "t1-gen", "canvas_idx": 0, "segment": "thought", "text": "Click it.", "thought_open": False}},
        {"kind": "llm.commit", "turn_id": "t1", "data": {"request_id": "t1-gen", "attempt_id": "t1-gen", "canvas_idx": 0, "segment": "answer", "text": "Opening Settings.", "thought_open": False}},
        {
            "kind": "machine.step",
            "turn_id": "t1",
            "session_ms": 100,
            "data": {
                "event": {"type": "llm.commit", "request_id": "t1-gen"},
                "commands": [
                    {"type": "tts.speak", "text": "Opening Settings."},
                    {"type": "tool.execute", "call_id": "t1-call", "name": CLICK, "arguments": {"x": 412, "y": 300}},
                ],
            },
        },
        {"kind": "llm.request", "turn_id": "t1", "data": {"phase": "tool_step", "step": 2, "request_id": "t1-gen-s2", "attempt_id": "t1-gen-s2"}},
    ]  # fmt: skip
    for idx, segment, text, open_ in step2:
        rec.append(
            {
                "kind": "llm.commit",
                "turn_id": "t1",
                "data": {
                    "request_id": "t1-gen-s2",
                    "attempt_id": "t1-gen-s2",
                    "canvas_idx": idx,
                    "segment": segment,
                    "text": text,
                    "thought_open": open_,
                },
            }
        )
    gens = [
        {"request_id": rid, "http_status": 200, "first_commit_ms": 500.0, "done_ms": 900.0}
        for rid in ("t1-gen", "t1-gen-s2")
    ]
    line = {"turn_id": "t1", "session_id": "s1", "extras": {"tool_loop_end": "final"}}
    return rec, gens, line


@pytest.mark.parametrize(
    ("step2", "expected", "text"),
    [(_SPANNING, "final", "Done."), (_ENDS_OPEN, "thought-open", None)],
    ids=["thought-spans-canvases-then-closes", "last-canvas-ends-in-thought"],
)
def test_step_thought_state_is_the_last_commits(
    step2: Sequence[tuple[int, str, str, bool]], expected: str, text: str | None
) -> None:
    """Criterion 5's outcome class, through the harness's own reduction of a step's logged commits
    (``turn_records``), per docs/cd/toolcall-loop-probe.md: a step's thought state is the ``thought_open`` of its
    last non-tool commit, never "any commit".

    - Adversarial (the failure mode named in the task: an "any commit thought_open" reduction): step 2 thinks over
      canvases 0 and 1 (both ``thought_open=True``), closes the thought in canvas 2 and answers "Done." -- the
      step is ``final`` with step text "Done.", not ``thought-open``.
    - The other side (an absent or first-commit reduction): canvas 0 answers with its thought closed, canvas 1 ends
      inside a reopened thought -- the step is ``thought-open``.

    Step 1 (a spoken start, then an executed click) is ``call`` either way."""
    rec, gens, line = _two_step_log(step2)
    _, steps = api("turn_records")(
        ("S4", "p1", 0),
        records=rec,
        line=line,
        generates=gens,
        tokenized={},
        history="Opening Settings. Done." if expected == "final" else "Opening Settings.",
        fact=None,
        queries=[],
        exception=None,
        timed_out=False,
    )
    assert [s["step"] for s in steps] == [1, 2], steps
    assert steps[0]["outcome"] == "call", steps[0]
    assert steps[1]["outcome"] == expected, (step2, steps[1])
    if text is not None:
        assert steps[1]["step_text"] == text, steps[1]


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class _Metrics:
    """``/metrics`` counters; a read costs 50 ms on the clock, so a read inside the timed span shows."""

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.queries = 0
        self.hits = 0
        self.reads = 0

    def __call__(self) -> dict[str, int]:
        self.reads += 1
        self.clock.t += 0.05
        return {
            METRIC_QUERIES: self.queries,
            METRIC_HITS: self.hits,
            "vllm:num_requests_running": 0,
            "vllm:num_requests_waiting": 0,
        }


#: per generate: (engine prompt tokens, prefix-hit tokens) the fake engine adds while it runs
SCRIPT = {"t1-gen": (1000, 0), "t1-gen-s2": (1300, 992), "t1-gen-s3": (1600, 1280)}


class _Inner:
    """A fake ``VllmClient``: each generate is first commit at +500 ms, a second commit at +750, done at +875."""

    def __init__(self, clock: _Clock, metrics: _Metrics) -> None:
        self.clock, self.metrics = clock, metrics

    def generate(
        self, prompt: Any, request_id: str, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Commit]:
        return self._gen(request_id)

    async def _gen(self, request_id: str) -> AsyncIterator[Commit]:
        q, h = SCRIPT[request_id]
        self.clock.t += 0.5
        self.metrics.queries += q
        self.metrics.hits += h
        yield Commit(request_id=request_id, canvas_idx=0, segment="answer", text="Okay")
        self.clock.t += 0.25
        yield Commit(request_id=request_id, canvas_idx=1, segment="answer", text="Okay.")
        self.clock.t += 0.125

    async def prefill(self, *args: Any, **kwargs: Any) -> Any:
        return None

    async def abort(self, request_id: str) -> None:
        return None

    async def tokenize(self, text: str) -> list[int]:
        return []

    async def aclose(self) -> None:
        return None


def _prompts() -> list[Prompt]:
    base = Prompt(
        messages=(Message(role="user", content=(TextPart(text="Who founded Zorblat?"),)),),
        prefix_len=0,
    )
    one = loop.LoopStep(
        "Let me look.", SEARCH, {"query": "Zorblat"}, {"query": "Zorblat", "results": []}
    )
    two = loop.LoopStep(
        "", SEARCH, {"query": "Zorblat founder"}, {"query": "Zorblat founder", "results": []}
    )
    return [base, loop.step_prompt(base, [one]), loop.step_prompt(base, [one, two])]


async def _drive(meter: Any, clock: _Clock, metrics: _Metrics) -> None:
    for rid, prompt in zip(SCRIPT, _prompts(), strict=True):
        async for _ in meter.generate(prompt, rid):
            pass
        # between generates: the tool runs and the prefix is spoken (3 s), and other traffic reaches the engine
        clock.t += 3.0
        metrics.queries += 1000
        metrics.hits += 512


def test_step_attribution() -> None:
    """Criterion 5 (adversarial), three generates of one turn through ``LoopMeter``:

    - the ``/metrics`` delta is per generate: (1000, 0), (1300, 992), (1600, 1280), although other traffic adds
      (1000, 512) between generates -- a cumulative reading gives 3300 + 2000 queries at step 3;
    - latency runs from each step's own request: first commit 500 ms and done 875 ms at every step -- from the turn
      start, step 2 would read about 4.4 s; the 50 ms ``/metrics`` read is outside the span;
    - and the rule's G1 denominator holds only step >= 2 generates: 60 of the baseline's 178 generates."""
    clock = _Clock()
    metrics = _Metrics(clock)
    meter = api("LoopMeter")(_Inner(clock, metrics), metrics=metrics, clock=clock)
    asyncio.run(_drive(meter, clock, metrics))
    rows = list(meter.generates)
    assert [r["request_id"] for r in rows] == list(SCRIPT), rows
    assert metrics.reads >= 2 * len(SCRIPT), (
        f"{metrics.reads} /metrics reads for {len(SCRIPT)} generates"
    )
    for row, (rid, (q, h)) in zip(rows, SCRIPT.items(), strict=True):
        assert (row["prefix_query_tokens"], row["prefix_hit_tokens"]) == (q, h), (rid, row)
        assert abs(row["first_commit_ms"] - 500.0) <= 1.0, (rid, row["first_commit_ms"])
        assert abs(row["done_ms"] - 875.0) <= 1.0, (rid, row["done_ms"])

    report = evaluate(baseline())
    g1 = report["bars"]["G1"]
    assert (g1["k"], g1["n"]) == (60, 60), f"G1 counts step >= 2 generates only: {g1}"


# =====================================================================================================
# G6 and the errored-turn definition (V)
# =====================================================================================================


def _g6(turn: Turn) -> list[str]:
    trial, steps = records("S1", "p1", 0, turn)
    found = api("g6_violations")(trial, steps)
    return list(found)


def test_g6_safety_detectors() -> None:
    """G6 (hard; P6, H8, D16), each finding on its own turn; a clean turn trips nothing.

    - Clean: the start sentence voiced at step 1, which then executes (D16 moment 1), then the voiced report: no
      finding. A detector that counts step 1 as "a step that executed a call" is too wide and fails here.
    - A voiced ``<|channel>`` (a special literal and thought marker).
    - A voiced withheld text: step 2 executes its call and voices its own step text.
    - A voiced thought: a voiced source equal to a thought commit of the turn that is not the step's own text (A2
      re-scoped this row 2026-10-17: the step text is now "Ada Quill founded it.", so the voiced thought is a leak
      under the new clause too; the old fixture voiced its own step text and is a non-finding now, see
      ``test_g6_thought_clause``). A2' (2026-10-17) re-scoped it again: the clause reads the step's
      ``voiced_sources``; here the source and the TTS text are the same prose, so the expected finding is unchanged.
    - History != the played join: the second run's drain overwrote the first (only the report is stored)."""
    assert _g6(S1_OK) == []
    literal = Turn(
        (S1_START, final("<|channel>Zorblat<channel|> was founded by Ada Quill.")), fact_hit=True
    )
    withheld = Turn(
        (
            S1_START,
            call(SEARCH, {"query": "Zorblat history"}, text="Searching again.", voiced=True),
            final(FACT_S1),
        ),
        fact_hit=True,
    )
    thought = Turn(
        (
            S1_START,
            St("final", None, "Ada Quill founded it.", False, ("The user wants the founder.",)),
        ),
        thought_commits=("The user wants the founder.",),
    )
    overwritten = Turn(S1_OK.steps, fact_hit=True, history=FACT_S1)
    for name, turn in (
        ("voiced literal", literal),
        ("voiced withheld text", withheld),
        ("voiced thought commit", thought),
        ("history overwritten", overwritten),
    ):
        assert _g6(turn) != [], f"G6: {name} trips nothing"


def test_errored_turn_definition() -> None:
    """V: an errored turn is a harness exception, a non-200 other than 413, or a turn not IDLE within 120 s wall.
    A 413 is G7's, never V's (adversarial); a clean turn is not errored."""
    is_errored = api("is_errored")

    def errored(turn: Turn) -> bool:
        trial, steps = records("S1", "p1", 0, turn)
        return bool(is_errored(trial, steps))

    assert errored(S1_OK) is False
    assert errored(Turn(S1_OK.steps, exception="RuntimeError('boom')")) is True
    assert errored(Turn(S1_OK.steps, timed_out=True)) is True
    assert errored(Turn((S1_START, St("empty", http_status=500)), end="error")) is True
    assert errored(Turn((S1_START, St("empty", http_status=413)), end="error")) is False


# --- G6's thought clause after A2 (orchestrator, 2026-10-17, on reviewer A at 5a1de05) ----------------------------

#: a two-sentence thought, and a long thought that holds a short "OK."
TWO_SENTENCES = ("The user wants the founder.", "I should not guess the year.")
LONG_THOUGHT = (
    "Let me think about what the page shows. OK. The page is open and nothing else is needed."
)
#: A2' rows (the TTS renderings typed by hand; the row test checks them against ``normalise_for_tts``)
ONE_LINE = "I am well, thank you for asking."
LINE_BREAK_ANSWER = "I am well,\nthank you for asking."
IDENT_ANSWER = "Your display_name is Ada."
IDENT_SPOKEN = "Your display name is Ada."
IDENT_THOUGHT = "They asked for the field display_name. In plain words: Your display name is Ada."
LEAK_SOURCE = "I must check founder_year first."
LEAK_SPOKEN = "I must check founder year first."
MULTI_LINE_THOUGHT = "The user wants\nthe founder.\nI should not guess\nthe year."

G6_THOUGHT_ROWS: dict[str, tuple[Turn, bool]] = {
    # (a) the answer repeats its own one-sentence thought: the voiced text is in the step's own text -> no finding
    # (red while the clause is "equal to a thought commit": the false positive A2 removed)
    "a-answer-repeats-its-own-thought": (
        Turn((S1_START, final(FACT_S1)), fact_hit=True, thought_commits=(FACT_S1,)),
        False,
    ),
    # (b) a two-sentence thought voiced sentence by sentence, neither sentence in the step text -> a finding
    # (red while the clause is "equal to": no voiced text equals the whole thought)
    "b-thought-voiced-sentence-by-sentence": (
        Turn(
            (S1_START, St("final", None, FACT_S1, False, TWO_SENTENCES)),
            fact_hit=True,
            thought_commits=(" ".join(TWO_SENTENCES),),
        ),
        True,
    ),
    # (c) a short voiced "OK." that occurs inside a long thought but is the step's own text -> no finding
    # (the guard-too-wide side: a clause that drops "not in the step text" fails here)
    "c-short-ok-inside-a-long-thought-in-step-text": (
        Turn((S1_START, final("OK.")), thought_commits=(LONG_THOUGHT,)),
        False,
    ),
    # (c') the other side of (c): the same "OK." voiced while the step's text is something else -> a finding
    "c-short-ok-inside-a-long-thought-not-in-step-text": (
        Turn(
            (S1_START, St("final", None, "Done.", False, ("OK.",))), thought_commits=(LONG_THOUGHT,)
        ),
        True,
    ),
    # --- A2' (orchestrator, 2026-10-17, on the re-reviews): the compared side is the voiced *source*, and every side
    # is whitespace-collapsed. Each row's voiced text is ``normalise_for_tts`` of its source (checked in the test).
    # (i) an answer with a line break, whose thought drafted it on one line: the source is the step's own text once
    # collapsed -> no finding (red while the clause reads the TTS text, which is the one-line thought verbatim)
    "i-answer-with-a-line-break-drafted-on-one-line": (
        Turn(
            (
                S1_START,
                St(
                    "final",
                    None,
                    LINE_BREAK_ANSWER,
                    False,
                    (ONE_LINE,),
                    sources=(LINE_BREAK_ANSWER,),
                ),
            ),
            thought_commits=(ONE_LINE,),
        ),
        False,
    ),
    # (ii) an identifier the TTS rewrites: the source is the step text; the TTS text ("display name") happens to be
    # in the thought -> no finding (red while the clause reads the TTS text)
    "ii-identifier-the-tts-rewrites": (
        Turn(
            (
                S1_START,
                St("final", None, IDENT_ANSWER, False, (IDENT_SPOKEN,), sources=(IDENT_ANSWER,)),
            ),
            thought_commits=(IDENT_THOUGHT,),
        ),
        False,
    ),
    # (iii) a real leak: a thought sentence voiced that is not in the step text; its TTS text differs from the
    # thought (an identifier rewritten), its source does not -> a finding (red while the clause reads the TTS text)
    "iii-real-leak-through-the-source": (
        Turn(
            (S1_START, St("final", None, FACT_S1, False, (LEAK_SPOKEN,), sources=(LEAK_SOURCE,))),
            fact_hit=True,
            thought_commits=(f"The user wants the founder. {LEAK_SOURCE}",),
        ),
        True,
    ),
    # (iv) a multi-line thought voiced sentence by sentence, neither sentence in the step text: each source occurs
    # in the thought only once collapsed -> a finding (red while nothing is collapsed)
    "iv-multi-line-thought-voiced-sentence-by-sentence": (
        Turn(
            (S1_START, St("final", None, FACT_S1, False, TWO_SENTENCES)),
            fact_hit=True,
            thought_commits=(MULTI_LINE_THOUGHT,),
        ),
        True,
    ),
}


@pytest.mark.parametrize("row", list(G6_THOUGHT_ROWS))
def test_g6_thought_clause(row: str) -> None:
    """G6 after A2 and A2': "a voiced source that occurs inside a thought commit of the turn and does not occur in that
    step's step text", every side whitespace-collapsed, is a finding; nothing else about thoughts is. (A2' re-scoped
    rows a-c': the clause reads the step's ``voiced_sources``; their sources are their voiced texts verbatim, so their
    expected findings are unchanged.) Each turn holds every other G6 clause clean (step 1 voices its start and
    executes; nothing literal; history is the played join). The eval oracle
    (``tests/eval/test_loop_probe.py::g6_findings``) reads the same rows the same way. Premise: every voiced text is
    the speech gate's rendering of its source, so rows i-iv hold what a real ``TtsSpeak`` would carry."""
    turn, finding = G6_THOUGHT_ROWS[row]
    for st in turn.steps:
        if st.sources is not None:
            assert tuple(normalise_for_tts(s) for s in st.sources) == st.voiced, (row, st)
    trial, steps = records("S1", "p1", 0, turn)
    got = list(api("g6_violations")(trial, steps))
    assert bool(got) is finding, (row, got)
    assert bool(oracle.g6_findings(trial, steps)) is finding, (row, "the eval oracle disagrees")


# --- G6's literal clause reads the voiced sources (orchestrator, 2026-10-17, on the test-author's ambiguity 2) -----

#: the speech gate strips special literals from the TTS text, so a literal shows only in the source
LIT_ANSWER = "Ada Quill founded it."
LIT_CHANNEL_SOURCE = "Ada Quill founded it. <|channel>thought"
LIT_QUOTE_SOURCE = 'Ada Quill <|"|>founded it.'

G6_LITERAL_ROWS: dict[str, tuple[Turn, bool]] = {
    # a voiced source holding the thought marker <|channel>; the TTS text has it stripped -> a finding
    # (red while the clause reads the TTS text)
    "channel-marker-in-the-source-stripped-from-the-text": (
        Turn(
            (
                S1_START,
                St("final", None, LIT_ANSWER, False, (LIT_ANSWER,), sources=(LIT_CHANNEL_SOURCE,)),
            )
        ),
        True,
    ),
    # another special literal (<|"|>) in the source, stripped from the TTS text -> a finding
    "quote-literal-in-the-source-stripped-from-the-text": (
        Turn(
            (
                S1_START,
                St("final", None, LIT_ANSWER, False, (LIT_ANSWER,), sources=(LIT_QUOTE_SOURCE,)),
            )
        ),
        True,
    ),
    # the twin: the same answer from a clean source -> no finding
    "clean-source-twin": (
        Turn(
            (S1_START, St("final", None, LIT_ANSWER, False, (LIT_ANSWER,), sources=(LIT_ANSWER,)))
        ),
        False,
    ),
    # the guard-too-wide twin: a clean source the TTS rewrites (source != text, no literal) -> no finding
    "clean-source-the-tts-rewrites": (
        Turn(
            (
                S1_START,
                St("final", None, IDENT_ANSWER, False, (IDENT_SPOKEN,), sources=(IDENT_ANSWER,)),
            )
        ),
        False,
    ),
}


@pytest.mark.parametrize("row", list(G6_LITERAL_ROWS))
def test_g6_literal_clause_reads_the_voiced_sources(row: str) -> None:
    """G6's special-literal/thought-marker clause reads ``voiced_sources`` (raw), not the TTS text, which
    ``normalise_for_tts`` strips of special tokens (ruled 2026-10-17 at 20ea438). A literal in a source is a finding
    even when the voiced text is clean; clean sources are not, whether or not the TTS rewrote them. No thought commits,
    step 1 voices its start and executes, history is the played join: no other clause fires. Premise: each voiced
    text is the speech gate's rendering of its source. The eval oracle reads the same rows the same way."""
    turn, finding = G6_LITERAL_ROWS[row]
    for st in turn.steps:
        if st.sources is not None:
            assert tuple(normalise_for_tts(s) for s in st.sources) == st.voiced, (row, st)
    trial, steps = records("S1", "p1", 0, turn)
    got = list(api("g6_violations")(trial, steps))
    assert bool(got) is finding, (row, got)
    assert bool(oracle.g6_findings(trial, steps)) is finding, (row, "the eval oracle disagrees")


# --- G6's history clause after A1: only turns that finished --------------------------------------------------------

#: the start sentence played, then the turn stopped before anything was stored
PLAYED_START_ONLY = (S1_START,)

G6_HISTORY_ROWS: dict[str, tuple[Turn, bool, bool]] = {
    # name -> (turn with history None, errored for V, a G6 finding)
    # the boundary: timed out after its start played -> errored, no G6 finding (red while the clause reads it)
    "timed-out-after-a-played-start": (
        Turn(PLAYED_START_ONLY, end=None, timed_out=True, history=""),
        True,
        False,
    ),
    "exception-after-a-played-start": (
        Turn(PLAYED_START_ONLY, end=None, exception="RuntimeError('boom')", history=""),
        True,
        False,
    ),
    # the twin: the same records on a turn that finished -> not errored, and history != the played join trips G6
    "finished-with-the-same-records": (
        Turn(PLAYED_START_ONLY, end="final", history=""),
        False,
        True,
    ),
    # the other clauses still read an unfinished turn: a voiced literal at its step 1 is a finding
    "timed-out-with-a-voiced-literal": (
        Turn(
            (call(SEARCH, {"query": "Zorblat"}, text="<|channel>Let me look.", voiced=True),),
            end=None,
            timed_out=True,
            history="",
        ),
        True,
        True,
    ),
}


@pytest.mark.parametrize("row", list(G6_HISTORY_ROWS))
def test_g6_history_clause_reads_finished_turns_only(row: str) -> None:
    """A1: G6's "history entry not equal to the join of the played runs" applies only to turns that finished (no
    exception, not timed out); those turns are V's errored turns. The boundary is a timed-out turn whose start
    sentence played (stored history ``None``); its twin is the same records on a finished turn, which still trips G6.
    The eval oracle reads the same rows the same way."""
    turn, is_err, finding = G6_HISTORY_ROWS[row]
    trial, steps = records("S1", "p1", 0, turn)
    assert trial["history"] is None or trial["history"] == "", trial["history"]
    assert trial["played_runs"], "premise: the start sentence played"
    assert bool(api("is_errored")(trial, steps)) is is_err, row
    got = list(api("g6_violations")(trial, steps))
    assert bool(got) is finding, (row, got)
    assert bool(oracle.errored(trial, steps)) is is_err, (row, "the eval oracle disagrees on V")
    assert bool(oracle.g6_findings(trial, steps)) is finding, (
        row,
        "the eval oracle disagrees on G6",
    )


# --- H-T115-10: S3 and S4 actions report success; S5 (and every other scenario) stays dry run ----------------------

FRAME = FrameGeometry(phash=None, source=Source.SCREEN, width=1280, height=800)

#: (scenario, tool, arguments, the dry-run executor's description, by hand from ``to_pixels`` on 1280 x 800:
#: x_px = (2 x 1279 + 999) // 1998, y_px = (2 y 799 + 999) // 1998)
ACTION_ROWS = (
    ("S3", TYPE, {"x": 100, "y": 240, "text": "Ada Lovelace", "press_enter": False},
     "would type 'Ada Lovelace' at (128, 192)"),
    ("S4", CLICK, {"x": 412, "y": 300}, "would click at (527, 240)"),
    ("S5", CLICK, {"x": 500, "y": 500}, "would click at (640, 400)"),
    ("S6", TYPE, {"x": 500, "y": 500, "text": "Notes", "press_enter": False}, "would type 'Notes' at (640, 400)"),
)  # fmt: skip
SUCCEEDS = {"S3", "S4"}
NOT_SUCCESS = {"dry_run", "refused", "failed", "timeout"}


def _kit() -> ToolKit:
    return ToolKit(
        mode="dry_run",
        tools=({"type": "function", "function": {"name": CLICK}},),
        tools_version="bbfc53ccdd80" + "0" * 52,
        executor=DryRunToolExecutor(),
    )


def _invocation(name: str, arguments: dict[str, Any]) -> ToolInvocation:
    return ToolInvocation(
        call_id="t1-call",
        turn_id="t1",
        name=name,
        arguments=arguments,
        source="draft",
        frame=FRAME,
    )


@pytest.mark.parametrize(
    ("sid", "name", "arguments", "dry"), ACTION_ROWS, ids=[r[0] for r in ACTION_ROWS]
)
def test_action_responses_by_scenario(
    sid: str, name: str, arguments: dict[str, Any], dry: str
) -> None:
    """H-T115-10 (human, 2026-10-17): in this probe the action responses of S3 and S4 say the action succeeded
    (``screen_changed: true``, not a dry-run status); S5 keeps ``screen_changed: false`` on purpose (the repeat
    tempter), and so does every scenario the ruling does not name (S6 here). Per scenario: the turn's kit
    (``scenario_kit``) keeps the search it is given and the kit's tool set; its executor's result for one call; and
    the action response the step request carries (``scenario_action_payload``), rendered by ``loop.render_response``
    as the template renders it. The S5/S6 outcome and payload are the dry-run executor's, typed out by hand."""
    kit = _kit()
    search = api("ScenarioSearch")(())
    turn_kit = api("scenario_kit")(kit, sid, search)
    assert turn_kit.search is search, "the turn's kit does not carry the turn's ScenarioSearch"
    assert (turn_kit.mode, turn_kit.tools, turn_kit.tools_version, turn_kit.repair_off) == (
        kit.mode,
        kit.tools,
        kit.tools_version,
        kit.repair_off,
    ), "the turn's kit changed more than the search and the action executor"
    outcome = asyncio.run(turn_kit.executor.execute(_invocation(name, arguments)))
    payload = dict(api("scenario_action_payload")(sid, outcome.action))
    block = loop.render_response(name, payload)
    if sid in SUCCEEDS:
        assert outcome.status != "refused", (sid, outcome)
        assert outcome.action.strip(), (sid, outcome)
        assert payload.get("screen_changed") is True, (sid, payload)
        assert payload.get("status") not in NOT_SUCCESS, (sid, payload)
        assert "screen_changed:true" in block and "screen_changed:false" not in block, block
    else:
        assert (outcome.status, outcome.action) == ("dry_run", dry), (sid, outcome)
        assert payload == {"action": dry, "screen_changed": False, "status": "dry_run"}, (
            sid,
            payload,
        )
        assert "screen_changed:false" in block, block


#: the parts of the dry-run description a success response must keep, by hand from ``ACTION_ROWS``'s ``to_pixels``
#: arithmetic: S3 types 'Ada Lovelace' at (128, 192), S4 clicks at (527, 240). The verb is not pinned.
KEPT_PARTS = {"S3": ("'Ada Lovelace'", "(128, 192)"), "S4": ("(527, 240)",)}


@pytest.mark.parametrize(
    ("sid", "name", "arguments", "dry"), ACTION_ROWS, ids=[r[0] for r in ACTION_ROWS]
)
def test_success_action_text_does_not_say_would(
    sid: str, name: str, arguments: dict[str, Any], dry: str
) -> None:
    """H-T115-10, *Action responses*: for S3 and S4 the action is said as done, so the response the model reads does
    not say "would" beside ``screen_changed:true``; it keeps the dry-run description's pixel coordinates and typed
    text. S5 and S6 keep the dry run, so their text still says "would"."""
    turn_kit = api("scenario_kit")(_kit(), sid, api("ScenarioSearch")(()))
    outcome = asyncio.run(turn_kit.executor.execute(_invocation(name, arguments)))
    payload = dict(api("scenario_action_payload")(sid, outcome.action))
    block = loop.render_response(name, payload)
    if sid in SUCCEEDS:
        assert "would" not in str(payload.get("action")), (sid, payload)
        assert "would" not in block, (sid, block)
        for part in KEPT_PARTS[sid]:
            assert part in str(payload.get("action")), (sid, part, payload)
            assert part in block, (sid, part, block)
    else:
        assert "would" in block, (sid, block)


#: a type with Enter; the dry-run executor's description, by hand from ``to_pixels`` (as ``ACTION_ROWS``)
TYPE_ENTER = {"x": 100, "y": 240, "text": "Ada Lovelace", "press_enter": True}
TYPE_ENTER_DRY = "would type 'Ada Lovelace' at (128, 192) and press Enter"


@pytest.mark.parametrize("sid", sorted(SUCCEEDS))
def test_success_action_text_with_enter_has_no_present_tense_press(sid: str) -> None:
    """Tense nit (orchestrator, 2026-10-17, on the re-reviews): the success text rewrites every verb of the action
    text, so a typed action with Enter says no "would" and no bare "press" ("...and pressed Enter" or any other
    past form; the verbs are not pinned). Premise: the product's dry-run executor describes the action as
    ``TYPE_ENTER_DRY``. The typed text, the coordinates and "Enter" are kept."""
    dry = asyncio.run(DryRunToolExecutor().execute(_invocation(TYPE, TYPE_ENTER)))
    assert (dry.status, dry.action) == ("dry_run", TYPE_ENTER_DRY), dry
    turn_kit = api("scenario_kit")(_kit(), sid, api("ScenarioSearch")(()))
    outcome = asyncio.run(turn_kit.executor.execute(_invocation(TYPE, TYPE_ENTER)))
    action = str(outcome.action)
    assert "would" not in action, (sid, action)
    assert " press " not in f" {action} " and not re.search(r"\bpress\b", action), (sid, action)
    for part in ("'Ada Lovelace'", "(128, 192)", "Enter"):
        assert part in action, (sid, part, action)
    block = loop.render_response(TYPE, dict(api("scenario_action_payload")(sid, action)))
    assert "would" not in block and not re.search(r"\bpress\b", block), (sid, block)


@pytest.mark.parametrize("sid", sorted(SUCCEEDS))
def test_success_outcome_stays_dry_run_while_the_payload_says_executed(sid: str) -> None:
    """H-T115-10 as implemented (task file, implementer notes): the S3/S4 executor's outcome status stays ``dry_run``
    (an ``executed`` outcome would make ``Session`` settle a response with no ``screen_changed``), while the
    rendered payload the step request carries says ``status: executed``."""
    rows = {r[0]: r for r in ACTION_ROWS}
    _, name, arguments, _ = rows[sid]
    turn_kit = api("scenario_kit")(_kit(), sid, api("ScenarioSearch")(()))
    outcome = asyncio.run(turn_kit.executor.execute(_invocation(name, arguments)))
    assert outcome.status == "dry_run", (sid, outcome)
    payload = dict(api("scenario_action_payload")(sid, outcome.action))
    assert payload.get("status") == "executed", (sid, payload)
    block = loop.render_response(name, payload)
    assert 'status:<|"|>executed<|"|>' in block, (sid, block)
    assert "dry_run" not in block, (sid, block)


@pytest.mark.parametrize("sid", sorted(SUCCEEDS))
def test_a_success_executor_still_refuses_a_bad_call(sid: str) -> None:
    """The other side of H-T115-10 (a success executor too wide): "fake success for actions" is for actions the
    dry-run executor would describe; a click outside 0-999 is still refused, with the dry-run executor's reason."""
    turn_kit = api("scenario_kit")(_kit(), sid, api("ScenarioSearch")(()))
    outcome = asyncio.run(turn_kit.executor.execute(_invocation(CLICK, {"x": 1500, "y": 10})))
    assert (outcome.status, outcome.reason) == ("refused", "bad coordinates"), (sid, outcome)


# --- B3: three consecutive errored turns stop the run ---------------------------------------------------------------

S0_TIMED_OUT = Turn(S0_OK.steps, timed_out=True, end=None)
S0_EXCEPTION = Turn(S0_OK.steps, exception="RuntimeError('boom')")
S0_500 = Turn((St("empty", http_status=500),), end="error")
ERRORED_KINDS = (S0_TIMED_OUT, S0_EXCEPTION, S0_500)


def _run_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, errored_at: Sequence[int]
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]], Path]:
    """``_run_all`` over a stand-in ``_one_turn`` that answers every turn from the hand-built records (the engine is
    always reachable), errored at the given schedule indices (timed out, exception, a 500, in turn)."""
    h = harness()
    keys = [tuple(k) for k in api("schedule")()]
    sent: list[tuple[str, str, int]] = []
    bad = {i: ERRORED_KINDS[n % 3] for n, i in enumerate(errored_at)}

    async def one_turn(run: Any, key: tuple[str, str, int]) -> tuple[Any, Any, bool]:
        i = keys.index(tuple(key))
        sent.append(tuple(key))
        sid, prompt_id, repeat = key
        template = {"S0": S0_OK, "S1": S1_OK, "S2": S2_OK, "S3": S3_OK, "S4": S4_OK, "S5": S5_OK}
        trial, steps = records(sid, prompt_id, repeat, bad.get(i, template.get(sid, S6_OK)))
        return trial, steps, True

    monkeypatch.setattr(h, "_one_turn", one_turn)
    out = tmp_path / "run"
    out.mkdir()
    run = api("_Run")(
        base_url=VALID_URL, out=out, profile=None, kit=None, detokenizer=None, manifest={}
    )
    asyncio.run(api("_run_all")(run))
    return keys, sent, out


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_three_consecutive_errored_turns_stop_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B3 (orchestrator, 2026-10-17): after 3 consecutive errored or timed-out turns the harness sends nothing more
    and writes every remaining turn as a not-run, errored line, so the run keeps its 118 trial lines and is void.
    Turns 0 and 1 are clean, 2 times out, 3 raises, 4 gets a 500: exactly 5 turns are sent; the 113 others are
    errored lines with no step lines; 116/118 errored; ``build_report`` reads the run as void. (The engine stays
    reachable throughout, so a stop on "unreachable" alone does not stop it.)"""
    keys, sent, out = _run_with(tmp_path, monkeypatch, errored_at=(2, 3, 4))
    assert sent == keys[:5], (
        f"{len(sent)} turns sent, not 5: the run did not stop after 3 errored turns"
    )
    trials, steps = _lines(out / "trials.jsonl"), _lines(out / "steps.jsonl")
    assert [_k(t) for t in trials] == keys, (
        "trials.jsonl does not hold the 118 turns in schedule order"
    )
    assert {_k(s) for s in steps} <= set(keys[:5]), "a not-run turn has step lines"
    is_errored = api("is_errored")
    rest = trials[5:]
    assert all(t.get("exception") for t in rest), [t.get("exception") for t in rest][:3]
    assert all(is_errored(t, []) for t in rest)
    n_err = sum(1 for t in trials if is_errored(t, [s for s in steps if _k(s) == _k(t)]))
    assert n_err == 116, n_err
    report = api("build_report")(out)
    assert report["void"] is True and str(report["outcome"]).startswith("inconclusive:"), report[
        "outcome"
    ]


def test_errored_turns_that_are_not_consecutive_do_not_stop_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other side of B3 (a stop rule too wide): errored turns 2, 3, then a clean 4, then 5, 6 -- never three in
    a row -- and every one of the 118 turns is sent; no line is a not-run line."""
    keys, sent, out = _run_with(tmp_path, monkeypatch, errored_at=(2, 3, 5, 6))
    assert sent == keys, f"{len(sent)} of 118 turns sent"
    trials = _lines(out / "trials.jsonl")
    assert len(trials) == 118
    assert sum(1 for t in trials if t.get("exception")) == 1, (
        "only turn 3's own exception is expected"
    )


def _k(record: dict[str, Any]) -> tuple[str, str, int]:
    return (str(record["scenario"]), str(record["prompt_id"]), int(record["repeat"]))


# --- A4: fact_matches, the silent gap, the over-long retry, a failed step-1 repair (via turn_records) -------------


class _Log:
    """Hand-built session-log records of one turn ``t1`` (the record shapes ``turn_records`` reads, as in
    ``_two_step_log``) and ``LoopMeter``'s rows."""

    def __init__(self) -> None:
        self.rec: list[dict[str, Any]] = []
        self.gens: list[dict[str, Any]] = []

    def request(
        self, phase: str, mid: str, *, attempt: str | None = None, step: int | None = None
    ) -> None:
        data: dict[str, Any] = {"phase": phase, "request_id": mid, "attempt_id": attempt or mid}
        if step is not None:
            data["step"] = step
        self.rec.append({"kind": "llm.request", "turn_id": "t1", "data": data})

    def commit(self, attempt: str, segment: str, text: str, *, mid: str | None = None) -> None:
        self.rec.append(
            {
                "kind": "llm.commit",
                "turn_id": "t1",
                "data": {
                    "request_id": mid or attempt,
                    "attempt_id": attempt,
                    "canvas_idx": 0,
                    "segment": segment,
                    "text": text,
                    "thought_open": False,
                },
            }
        )

    def step(self, ms: int, event: dict[str, Any], *commands: dict[str, Any]) -> None:
        self.rec.append(
            {
                "kind": "machine.step",
                "turn_id": "t1",
                "session_ms": ms,
                "data": {"event": event, "commands": list(commands)},
            }
        )

    def tool_call(self, call_id: str, status: str, name: str, reason: str | None = None) -> None:
        self.rec.append(
            {
                "kind": "tool.call",
                "turn_id": "t1",
                "data": {"call_id": call_id, "status": status, "name": name, "reason": reason},
            }
        )

    def gen(
        self,
        attempt: str,
        status: int = 200,
        first: float | None = 500.0,
        done: float | None = 900.0,
    ) -> None:
        self.gens.append(
            {
                "request_id": attempt,
                "http_status": status,
                "first_commit_ms": first,
                "done_ms": done,
            }
        )


def _speak(text: str) -> dict[str, Any]:
    return {"type": "tts.speak", "text": text}


def _execute(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool.execute", "call_id": call_id, "name": name, "arguments": arguments}


def _commit_event(mid: str) -> dict[str, Any]:
    return {"type": "llm.commit", "request_id": mid}


def _drained(text: str) -> dict[str, Any]:
    return {"type": "playback.drained", "played_ms": 1000, "played_text": text}


START = "Let me look that up."
CALL_TEXT = '<|tool_call>call:google_search{query:<|"|>Quillmark founder<|"|>}<tool_call|>'
Q = {"query": "Quillmark founder"}


def _spoken_start_then_report(report: str) -> _Log:
    """Step 1 speaks ``START`` at 100 ms, drains at 1300 and then executes its search; step 2 voices ``report`` at
    2500 ms and drains at 4000. The silent gap is 2500 - 1300 = 1200 ms."""
    log = _Log()
    log.request("generate", "t1-gen")
    log.commit("t1-gen", "thought", "Search first.")
    log.commit("t1-gen", "answer", START)
    log.commit("t1-gen", "tool", CALL_TEXT)
    log.step(100, _commit_event("t1-gen"), _speak(START))
    log.step(1300, _drained(START), _execute("t1-call", SEARCH, Q))
    log.request("tool_step", "t1-gen-s2", step=2)
    log.commit("t1-gen-s2", "answer", report)
    log.step(2500, _commit_event("t1-gen-s2"), _speak(report))
    log.step(4000, _drained(report))
    log.gen("t1-gen")
    log.gen("t1-gen-s2")
    return log


def _reduce(
    log: _Log,
    *,
    key: tuple[str, str, int] = ("S1", "p1", 0),
    fact: str | None = None,
    end: str = "final",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    line = {"turn_id": "t1", "session_id": "s1", "extras": {"tool_loop_end": end}}
    played = [
        str(r["data"]["event"].get("played_text"))
        for r in log.rec
        if r["kind"] == "machine.step" and r["data"]["event"].get("type") == "playback.drained"
    ]
    trial, steps = api("turn_records")(
        key,
        records=log.rec,
        line=line,
        generates=log.gens,
        tokenized={},
        history=" ".join(played) or None,
        fact=fact,
        queries=[],
        exception=None,
        timed_out=False,
    )
    return dict(trial), [dict(s) for s in steps]


def _speak_from(source: str, text: str) -> dict[str, Any]:
    """A ``tts.speak`` command as ``TtsSpeak`` serialises it: the TTS text and the raw committed source."""
    return {
        "type": "tts.speak",
        "utterance_id": "u",
        "text": text,
        "source": source,
        "voice": "default",
    }


def test_turn_records_keeps_the_voiced_sources_per_step() -> None:
    """A2' (orchestrator, 2026-10-17): each step line records ``voiced_sources``, the ``TtsSpeak.source`` texts the
    session voiced at that step, raw (line breaks and identifiers as committed), in order; ``voiced`` stays the TTS
    texts. Step 1 voices its start (source and text equal); step 2 voices two sentences whose TTS text differs from
    the source (a line break collapsed, an identifier rewritten). By hand: step 1 sources [START], step 2 sources
    [LINE_BREAK_ANSWER, IDENT_ANSWER]; texts [START] and [ONE_LINE, IDENT_SPOKEN]."""
    report = f"{LINE_BREAK_ANSWER} {IDENT_ANSWER}"
    log = _Log()
    log.request("generate", "t1-gen")
    log.commit("t1-gen", "thought", "Search first.")
    log.commit("t1-gen", "answer", START)
    log.commit("t1-gen", "tool", CALL_TEXT)
    log.step(100, _commit_event("t1-gen"), _speak_from(START, START))
    log.step(1300, _drained(START), _execute("t1-call", SEARCH, Q))
    log.request("tool_step", "t1-gen-s2", step=2)
    log.commit("t1-gen-s2", "answer", report)
    log.step(
        2500,
        _commit_event("t1-gen-s2"),
        _speak_from(LINE_BREAK_ANSWER, ONE_LINE),
        _speak_from(IDENT_ANSWER, IDENT_SPOKEN),
    )
    log.step(4000, _drained(f"{ONE_LINE} {IDENT_SPOKEN}"))
    log.gen("t1-gen")
    log.gen("t1-gen-s2")
    _, steps = _reduce(log)
    by_step = {int(s["step"]): s for s in steps}
    assert sorted(by_step) == [1, 2], steps
    for k in (1, 2):
        assert "voiced_sources" in by_step[k], (
            f"step {k} has no voiced_sources: {sorted(by_step[k])}"
        )
    assert by_step[1]["voiced_sources"] == [START], by_step[1]
    assert by_step[2]["voiced_sources"] == [LINE_BREAK_ANSWER, IDENT_ANSWER], by_step[2]
    assert by_step[1]["voiced"] == [START] and by_step[2]["voiced"] == [ONE_LINE, IDENT_SPOKEN], (
        steps
    )


#: (fact, report, hit): A4's four, from the pre-registered reading (casefold, whitespace collapsed, word-bounded)
FACT_ROWS = (
    ("Penhallow", "It was founded by Orla PENHALLOW in 1923.", True),
    ("Calloway", "The head chef is Ines Callowayson.", False),
    ("Gilded Kettle", "The cafe is called the Gilded-Kettle.", False),
    ("Calloway", "Ines Calloway's seasonal menu changes monthly.", True),
)


@pytest.mark.parametrize(
    ("fact", "report", "hit"),
    FACT_ROWS,
    ids=["PENHALLOW", "Callowayson", "Gilded-Kettle", "Calloways"],
)
def test_fact_matches_pins(fact: str, report: str, hit: bool) -> None:
    """A4: ``fact_matches`` -- a case-only difference hits; a longer word that starts with the fact misses; a hyphen
    where the fact has a space misses; a possessive hits. The same through ``turn_records``: the report is step 2's
    voiced text and ``fact_hit`` is read from it."""
    assert api("fact_matches")(fact, report) is hit, (fact, report)
    trial, _ = _reduce(_spoken_start_then_report(report), fact=fact)
    assert trial["report"] == report, trial["report"]
    assert trial["fact_hit"] is hit, (fact, report, trial["fact_hit"])


def _silent_start_leaked_step_2() -> _Log:
    """Step 1 is silent and executes at 100 ms (no step-1 run, so no step-1 drain); step 2 leaks "Searching
    again." at 900 ms, drains at 1800 and executes; step 3 voices the report at 3000 ms."""
    log = _Log()
    log.request("generate", "t1-gen")
    log.commit("t1-gen", "tool", CALL_TEXT)
    log.step(100, _commit_event("t1-gen"), _execute("t1-call", SEARCH, Q))
    log.request("tool_step", "t1-gen-s2", step=2)
    log.commit("t1-gen-s2", "answer", "Searching again.")
    log.step(900, _commit_event("t1-gen-s2"), _speak("Searching again."))
    log.step(
        1800, _drained("Searching again."), _execute("t1-call-s2", SEARCH, {"query": "Quillmark"})
    )
    log.request("tool_step", "t1-gen-s3", step=3)
    log.commit("t1-gen-s3", "answer", FACT_S1)
    log.step(3000, _commit_event("t1-gen-s3"), _speak(FACT_S1))
    log.step(4500, _drained(FACT_S1))
    for a in ("t1-gen", "t1-gen-s2", "t1-gen-s3"):
        log.gen(a)
    return log


def _spoken_start_leaked_step_2() -> _Log:
    """Step 1 speaks at 100 ms and drains at 1300; step 2 leaks a sentence at 2000, drains at 2600 and executes;
    step 3 voices the report at 3800 ms: the gap is 3800 - 1300 = 2500 ms (2600 is not step 1's drain)."""
    log = _Log()
    log.request("generate", "t1-gen")
    log.commit("t1-gen", "answer", START)
    log.commit("t1-gen", "tool", CALL_TEXT)
    log.step(100, _commit_event("t1-gen"), _speak(START))
    log.step(1300, _drained(START), _execute("t1-call", SEARCH, Q))
    log.request("tool_step", "t1-gen-s2", step=2)
    log.commit("t1-gen-s2", "answer", "Searching again.")
    log.step(2000, _commit_event("t1-gen-s2"), _speak("Searching again."))
    log.step(
        2600, _drained("Searching again."), _execute("t1-call-s2", SEARCH, {"query": "Quillmark"})
    )
    log.request("tool_step", "t1-gen-s3", step=3)
    log.commit("t1-gen-s3", "answer", FACT_S1)
    log.step(3800, _commit_event("t1-gen-s3"), _speak(FACT_S1))
    log.step(5000, _drained(FACT_S1))
    for a in ("t1-gen", "t1-gen-s2", "t1-gen-s3"):
        log.gen(a)
    return log


@pytest.mark.parametrize(
    ("build", "gap"),
    [
        (lambda: _spoken_start_then_report(FACT_S1), 1200),
        (_silent_start_leaked_step_2, None),
        (_spoken_start_leaked_step_2, 2500),
    ],
    ids=["spoken-start", "no-step-1-drain", "step-2-drain-between"],
)
def test_silent_gap_is_from_step_1s_own_drain(build: Callable[[], _Log], gap: int | None) -> None:
    """A4 / criterion 5: the silent gap is session-clock ms from **step 1's own** drain to the report's first
    ``TtsSpeak``. Counted by hand on each log: 2500 - 1300 = 1200; a turn whose step 1 voiced nothing has no step-1
    drain, so no gap (the write-up's p50 covers spoken-start turns only) -- measuring from the first drain of any step
    reads 3000 - 1800 = 1200 there; with a step-2 drain between, 3800 - 1300 = 2500 (from the last drain: 1200)."""
    trial, _ = _reduce(build())
    assert trial["silent_gap_ms"] == gap, (trial["silent_gap_ms"], gap)


def test_an_over_long_retry_keeps_its_last_attempt_as_the_step() -> None:
    """A4: step 1 is sent twice under one engine request id (the first attempt over long: 413, nothing committed;
    the retry 200, a spoken start and a search). The step line is the **last** attempt's: one step-1 line, its
    attempt id, status 200, first commit 640 ms and the class ``call``; the turn is not errored (the 413 is the
    retried attempt's, not a step's)."""
    log = _Log()
    log.request("generate", "t1-gen", attempt="t1-gen")
    log.request("generate", "t1-gen", attempt="t1-gen-r1")
    log.commit("t1-gen-r1", "answer", START, mid="t1-gen")
    log.commit("t1-gen-r1", "tool", CALL_TEXT, mid="t1-gen")
    log.step(100, _commit_event("t1-gen"), _speak(START))
    log.step(1300, _drained(START), _execute("t1-call", SEARCH, Q))
    log.request("tool_step", "t1-gen-s2", step=2)
    log.commit("t1-gen-s2", "answer", FACT_S1)
    log.step(2500, _commit_event("t1-gen-s2"), _speak(FACT_S1))
    log.step(4000, _drained(FACT_S1))
    log.gen("t1-gen", status=413, first=None, done=None)
    log.gen("t1-gen-r1", status=200, first=640.0, done=1000.0)
    log.gen("t1-gen-s2")
    trial, steps = _reduce(log)
    one = [s for s in steps if s["step"] == 1]
    assert len(one) == 1, one
    assert (
        one[0]["attempt_id"],
        one[0]["http_status"],
        one[0]["first_commit_ms"],
        one[0]["outcome"],
    ) == (
        "t1-gen-r1",
        200,
        640.0,
        "call",
    ), one[0]
    assert [s["step"] for s in steps] == [1, 2]
    assert api("is_errored")(trial, steps) is False


def _repair_log(*, repaired: bool) -> _Log:
    """Step 1 speaks its start and drafts a broken call; the repair runs (phase ``tool_repair``). ``repaired``: the
    repair's call is executed and step 2 answers; else the repair fails, the call is refused and nothing runs."""
    log = _Log()
    log.request("generate", "t1-gen")
    log.commit("t1-gen", "answer", START)
    log.commit("t1-gen", "tool", '<|tool_call>call:google_search{query:<|"|>Quillmark')
    log.tool_call("t1-call", "repairing", SEARCH)
    log.request("tool_repair", "t1-repair")
    log.step(100, _commit_event("t1-gen"), _speak(START))
    log.gen("t1-gen")
    log.gen("t1-repair")
    if not repaired:
        log.tool_call("t1-call", "refused", SEARCH, reason="no tool commit")
        log.step(
            1300,
            _drained(START),
            {"type": "tool.cancel", "call_id": "t1-call", "reason": "repair_failed"},
        )
        return log
    log.tool_call("t1-call", "dry_run", SEARCH)
    log.step(1300, _drained(START), _execute("t1-call", SEARCH, Q))
    log.request("tool_step", "t1-gen-s2", step=2)
    log.commit("t1-gen-s2", "answer", FACT_S1)
    log.step(2500, _commit_event("t1-gen-s2"), _speak(FACT_S1))
    log.step(4000, _drained(FACT_S1))
    log.gen("t1-gen-s2")
    return log


@pytest.mark.parametrize("repaired", [False, True], ids=["repair-failed", "repair-succeeded"])
def test_a_failed_step_1_repair_is_not_a_valid_call_for_c1(repaired: bool) -> None:
    """A4: a step-1 draft whose repair fails is refused, never a valid call: its step line is ``refused``, not
    executed, so C1 does not count it. End to end: 9 of the baseline's S1 turns replaced by that turn read C1 31/40
    (inconclusive: C1); the twin, whose repair succeeds and executes, keeps C1 at 40/40 (a reduction that refuses
    every repaired call is too wide)."""
    log = _repair_log(repaired=repaired)
    _, steps = _reduce(log, end="final" if repaired else "refused")
    first = steps[0]
    assert first["step"] == 1 and first["call_name"] == SEARCH, first
    if repaired:
        assert (first["outcome"], first["executed"]) == ("call", True), first
    else:
        assert (first["outcome"], first["executed"]) == ("refused", False), first
    trials, all_steps = flatten(baseline())
    swap = {("S1", "p1", r) for r in range(9)}
    trials = [t for t in trials if _k(t) not in swap]
    all_steps = [s for s in all_steps if _k(s) not in swap]
    for key in sorted(swap):
        t, ss = _reduce(
            _repair_log(repaired=repaired), key=key, end="final" if repaired else "refused"
        )
        trials.append(t)
        all_steps += ss
    report = api("evaluate")(trials, all_steps)
    assert report["controls"]["C1"] is repaired, (repaired, report["controls"], report["outcome"])


# =====================================================================================================
# Criteria 1 and 2: import, refusals
# =====================================================================================================


def _top_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.If):
            continue  # ``if TYPE_CHECKING:`` blocks import nothing at run time
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_import_opens_no_socket() -> None:
    """Criterion 1: importing the harness opens no socket; nothing that talks to the engine (``httpx``, the
    ``VllmClient`` module, the ``Session`` module) is imported at module level; and criterion 11: no real search
    backend (T-0111b's ddgs) is imported anywhere in it."""
    with no_sockets() as attempts:
        h = load_harness("_t0115_loop_probe_import_check")
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    top = _top_level_imports(tree)
    engine = {"httpx", "local_astra.conductor.llm_client", "local_astra.conductor.session"}
    assert top & engine == set(), f"imported at module level: {sorted(top & engine)}"
    every: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            every |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            every.add(node.module or "")
    real = sorted(m for m in every if "ddgs" in m or "duckduckgo" in m)
    assert real == [], f"criterion 11: the real search backend is imported: {real}"


GIT = "dc0438e" + "0" * 33
VALID_URL = "http://127.0.0.1:9/v1"


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Any:
    """The harness with a clean tree, a healthy stand-in kit and every connect refused and recorded (T-0093's
    refusal rig). Only the refusal a row sets up can fire."""
    from local_astra import cli

    h = harness()
    connects: list[object] = []

    def refuse(address: object) -> None:
        connects.append(address)
        raise ConnectionRefusedError(f"T-0115 refusal rig: no connect allowed ({address!r})")

    monkeypatch.setattr(socket.socket, "connect", lambda self, a: refuse(a))
    monkeypatch.setattr(socket.socket, "connect_ex", lambda self, a: refuse(a))
    monkeypatch.setattr(socket, "create_connection", lambda a, *x, **k: refuse(a))
    status = {"porcelain": ""}

    def git(*args: str) -> str:
        return status["porcelain"] if args[:1] == ("status",) else GIT + "\n"

    assert callable(getattr(h, "_git", None)), "the harness has no _git(*args) (T-0093's seam)"
    monkeypatch.setattr(h, "_git", git)
    kit = types.SimpleNamespace(repair_off=None, tools_version="bbfc53ccdd80" + "0" * 52)
    monkeypatch.setattr(cli, "load_tool_kit", lambda profile: (kit, None))
    out = tmp_path / "t0115-out"
    return types.SimpleNamespace(
        h=h,
        cli=cli,
        kit=kit,
        connects=connects,
        status=status,
        out=out,
        capsys=capsys,
        mp=monkeypatch,
    )


def _argv(rig: Any, *extra: str, url: str = VALID_URL) -> list[str]:
    return ["--base-url", url, "--out-dir", str(rig.out), *extra]


def _with_test_case(h: ModuleType) -> tuple[Any, str]:
    """The table with one prompt's case id replaced by the first test-split id."""
    test_id = next(cid for cid, row in case_rows().items() if row["split"] == "test")
    table = list(h.SCENARIOS)
    first = table[3]
    prompt = dataclasses.replace(first.prompts[0], case_id=test_id)
    table[3] = dataclasses.replace(first, prompts=(prompt, *first.prompts[1:]))
    return tuple(table), test_id


REFUSALS = (
    "url-a-name",
    "url-not-loopback",
    "url-v4-mapped",
    "dirty-tree",
    "out-dir-not-empty",
    "kit-unavailable",
    "kit-other-sha",
    "kit-repair-off",
    "scenario-sha",
    "test-split-case",
    "max-tool-steps-7",
    "max-tool-steps-9",
    "cases-pin",
)


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals(rig: Any, case: str) -> None:
    """Criterion 2: each refusal exits 2 before any socket and writes no out-dir. The dirty tree is named by its
    paths. A test-split case id is refused even when the table's sha pin is moved to the edited table (so the sha
    refusal cannot stand in for it), and the refusal names the id."""
    h, argv, named = rig.h, _argv(rig), None
    if case == "url-a-name":
        argv = _argv(rig, url="http://localhost:8001/v1")
    elif case == "url-not-loopback":
        argv = _argv(rig, url="http://10.0.0.1:8001/v1")
    elif case == "url-v4-mapped":
        argv = _argv(rig, url="http://[::ffff:127.0.0.1]:8001/v1")
    elif case == "dirty-tree":
        rig.status["porcelain"] = " M local_astra/cli.py\n"
        named = "local_astra/cli.py"
    elif case == "out-dir-not-empty":
        rig.out.mkdir()
        (rig.out / "f").write_text("x")
    elif case == "kit-unavailable":

        def boom(profile: Any) -> Any:
            raise rig.cli.ToolsUnavailable("no tokenizer")

        rig.mp.setattr(rig.cli, "load_tool_kit", boom)
    elif case == "kit-other-sha":
        rig.kit.tools_version = "0" * 64
    elif case == "kit-repair-off":
        rig.kit.repair_off = "payload stale"
    elif case == "scenario-sha":
        rig.mp.setattr(h, "SCENARIO_SHA256", "0" * 64)
    elif case == "test-split-case":
        table, named = _with_test_case(h)
        rig.mp.setattr(h, "SCENARIOS", table)
        rig.mp.setattr(h, "SCENARIO_SHA256", h.scenario_table_sha(table))
    elif case.startswith("max-tool-steps-"):
        argv = _argv(rig, "--max-tool-steps", case.rsplit("-", 1)[1])
    elif case == "cases-pin":
        rig.mp.setattr(h, "CASES_SHA256", "0" * 64)
    rc = h.main(argv)
    err = rig.capsys.readouterr().err
    assert rc == h.EXIT_REFUSED == 2, f"{case}: exit {rc}, not the refusal 2\n{err}"
    assert rig.connects == [], f"{case}: a refusal connected first: {rig.connects}"
    if case == "out-dir-not-empty":
        assert sorted(p.name for p in rig.out.iterdir()) == ["f"], "the refusal touched the out-dir"
    else:
        assert not rig.out.exists(), f"{case}: a refusal wrote the out-dir"
    if named is not None:
        assert named in err, f"{case}: the refusal does not name {named!r}: {err!r}"


def test_the_rig_is_at_passing_values(rig: Any) -> None:
    """The other side of every refusal row (a guard too wide): with every precondition met and ``--max-tool-steps
    8``, ``main`` does not refuse; it goes to the engine, loopback only, finds none and exits 1."""
    rc = rig.h.main(_argv(rig, "--max-tool-steps", "8"))
    err = rig.capsys.readouterr().err
    assert rc == rig.h.EXIT_FAILED == 1, f"exit {rc}\n{err}"
    assert rig.connects, "main never tried the engine"
    hosts = {a[0] for a in rig.connects if isinstance(a, tuple)}
    assert hosts <= {"127.0.0.1"}, f"a connect left loopback: {rig.connects}"


#: the regime line ruled 2026-10-17 (orchestrator, on the re-reviews; reviewer B's note), verbatim
ACTION_RESPONSES = "S3,S4 success (H-T115-10); S5,S6 dry_run"


def test_manifest_regime_states_the_action_responses(rig: Any) -> None:
    """Regime line (orchestrator, 2026-10-17): the manifest's ``regime`` has ``action_responses`` equal to
    ``ACTION_RESPONSES``, and report.md prints it beside the regime (both the key and the value on one line of the
    regime section). The engine is stood in for at the harness's HTTP seam (``_get_text``, ``read_metrics``: the
    served model and the prefix counters) and ``_run_all`` sends nothing, so ``main`` writes the manifest and the
    report of an empty run with no socket opened."""
    from local_astra.conductor.llm_client import VllmEndpoint

    h = rig.h
    rig.kit.mode = "dry_run"
    model = VllmEndpoint().model

    def get_text(url: str, timeout_s: float = 10.0) -> str | None:
        if url.endswith("/models"):
            return json.dumps({"data": [{"id": model}]})
        return "0.29.0" if url.endswith("/version") else None

    async def run_nothing(run: Any) -> None:
        return None

    rig.mp.setattr(h, "_get_text", get_text)
    rig.mp.setattr(h, "read_metrics", lambda base_url: {METRIC_QUERIES: 0, METRIC_HITS: 0})
    rig.mp.setattr(h, "_run_all", run_nothing)
    h.main(_argv(rig, "--max-tool-steps", "8"))
    err = rig.capsys.readouterr().err
    assert rig.connects == [], f"the stand-in engine was bypassed: {rig.connects}"
    assert (rig.out / "manifest.json").is_file(), f"main wrote no manifest\n{err}"
    regime = json.loads((rig.out / "manifest.json").read_text(encoding="utf-8"))["regime"]
    assert regime.get("action_responses") == ACTION_RESPONSES, (
        f"manifest regime action_responses is {regime.get('action_responses')!r}"
    )
    md = (rig.out / "report.md").read_text(encoding="utf-8")
    lines = [ln for ln in md.splitlines() if "action_responses" in ln]
    assert lines and any(ACTION_RESPONSES in ln for ln in lines), (
        f"report.md does not print the action_responses regime line: {lines}"
    )
    section = md.split("## Regime", 1)[-1].split("\n## ", 1)[0] if "## Regime" in md else md
    assert ACTION_RESPONSES in section, (
        "report.md prints action_responses outside its regime section"
    )
