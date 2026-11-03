"""T-0107a criterion 7 (negative): turns that do not ask are unchanged against the literal base d3765de, and the
paths the task does not own are untouched.

A separate module because the base run is a subprocess over ``git archive d3765de`` (built by T-0114's ``_base_tree``
with its ``BASE_SHA`` monkeypatched for the call; its ``local_astra`` first on ``sys.path``, the origin printed and
checked), so under ``--dist loadfile`` it is a worker of its own. The base run is made once per module (a
module-scoped fixture) and every row compares its own in-process session with it. The harness is T-0117's, T-0121's
and T-0126's (``_PolicyMachine``, ``_with_history``, ``_assert_equal``, ``_ls_tree``), imported, not copied.

Every row here holds at the base (the working tree is the base's code until the implementation lands) and must stay
green; none is pending (docs/tasks/T-0107a.md, "Tests to write").

The scenarios (none asks, by criterion 1's definition, worked by hand for each voiced text):

- **T-0092c tool turns, loop off, one session:** a valid click after ``SPOKEN``, a repairable click and its valid
  repair, a repair still invalid, a nameless draft, a ``<cc>`` commit, a narrated google_search with no prefix, an
  out-of-range click, an out-of-list key; then four turns whose prefix holds a ``?`` that ends no sentence (the bare
  and the quoted "Forgot password?" label, ``3?4``, ``what? no idea.``), each with a valid click -- the
  "contains ?" mutation's power rows;
- **T-0114 loop turns, loop on, one session:** the two-step search turn, the three-step click turn, and a two-step
  turn whose withheld step-2 text holds the label form before a valid click.

Readings (listed in the test-author's report):

- ``_assert_equal``'s message names T-0121's base; every call here passes a ``what`` naming T-0107a and d3765de;
- the byte-identity rows compare the closed range ``T0107A_BASE..T0107A_TIP`` = d3765de..27777bf (T-0094; the
  orchestrator's brief of 2026-10-19, T-0124/T-0126's ``_prove_open_range``), never the working tree; TIP is the
  branch's last commit touching non-test files and moves if the implementer commits again;
- "No new MachineState field, command, event, timer or ToolCancelReason" (criterion 5) is pinned here by the
  byte-identity of ``turn_machine.py`` and ``contracts/**`` (``MachineState`` lives in ``contracts/turn.py``,
  ``ToolCancelReason`` in ``contracts/commands.py``); "no HUD element" by ``hud/**``.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``dry_run``, loop off
or on as each scenario says, n = 1, deterministic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.llm import Commit
from local_astra.contracts.turn import TurnState
from tests import _git_range as gr
from tests.fakes.search import FakeSearch
from tests.unit import test_session_tool_loop as tsl
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins
from tests.unit._tool_turn_support import (
    CC_CALL,
    NAMELESS,
    OUT_OF_LIST_KEY,
    OUT_OF_RANGE_CLICK,
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    SEARCH_NARRATED,
    SPOKEN,
    STILL_INVALID,
    VALID_CLICK,
    answer,
    thought,
    tool,
    tool_rig,
)
from tests.unit.test_t0117_differential import _ls_tree, _PolicyMachine
from tests.unit.test_t0121_differential import _assert_equal, _with_history

REPO = Path(__file__).resolve().parents[2]
#: T-0107a's base (the branch was cut from main at d3765de; the orchestrator's brief, "BASE = d3765de"). A literal,
#: never a moving ref (tests.md).
BASE_SHA = "d3765decf5a53949885edf401352415ed8749eda"

SECTIONS = ("machine", "hud", "prompts", "requests", "log", "records", "history")
LOOP = tsl.LOOP

#: (question, step-1 script, repair continuation or None)
Turn = tuple[str, tuple[Commit, ...], tuple[Commit, ...] | None]
#: (question, step-1 script, step scripts)
LoopTurn = tuple[str, tuple[Commit, ...], tuple[tuple[Commit, ...], ...]]

#: the four non-asking ``?`` prefixes (criterion 1's false forms that a voiced prefix can carry)
LABELS = (
    "I will click the Forgot password? link. ",
    "I will click the 'Forgot password?' link. ",
    "Is it 3?4 ",
    "what? no idea. I will click it. ",
)

TOOL_TURNS: tuple[Turn, ...] = (
    (
        "click the sign in button",
        (thought("the user wants it"), answer(SPOKEN), tool(VALID_CLICK)),
        None,
    ),
    ("click the small one", (answer(SPOKEN), tool(REPAIRABLE_CLICK)), (tool(REPAIRED_CLICK),)),
    ("click it again", (answer(SPOKEN), tool(REPAIRABLE_CLICK)), (tool(STILL_INVALID),)),
    ("open example dot org", (answer(SPOKEN), tool(NAMELESS)), None),
    ("click the corner", (answer(SPOKEN), tool(CC_CALL)), None),
    ("weather in lisbon", (tool(SEARCH_NARRATED),), None),
    ("click the far corner", (answer(SPOKEN), tool(OUT_OF_RANGE_CLICK)), None),
    ("save it", (answer(SPOKEN), tool(OUT_OF_LIST_KEY)), None),
    *((f"label {i}", (answer(text), tool(VALID_CLICK)), None) for i, text in enumerate(LABELS)),
)

LOOP_TURNS: tuple[LoopTurn, ...] = (
    (tsl.QUESTION, (answer(tsl.PREFIX), tool(tsl.SEARCH_CALL)), ((answer(tsl.REPORT),),)),
    (
        "open the settings",
        (answer(tsl.S1), tool(VALID_CLICK)),
        ((answer(tsl.S2), tool(tsl.CLICK2)), (answer(tsl.S3),)),
    ),
    (
        "reset my password",
        (answer(tsl.S1), tool(VALID_CLICK)),
        ((answer(LABELS[1]), tool(tsl.CLICK2)), (answer(tsl.S3),)),
    ),
)


async def _until_turns(rig: Any, n: int, what: str) -> None:
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n,
        f"T-0107a criterion 7: {what} never ended",
        limit_ms=30_000,
    )
    await rig.advance(300)


async def _tool_turns(tmp: Path, turns: Sequence[Turn] = TOOL_TURNS) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, machine_cls=_PolicyMachine) as rig:
        for i, (question, script, continuation) in enumerate(turns):
            rig.llm.script = list(script)
            if continuation is not None:
                rig.llm.script_continuation(*continuation)
            await rig.ask(question)
            await _until_turns(rig, i + 1, repr(question))
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path)


async def _loop_turns(tmp: Path, turns: Sequence[LoopTurn] = LOOP_TURNS) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, machine_cls=_PolicyMachine, policy=LOOP, search=FakeSearch()) as rig:
        for i, (question, script, steps) in enumerate(turns):
            rig.llm.script = list(script)
            rig.llm.script_steps(*steps)
            await rig.ask(question)
            await tsl.until_steps(rig, len(steps), turn=f"t{i + 1}")
            await _until_turns(rig, i + 1, repr(question))
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path)


SCENARIOS = {"tool_turns": _tool_turns, "loop_turns": _loop_turns}


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario session. Runs in-process on the working tree and in a subprocess on the d3765de tree."""
    return {name: await run(tmp / name) for name, run in SCENARIOS.items()}


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0107a_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The d3765de run of every scenario, once per module."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0107a_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_SHA)
        base_dir = tsl._base_tree(tmp / "base_tree")
    out = tmp / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {data['origin']}, not from the {BASE_SHA} tree"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


def _same(expected: Any, got: Any, what: str) -> None:
    _assert_equal(
        expected,
        got,
        f"T-0107a {what} (base {BASE_SHA[:7]}; message prefix below is T-0121's helper):",
    )


def _statuses(sections: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in sections["log"]:
        if r["kind"] == "tool.call":
            out.setdefault(r["data"]["call_id"], []).append(r["data"]["status"])
    return out


# =====================================================================================================
# Criterion 7: the differential
# =====================================================================================================


async def test_non_asking_turns_equal_d3765de(base: dict[str, Any], tmp_path: Path) -> None:
    """Criterion 7, second item (mutations 1, "contains ?", and 6, the log fields on every call): T-0092c's tool
    turns, four turns whose ``?`` ends no sentence, and T-0114's loop turns (one with the label form in a withheld
    step-2 text) give the same machine steps, HUD messages, prompts, request bodies, session-log records,
    ``turns.jsonl`` lines and history as d3765de."""
    now = observed(  # T-0155 O-8: the persona sentence only
        {
            "tool_turns": await _tool_turns(tmp_path / "tools"),
            "loop_turns": await _loop_turns(tmp_path / "loop"),
        }
    )
    for scenario, sections in now.items():
        for section in SECTIONS:
            _same(
                base[scenario][section], sections[section], f"criterion 7: {scenario} {section!r}"
            )
    # premises, by hand: the scenarios ran the paths they name
    tools = _statuses(now["tool_turns"])
    assert tools["t1-call"] == ["proposed", "dry_run"], tools
    assert "repairing" in tools["t2-call"] and tools["t2-call"][-1] == "dry_run", tools
    assert tools["t3-call"][-1] == "refused" and tools["t4-call"] == ["refused"], tools
    labels = [f"t{9 + i}-call" for i in range(len(LABELS))]
    assert all(tools[c] == ["proposed", "dry_run"] for c in labels), (
        f"premise: every label turn executed at the base; {[tools.get(c) for c in labels]}"
    )
    stored = [h["assistant_spoken"] for h in now["tool_turns"]["history"]]
    # history keeps the last three turns (``TurnPolicy.max_history_turns = 3``): the last three label prefixes
    assert stored == list(LABELS[-3:]), stored
    loop = _statuses(now["loop_turns"])
    assert loop["t2-call-s2"][-1] == "dry_run" and loop["t3-call-s2"][-1] == "dry_run", loop
    steps = [r for r in now["loop_turns"]["log"]
             if r["kind"] == "llm.request" and r["data"].get("phase") == "tool_step"]  # fmt: skip
    assert len(steps) == 1 + 2 + 2, (
        f"premise: 5 step requests over the three loop turns; got {len(steps)}"
    )


# =====================================================================================================
# Criterion 7: byte-identity over the closed range T0107A_BASE..T0107A_TIP (T-0094)
# =====================================================================================================

#: criterion 7, first item, written out from the task
FROZEN = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/contracts",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/native.py",
    "local_astra/toolcall/router.py",
    "local_astra/toolcall/live.py",
    "local_astra/toolcall/loop.py",
    "local_astra/toolcall/warmup.py",
    "local_astra/toolcall/tools.py",
    "local_astra/conductor/executor.py",
    "local_astra/conductor/tool_executor.py",
    "local_astra/cli.py",
    "local_astra/config.py",
    "hud",
    "tests/fakes",
)
#: the single files of ``FROZEN`` (the rest are directories)
FROZEN_FILES = tuple(p for p in FROZEN if p not in ("local_astra/contracts", "hud", "tests/fakes"))

#: criterion 7, third item, and criterion 6's last item: the modules that stay green *unedited*
UNEDITED_TESTS = (
    "tests/unit/test_session_tool_turn.py",
    "tests/unit/test_session_tool_loop.py",
    "tests/unit/test_session_loop.py",
    "tests/unit/test_session_tools_off_golden.py",
    "tests/fixtures/session/t0092b_tools_off_golden.json",
    "tests/unit/test_session_stale_playback_golden.py",
    "tests/fixtures/session/t0092a_golden.json",
    "tests/unit/test_t0117_differential.py",
    "tests/unit/test_t0121_differential.py",
    "tests/unit/test_t0124_differential.py",
    "tests/unit/test_t0126_differential.py",
    "tests/unit/test_speaking_policy_doc.py",
    "tests/unit/_tool_turn_support.py",
)


#: The byte-identity rows are closed over T-0107a's own range (tests.md "pin both ends", T-0094; orchestrator brief
#: 2026-10-19, as T-0124/T-0126 did with ``_prove_open_range``): BASE = d3765de (the task's literal base), TIP =
#: 27777bf (the branch's last commit touching non-test files: the rulings line replacing the predicate definition;
#: the implementation is d4047c9). The working tree is never read, so a later task's legitimate edit of a guarded
#: path cannot redden these rows.
#:
#: **TIP moves**: if the implementer commits again to any non-test file (``ask_first.py``, ``session.py``, docs),
#: ``T0107A_TIP`` must be bumped to that commit before review. There is no merge commit yet, so ``gr.prove_range``
#: (which needs M) cannot run; the rows check its pre-merge half (both commits present, BASE an ancestor of TIP). At
#: merge, whoever merges replaces it with ``gr.prove_range(T0107A_BASE, T0107A_TIP, M, later=...)`` if TIP is M^2.
T0107A_BASE = "d3765decf5a53949885edf401352415ed8749eda"
T0107A_TIP = "9fd1df320a1ed21edc07e36561ac6301c714c771"


def _prove_open_range() -> None:
    gr.require_commits(T0107A_BASE, T0107A_TIP)
    assert gr.is_ancestor(T0107A_BASE, T0107A_TIP), (
        f"{T0107A_BASE} is not an ancestor of {T0107A_TIP}"
    )
    assert T0107A_BASE == BASE_SHA, f"premise: T0107A_BASE is the task's literal base {BASE_SHA}"


def _diff_over_range(
    paths: tuple[str, ...],
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """(changed, added, removed, the BASE listing) of the files under ``paths`` between ``T0107A_BASE`` and
    ``T0107A_TIP`` (git blob hashes of both trees; never the working tree)."""
    old = _ls_tree(T0107A_BASE, paths)
    now = _ls_tree(T0107A_TIP, paths)
    added, removed = sorted(now.keys() - old.keys()), sorted(old.keys() - now.keys())
    changed = sorted(p for p in old.keys() & now.keys() if old[p] != now[p])
    return changed, added, removed, old


def test_frozen_paths_equal_d3765de() -> None:
    """Criterion 7, first item (and criterion 5's "no new MachineState field, command, event, timer or
    ToolCancelReason"; "no HUD element is added"): the machine, the contracts, the prompt builder, the tool
    declarations, the other ``toolcall`` modules, both executors, the CLI, the config, the HUD and the fakes are
    byte-identical (git blob hashes) at ``T0107A_BASE`` (d3765de) and ``T0107A_TIP`` (27777bf), with no file added
    or removed under them."""
    _prove_open_range()
    changed, added, removed, old = _diff_over_range(FROZEN)
    assert set(FROZEN_FILES) <= set(old), (
        f"premise: d3765de has the frozen files; missing {sorted(set(FROZEN_FILES) - set(old))}"
    )
    assert any(p.startswith("hud/") for p in old) and any(p.startswith("tests/fakes/") for p in old)
    assert not (changed or added or removed), (
        f"T-0107a criterion 7: frozen paths differ in {T0107A_BASE[:7]}..{T0107A_TIP[:7]}: changed {changed}, "
        f"added {added}, removed {removed}"
    )


def test_existing_modules_unedited_since_d3765de() -> None:
    """Criterion 7, third item (and criterion 6: ``test_speaking_policy_doc.py`` unedited): the session tool-turn,
    tool-loop and loop modules, both goldens and their fixtures, the T-0117/T-0121/T-0124/T-0126 differentials, the
    speaking-policy doc test and the shared rig are byte-identical at ``T0107A_BASE`` (d3765de) and ``T0107A_TIP``
    (27777bf). (A test-only commit after TIP is outside the range; review covers it.)"""
    _prove_open_range()
    changed, _added, removed, old = _diff_over_range(UNEDITED_TESTS)
    assert set(UNEDITED_TESTS) == set(old), (
        f"premise: d3765de has every listed module; missing {sorted(set(UNEDITED_TESTS) - set(old))}"
    )
    assert not (changed or removed), (
        f"T-0107a criterion 7: existing test modules edited in {T0107A_BASE[:7]}..{T0107A_TIP[:7]}: "
        f"changed {changed}, removed {removed}"
    )


def test_live_is_still_refused(tmp_path: Path) -> None:
    """Criterion 7, fourth item: ``tools = "live"`` was still refused at config load at T-0107a's TIP, naming
    docs/01 §3. T-0111g re-scope R4: a historical claim, re-pinned against ``git show
    <T0107A_TIP>:local_astra/config.py`` loaded in isolation (``live`` is a mode from T-0111g on)."""
    from tests.unit.test_t0111c_unchanged import config_at

    load = config_at(T0107A_TIP, tmp_path).load
    p = tmp_path / "astra.toml"
    p.write_text('profile = "lean"\n[profiles.lean]\ntools = "live"\n')
    with pytest.raises(ValueError) as excinfo:
        load(p)
    assert "docs/01 §3" in str(excinfo.value), str(excinfo.value)
