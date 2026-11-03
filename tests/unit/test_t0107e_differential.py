"""T-0107e criterion 10 (negative): turns that are not a clear no are unchanged against the literal base 9275442, and
the paths the task does not own are untouched.

A separate module because the base run is a subprocess over ``git archive 9275442`` (built by T-0114's ``_base_tree``
with its ``BASE_SHA`` monkeypatched for the call; its ``local_astra`` first on ``sys.path``, the origin printed and
checked), so under ``--dist loadfile`` it is a worker of its own. The base run is made once per module (a module-scoped
fixture) and the row compares its own in-process sessions with it. The harness is T-0107a's (``_tool_turns``,
``_loop_turns``, ``TOOL_TURNS``, ``LOOP_TURNS``, ``SECTIONS``, ``_statuses``), and through it T-0117's and T-0121's
(``_PolicyMachine``, ``_with_history``, ``_assert_equal``, ``_ls_tree``), imported, not copied.

Every row here holds at the base (the working tree is the base's code until the implementation lands) and must stay
green; none is pending (docs/tasks/T-0107e.md, "Tests to write").

The scenarios (none is a clear no by criterion 1's Definition; each power transcript is one of criterion 1's false or
known-miss rows, worked by hand):

- **T-0107a's scenarios:** its T-0092c tool turns and four non-asking ``?`` label turns (loop off), its T-0114 loop
  turns (loop on), and two asking turns (T-0107a's refusal, loop off and at loop step 2) -- the asked-first path is
  BASE behaviour too and must not move;
- **power turns (voice, loop off):** every criterion-1 false row that can be spoken (``""`` has no words and the
  ``[ui] …`` row is not a voice transcript, so both are left to ``test_session_user_declines``) and every known miss,
  each before ``SPOKEN`` and a valid click;
- **power turns (typed, loop off):** eight false rows typed through ``E.UserText`` (one ``Word`` holding the text);
- **power turns (loop on):** three false rows as the question of a two-step click turn.

Readings (listed in the test-author's report):

- ``_assert_equal``'s message names T-0121's base; every call here passes a ``what`` naming T-0107e and 9275442;
- the byte-identity rows are two-ended (T-0094): ``T0107E_BASE..T0107E_TIP`` = 9275442..d8979db (the branch's last
  commit touching non-test files), both literals, so the working tree is never read; while the task was open the tip
  end was the working tree (``T0107E_TIP = None``);
- "No new MachineState field, command, event, timer or ToolCancelReason" (criterion 6) is pinned here by the
  byte-identity of ``turn_machine.py`` and ``contracts/**``; "no HUD element" by ``hud/**``; "no conductor speech is
  added" by the differential (every voiced ``TtsSpeak`` of a non-declining turn equals the base's) and, for declined
  turns, by ``test_session_user_declines.assert_voiced_and_stored``.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``dry_run``, loop off or
on as each scenario says, n = 1, deterministic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_differential as a
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins
from tests.unit._tool_turn_support import SPOKEN, VALID_CLICK, answer, tool, tool_rig
from tests.unit.test_t0117_differential import _ls_tree, _PolicyMachine
from tests.unit.test_t0121_differential import _assert_equal, _with_history

REPO = Path(__file__).resolve().parents[2]
#: T-0107e's base (the branch T-0107e-honour-no was cut from main at 9275442; the orchestrator's brief, "BASE"). A
#: literal, never a moving ref (tests.md).
BASE_SHA = "927544280842e4827a4f2108d9729255c1e8d280"

#: criterion 1's false rows that can be spoken, verbatim
FALSE_SPOKEN = (
    "no wait, send it to Bob instead",
    "No, but send it tomorrow",
    "No. Open the settings instead.",
    "nothing",
    "know",
    "I know, do it",
    "Nobody is there",
    "Notify Bob",
    "No problem, go ahead",
    "No worries, send it",
    "No need to ask, just send it",
    "Cancel my 3pm meeting",
    "Stop the timer",
    "Annule la commande",
    "Don't forget to save",
    "Don't close the tab, click Save",
    "Don't send it to Alice, send it to Bob",
    "Don't worry, send it",
    "Don't worry about it",
    "Do not disturb mode on",
    "Non-stop flights to Paris",
    "No-code tools",
    "No. 10 Downing Street opening hours",
    "Oh no, the page crashed, reload it",
    "Which mark do you have ever seen? No",
    "Yes, go ahead.",
)
#: criterion 1's known misses, pinned false, verbatim
MISSES = (
    "Okay, no, leave it",
    "I don't want that, leave it",
    "no leave it",
    "Leave it.",
    "Never mind",
    "Ne l'envoie pas",
)
TYPED = (
    "Cancel my 3pm meeting",
    "Don't close the tab, click Save",
    "No, but send it tomorrow",
    "No worries, send it",
    "No. 10 Downing Street opening hours",
    "Don’t worry, send it",
    "no leave it",
    "Do not disturb mode on",
)

ASK = "Shall I place the order for 42.99 euros? "
CLICK: tuple[Commit, ...] = (answer(SPOKEN), tool(VALID_CLICK))

POWER_TURNS: tuple[a.Turn, ...] = tuple((q, CLICK, None) for q in FALSE_SPOKEN + MISSES)
ASKED_TURNS: tuple[a.Turn, ...] = (("buy it", (answer(ASK), tool(VALID_CLICK)), None),)
LOOP_POWER: tuple[a.LoopTurn, ...] = (
    *(
        (
            q,
            (answer(tsl.S1), tool(VALID_CLICK)),
            ((answer(tsl.S2), tool(tsl.CLICK2)), (answer(tsl.S3),)),
        )
        for q in ("Don't close the tab, click Save", "No, but send it tomorrow", "Stop the timer")
    ),
    # T-0107a's step-2 asking turn: one step request, refused at step 2 (asked the user first)
    ("open the settings", (answer(tsl.S1), tool(VALID_CLICK)), ((answer(ASK), tool(tsl.CLICK2)),)),
)


async def _typed_turns(tmp: Path, texts: tuple[str, ...] = TYPED) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, machine_cls=_PolicyMachine) as rig:
        for i, text in enumerate(texts):
            rig.llm.script = list(CLICK)
            rig.submit(E.UserText(t_ms=rig.clock.now_ms, text=text))
            await a._until_turns(rig, i + 1, repr(text))
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path)


async def _power_turns(tmp: Path) -> dict[str, Any]:
    return await a._tool_turns(tmp, POWER_TURNS)


async def _asked_turns(tmp: Path) -> dict[str, Any]:
    return await a._tool_turns(tmp, ASKED_TURNS)


async def _loop_power(tmp: Path) -> dict[str, Any]:
    return await a._loop_turns(tmp, LOOP_POWER)


SCENARIOS = {
    "t0107a_tool_turns": a._tool_turns,
    "t0107a_loop_turns": a._loop_turns,
    "asked_turns": _asked_turns,
    "power_turns": _power_turns,
    "typed_power": _typed_turns,
    "loop_power": _loop_power,
}


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario session. Runs in-process on the working tree and in a subprocess on the 9275442 tree."""
    return {name: await run(tmp / name) for name, run in SCENARIOS.items()}


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0107e_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The 9275442 run of every scenario, once per module."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0107e_base")
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
        timeout=600,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {data['origin']}, not from the {BASE_SHA} tree"
    )
    assert not (base_dir / "local_astra" / "toolcall" / "user_declines.py").exists(), (
        "premise: the base tree has no user_declines module"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


def _same(expected: Any, got: Any, what: str) -> None:
    _assert_equal(
        expected,
        got,
        f"T-0107e {what} (base {BASE_SHA[:7]}; message prefix below is T-0121's helper):",
    )


# =====================================================================================================
# Criterion 10: the differential
# =====================================================================================================


async def test_non_declining_turns_equal_9275442(base: dict[str, Any], tmp_path: Path) -> None:
    """Criterion 10, second item (mutations 1-4 and 8: a substring test, first word only, whole-transcript equality,
    no reversal check, the log fields on every call): T-0107a's scenarios, the T-0092c tool turns, the T-0114 loop
    turns, T-0107a's asking turns and power turns whose transcripts are criterion 1's false rows and known misses
    (voice, typed, loop on), each with a valid call, give the same machine steps, HUD messages, prompts, request
    bodies, session-log records, ``turns.jsonl`` lines and history as 9275442."""
    now = observed(await _capture(tmp_path / "now"))  # T-0155 O-8: the persona sentence only
    assert set(now) == set(base), (set(now), set(base))
    for scenario, sections in now.items():
        for section in a.SECTIONS:
            _same(
                base[scenario][section],
                sections[section],
                f"criterion 10: {scenario} {section!r}",
            )
    # premises, by hand: the scenarios ran the paths they name
    power = a._statuses(now["power_turns"])
    n = len(POWER_TURNS)
    assert all(power[f"t{i + 1}-call"] == ["proposed", "dry_run"] for i in range(n)), (
        f"premise: every power turn executed in dry run; {power}"
    )
    typed = a._statuses(now["typed_power"])
    assert all(typed[f"t{i + 1}-call"] == ["proposed", "dry_run"] for i in range(len(TYPED))), typed
    asked = a._statuses(now["asked_turns"])
    assert asked["t1-call"] == ["refused"], asked
    loop = a._statuses(now["loop_power"])
    assert all(loop[f"t{i}-call-s2"][-1] == "dry_run" for i in (1, 2, 3)), loop
    assert loop["t4-call-s2"] == ["refused"], loop
    users = [h["user_text"] for h in now["power_turns"]["history"]]
    assert users == list(MISSES[-3:]), (
        f"premise: history holds the last three power transcripts verbatim; {users}"
    )


# =====================================================================================================
# Criterion 10: byte-identity, two-ended (T-0094)
# =====================================================================================================

#: criterion 10, first item, written out from the task
FROZEN = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/contracts",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/conductor/executor.py",
    "local_astra/conductor/tool_executor.py",
    "local_astra/cli.py",
    "local_astra/config.py",
    "local_astra/toolcall",
    "hud",
    "tests/fakes",
    "scripts",
    "docs/01-design.md",
    "docs/cd",
)
#: the one file criterion 10 lets appear under ``FROZEN`` (the new module, owned by the implementer)
NEW_MODULE = "local_astra/toolcall/user_declines.py"
FROZEN_FILES = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/conductor/executor.py",
    "local_astra/conductor/tool_executor.py",
    "local_astra/cli.py",
    "local_astra/config.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/ask_first.py",
    "local_astra/toolcall/live.py",
    "docs/01-design.md",
)

#: criterion 10, third item: the modules that stay green *unedited*
UNEDITED_TESTS = (
    "tests/unit/test_session_tool_turn.py",
    "tests/unit/test_session_tool_loop.py",
    "tests/unit/test_session_loop.py",
    "tests/unit/test_session_ask_first.py",
    "tests/unit/test_ask_first.py",
    "tests/unit/test_t0107a_differential.py",
    "tests/unit/test_t0107a_docs.py",
    "tests/unit/test_session_tools_off_golden.py",
    "tests/fixtures/session/t0092b_tools_off_golden.json",
    "tests/unit/test_session_stale_playback_golden.py",
    "tests/fixtures/session/t0092a_golden.json",
    "tests/unit/test_t0117_differential.py",
    "tests/unit/test_t0121_differential.py",
    "tests/unit/test_t0124_differential.py",
    "tests/unit/test_t0126_differential.py",
    "tests/unit/_tool_turn_support.py",
)

#: The tip end, closed before merge (T-0094): d8979db, the branch's last commit touching non-test files (the docs/03
#: Q13 delta; the implementation is 4b93511). Both ends are literals, so the working tree is never read. ``None``
#: would mean "open: compare with the working tree"; it is kept only as the type.
T0107E_BASE = BASE_SHA
T0107E_TIP: str | None = "d8979dbb9a519f05849be5568a861200c7faa092"


def _hash_working(path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "hash-object", "--", path],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _tip_listing(paths: tuple[str, ...]) -> dict[str, str]:
    if T0107E_TIP is not None:
        gr.require_commits(T0107E_TIP)
        assert gr.is_ancestor(T0107E_BASE, T0107E_TIP), (
            f"{T0107E_BASE} is not an ancestor of {T0107E_TIP}"
        )
        return _ls_tree(T0107E_TIP, paths)
    listed = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *paths,
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.split()
    return {p: _hash_working(p) for p in listed if (REPO / p).is_file()}


def _diff(paths: tuple[str, ...]) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """(changed, added, removed, the BASE listing) of the files under ``paths`` between BASE and the tip end."""
    gr.require_commits(T0107E_BASE)
    old = _ls_tree(T0107E_BASE, paths)
    now = _tip_listing(paths)
    added, removed = sorted(now.keys() - old.keys()), sorted(old.keys() - now.keys())
    changed = sorted(p for p in old.keys() & now.keys() if old[p] != now[p])
    return changed, added, removed, old


def test_frozen_paths_equal_9275442() -> None:
    """Criterion 10, first item (and criterion 6's "no new MachineState field, command, event, timer or
    ToolCancelReason"; "no HUD element is added"): the machine, the contracts, the prompt builder, both executors, the
    CLI, the config, ``tools_v1.json`` and every other ``toolcall`` module, the HUD, the fakes, ``scripts/``, docs/01
    and ``docs/cd/`` are byte-identical (git blob hashes) at BASE and the tip end; the only file that may be added under
    them is ``toolcall/user_declines.py``, and nothing is removed."""
    changed, added, removed, old = _diff(FROZEN)
    assert set(FROZEN_FILES) <= set(old), (
        f"premise: 9275442 has the frozen files; missing {sorted(set(FROZEN_FILES) - set(old))}"
    )
    for d in ("hud/", "tests/fakes/", "scripts/", "docs/cd/", "local_astra/contracts/"):
        assert any(p.startswith(d) for p in old), f"premise: 9275442 has files under {d}"
    assert NEW_MODULE not in old, f"premise: {NEW_MODULE} is new"
    extra = [p for p in added if p != NEW_MODULE]
    assert not (changed or extra or removed), (
        f"T-0107e criterion 10: frozen paths differ from {T0107E_BASE[:7]} (tip {T0107E_TIP or 'working tree'}): "
        f"changed {changed}, added {extra}, removed {removed}"
    )


def test_existing_modules_unedited_since_9275442() -> None:
    """Criterion 10, third item: the session tool-turn, tool-loop, loop and ask-first modules, every T-0107a/T-0107d
    module (``test_ask_first.py``, ``test_session_ask_first.py``, ``test_t0107a_differential.py``,
    ``test_t0107a_docs.py``), both goldens and their fixtures, the T-0117/T-0121/T-0124/T-0126 differentials and the
    shared rig are byte-identical at BASE and the tip end."""
    changed, _added, removed, old = _diff(UNEDITED_TESTS)
    assert set(UNEDITED_TESTS) == set(old), (
        f"premise: 9275442 has every listed module; missing {sorted(set(UNEDITED_TESTS) - set(old))}"
    )
    assert not (changed or removed), (
        f"T-0107e criterion 10: existing test modules edited since {T0107E_BASE[:7]}: changed {changed}, "
        f"removed {removed}"
    )


def test_live_is_still_refused(tmp_path: Path) -> None:
    """Criterion 10, fourth item: ``tools = "live"`` was still refused at config load at T-0107e's TIP, naming
    docs/01 §3. T-0111g re-scope R5: a historical claim, re-pinned against ``git show
    <T0107E_TIP>:local_astra/config.py`` loaded in isolation (``live`` is a mode from T-0111g on)."""
    from tests.unit.test_t0111c_unchanged import config_at

    assert T0107E_TIP is not None
    load = config_at(T0107E_TIP, tmp_path).load
    p = tmp_path / "astra.toml"
    p.write_text('profile = "lean"\n[profiles.lean]\ntools = "live"\n')
    with pytest.raises(ValueError) as excinfo:
        load(p)
    assert "docs/01 §3" in str(excinfo.value), str(excinfo.value)
