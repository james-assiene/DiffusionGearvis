"""T-0111g B17 (negative): ``off`` and ``dry_run`` sessions unchanged against the literal BASE 9bebf69, and the frozen
paths byte-identical (docs/tasks/T-0111g.md).

Green at BASE and must stay green (no ``pending`` marker): until the implementation lands the working tree *is*
BASE's code.

- **The differential** is T-0107a's harness, imported, not copied: the base run is a subprocess over ``git archive
  9bebf69`` (T-0114's ``_base_tree`` with its ``BASE_SHA`` monkeypatched for the call; ``local_astra`` first on
  ``sys.path``, the origin printed and checked), once per module. Scenarios: T-0107a's tool turns (loop off) and loop
  turns (loop on), plus one tools-off turn; sections: machine steps, HUD messages, prompts, request bodies, session-log
  records, ``turns.jsonl`` lines and history (T-0121's ``_with_history``/``_assert_equal``). The ``--dry-run`` lines
  are compared too: ``cli.tools_line`` for off, dry_run with the loop, dry_run without it.
- **Byte identity** (B17's list), T-0111g pre-merge conversion (.claude/rules/tests.md "pin both ends", T-0094):
  the row compares two fixed commits, ``BASE_SHA..TIP_IMPL`` = ``9bebf69..e49b868``, and never reads the working
  tree. Before this conversion it compared BASE with the working tree; main's T-0128 merge (fbb97b5) edited
  ``hud/**``, so that open range would have gone red the moment T-0111g merged into main, for a reason that is not
  this task's (as happened to ``test_t0111c_unchanged.py`` when T-0111c merged). ``TIP_IMPL`` = e49b868 is the last
  implementation commit on ``T-0111g-live-wiring``; every commit after it on the branch touches tests or docs only
  (this conversion is the first, and its own sha cannot be named inside it). **After the merge** the orchestrator
  closes the row to the merge M, as T-0111c-close-guard did: ``TIP = M^2``, ``BASE = merge-base M^1 M^2`` and
  ``gr.prove_range``, plus ``gr.changed(TIP_IMPL, M^2, *FROZEN) == []`` for the tests/docs-only tail. The power row
  runs the same predicate over ``82da380..02ccff5``, known to change ``services/versions.lock``.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / ``dry_run``, loop off or on as
each scenario says, n = 1, deterministic.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from local_astra.toolcall import tools
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_differential as t0107a
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins
from tests.unit._tool_turn_support import answer, tool_rig
from tests.unit.test_t0117_differential import _PolicyMachine
from tests.unit.test_t0121_differential import _assert_equal, _with_history

#: T-0111g's BASE (docs/tasks/T-0111g.md, "Branch ... from main at 9bebf69"). A literal, never a moving ref.
BASE_SHA = "9bebf69d9d3ac9aacb5a41bf8bd2f82569afc81a"
#: T-0111g's last implementation commit on ``T-0111g-live-wiring`` (pre-merge conversion). A literal; the frozen-paths
#: row compares ``BASE_SHA..TIP_IMPL`` and never the working tree.
TIP_IMPL = "e49b868c583f6597d34c5c2c4584362327824dbe"
TOOLS_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
#: the power row's range: the human's P2 commit changed ``services/versions.lock`` (T-0111c)
P2_PARENT, P2 = "82da380", "02ccff5"

FROZEN = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/conductor/tool_executor.py",
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/toolcall/tools_v1.json",
    "services/versions.lock",
    "hud",
)


async def _tools_off(tmp: Path) -> dict[str, Any]:
    """One typed tools-off turn."""
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, kit=None, machine_cls=_PolicyMachine) as rig:
        rig.llm.script = [answer("It is a quiet page. ")]
        rig.submit(E.UserText(t_ms=rig.clock.now_ms, text="what is on the screen"))
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= 1,
            "T-0111g B17: the tools-off turn never ended",
        )
        await rig.advance(300)
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path)


def _dry_run_lines() -> list[str]:
    from local_astra import cli
    from tests.unit import _tools_switch_support as T

    kit = T.make_kit()
    return [cli.tools_line(None), cli.tools_line(kit, 8), cli.tools_line(kit, None)]


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the 9bebf69 tree."""
    return {
        "tool_turns": await t0107a._tool_turns(tmp / "tool_turns"),
        "loop_turns": await t0107a._loop_turns(tmp / "loop_turns"),
        "tools_off": await _tools_off(tmp / "tools_off"),
        "dry_run_lines": _dry_run_lines(),
    }


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0111g_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0111g_base")
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
        f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE_SHA} tree"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


async def test_off_and_dry_run_equal_9bebf69(base: dict[str, Any], tmp_path: Path) -> None:
    """B17 (mutation 17): T-0107a's tool turns and loop turns and one tools-off turn give the same machine steps, HUD
    messages, prompts, request bodies, session-log records, ``turns.jsonl`` lines and history as 9bebf69; the
    ``--dry-run`` tools lines are the same strings."""
    now = observed(await _capture(tmp_path))  # T-0155 O-8: the persona sentence only
    assert now["dry_run_lines"] == base["dry_run_lines"], (
        now["dry_run_lines"],
        base["dry_run_lines"],
    )
    for scenario in ("tool_turns", "loop_turns", "tools_off"):
        for section in t0107a.SECTIONS:
            _assert_equal(
                base[scenario][section],
                now[scenario][section],
                f"T-0111g B17 {scenario} {section!r} (base {BASE_SHA[:7]}; prefix is T-0121's helper):",
            )
    # premises, by hand: the scenarios ran the paths they name
    assert [h["user_text"] for h in now["tools_off"]["history"]] == ["what is on the screen"], (
        "premise: the tools-off turn ran"
    )
    steps = [
        r
        for r in now["loop_turns"]["log"]
        if r["kind"] == "llm.request" and r["data"].get("phase") == "tool_step"
    ]
    assert len(steps) == 5, f"premise: 5 step requests over the three loop turns; got {len(steps)}"


def test_frozen_paths_equal_base_over_closed_range() -> None:
    """B17: the machine, the speech gate, the prompt builder, the dry-run executor, the contracts, the fakes,
    ``tools_v1.json``, ``services/versions.lock`` and the HUD are byte-identical over the fixed range
    ``BASE_SHA..TIP_IMPL`` (9bebf69..e49b868; T-0111g pre-merge conversion, the working tree is never read), with no
    file added or removed under them. The power row: the same predicate sees the lock change of
    ``82da380..02ccff5``."""
    gr.require_commits(BASE_SHA, TIP_IMPL, P2_PARENT, P2)
    assert gr.is_ancestor(BASE_SHA, TIP_IMPL), (
        f"premise: {BASE_SHA[:7]} is an ancestor of {TIP_IMPL[:7]}"
    )
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    added_or_removed = gr.changed(BASE_SHA, TIP_IMPL, *FROZEN, diff_filter="ADR")
    assert added_or_removed == [], (
        f"T-0111g B17: files added or removed under the frozen paths in "
        f"{BASE_SHA[:7]}..{TIP_IMPL[:7]}: {added_or_removed}"
    )
    changed = gr.changed(BASE_SHA, TIP_IMPL, *FROZEN)
    assert changed == [], (
        f"T-0111g B17: frozen paths differ in {BASE_SHA[:7]}..{TIP_IMPL[:7]}: {changed}"
    )
    assert gr.changed(P2_PARENT, P2, *FROZEN) == ["services/versions.lock"], (
        "power: the predicate can see"
    )


def test_tools_sha_unchanged() -> None:
    """B17: the tools sha stays ``bbfc53cc…``: BASE's blob hashes to the literal, and so does the loaded set."""
    gr.require_commits(BASE_SHA)
    blob = gr._git("show", f"{BASE_SHA}:local_astra/toolcall/tools_v1.json")
    assert blob.returncode == 0, blob.stderr
    assert hashlib.sha256(blob.stdout.encode()).hexdigest() == TOOLS_SHA
    assert tools.load().version == TOOLS_SHA
