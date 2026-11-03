"""T-0131 A5 and C3 (negative): with ``visual_tokens`` unset and ``reground`` off, sessions match the literal BASE
f30365d, and the frozen paths are byte-identical over a closed range (docs/tasks/T-0131.md).

Green at BASE and must stay green (no ``pending`` marker): until the implementation lands the working tree *is*
BASE's code, and the byte-identity rows compare two fixed commits.

- **The differential** is T-0107a's and T-0111g's harness, imported, not copied: the base run is a subprocess over
  ``git archive f30365d`` (T-0114's ``_base_tree`` with its ``BASE_SHA`` monkeypatched; ``local_astra`` first on
  ``sys.path``, the origin printed and checked), once per module. ``off``/``dry_run``: T-0111g's ``_capture``
  (T-0107a's tool turns and loop turns, one tools-off turn, the three ``--dry-run`` tools lines). ``live``: one
  session over ``live_rig`` (a click, a scroll, a report; then a toolless turn) and its ``--dry-run`` line. Sections:
  machine steps, HUD messages, prompts, request bodies, session-log records, ``turns.jsonl`` lines and history.
- **"live matches except for the C sentences"**: ``cli`` (not the rig) sets the guidance, so the in-process live run
  installs it the way the task specifies (``PromptBuilder(..., tool_guidance=LIVE_TOOL_GUIDANCE)`` in place of the
  session's builder) whenever the constant exists; every occurrence must be the tail of a system message, and with
  each occurrence and its whitespace separator removed the sections equal the base's. Before the constant exists
  the two runs are compared as they are.
- **Byte identity** (A5's list, plus C3's pinned test modules): ``BASE_SHA..TIP_IMPL``, two literals, never the
  working tree. **Pre-merge conversion (2026-10-22, docs/tasks/T-0131.md Rulings):** ``TIP_IMPL`` is fixed at
  047f82b, the last implementation commit (the merge of group D's review fixes, d9a0fbc; every non-test change of
  the branch is in its tree), so these rows compare ``f30365d..047f82b`` and read nothing from the working tree;
  main's later edits to these paths cannot redden them. At the merge it may close to ``M^2`` (``gr.prove_range``),
  as T-0111g's did. The power row reuses T-0111g's ``82da380..02ccff5`` (it changes ``services/versions.lock``).

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / ``dry_run`` / ``live``, loop as
each scenario says, n = 1, deterministic.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb
from local_astra.toolcall import tools
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_differential as t0107a
from tests.unit import test_t0111g_differential as t0111g
from tests.unit._live_browser_support import FakeLiveBrowser, live_frame, live_rig
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_t0117_differential import _PolicyMachine
from tests.unit.test_t0121_differential import _assert_equal, _with_history

#: T-0131's BASE (docs/tasks/T-0131.md, "Branch ... from main at f30365d"). A literal, never a moving ref.
BASE_SHA = "f30365d9fae12dbf4f42b7e22d5a6e099b7303c4"
#: The branch's last implementation commit, 047f82b, fixed at the pre-merge conversion (see the docstring).
TIP_IMPL = "047f82bb283cae10c7a019286e331f4687bb0e32"
TOOLS_SHA = t0111g.TOOLS_SHA
Q = '<|"|>'
SCROLL = f"<|tool_call>call:scroll{{direction:{Q}down{Q}}}<tool_call|>"

#: A5's list, written out from the criterion
FROZEN = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/tool_executor.py",
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/toolcall/tools_v1.json",
    "services/vllm.sh",
    "services/versions.lock",
    "local_astra/computer",
    "local_astra/pages",
    "hud",
)
#: C3: the pins that stay verbatim
C3_MODULES = (
    "tests/cd/unit/test_native_tool_instructions_parity.py",
    "tests/unit/test_speaking_policy_doc.py",
    "tests/unit/test_t0107a_docs.py",
)


# --------------------------------------------------------------------------------------------- the live scenario


def _apply_guidance(rig: Any) -> None:
    """Install the guidance the way ``cli`` gives it to a live session (the task's builder keyword), when the
    constant exists; before it exists, nothing (the base run never has it)."""
    g = getattr(pb, "LIVE_TOOL_GUIDANCE", None)
    if g is None:
        return
    old = rig.session.prompts
    rig.session.prompts = pb.PromptBuilder(
        old.system_prompt, old.budget, old.thinking, tools=old.tools, tool_guidance=g
    )


async def _live_turns(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    browser = FakeLiveBrowser(
        [live_frame("onset")], post_frames=[live_frame("after-1"), live_frame("after-2")]
    )
    async with live_rig(
        tmp, browser, script=[answer(tsl.S1), tool(VALID_CLICK)], machine_cls=_PolicyMachine
    ) as rig:
        _apply_guidance(rig)
        rig.llm.script_steps([answer(tsl.S2), tool(SCROLL)], [answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 2, turn="t1")
        await t0107a._until_turns(rig, 1, "the live loop turn")
        rig.llm.script = [answer("It is a small shop. ")]
        await rig.ask("what is on the page")
        await t0107a._until_turns(rig, 2, "the live toolless turn")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path)


def _live_dry_run_line() -> str:
    from local_astra import cli
    from tests.unit import _tools_switch_support as T

    return cli.tools_line(T.make_kit(), 8, cli.LiveLine(browser="headless", viewport=(1280, 800)))


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the f30365d tree."""
    return {
        "t0111g": await t0111g._capture(tmp / "t0111g"),
        "live": await _live_turns(tmp / "live"),
        "live_dry_run_line": _live_dry_run_line(),
    }


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0131_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0131_base")
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


def _same(expected: Any, got: Any, what: str) -> None:
    _assert_equal(
        expected, got, f"T-0131 A5 {what} (base {BASE_SHA[:7]}; prefix is T-0121's helper):"
    )


async def test_off_dry_run_equal_f30365d(base: dict[str, Any], tmp_path: Path) -> None:
    """A5 (mutations 2, 3): T-0107a's tool and loop turns (``dry_run``) and one tools-off turn give the same machine
    steps, HUD messages, prompts, request bodies, session-log records, ``turns.jsonl`` lines and history as
    f30365d; the ``--dry-run`` tools lines of off and dry_run (loop on and off) are the same strings."""
    now = observed(await t0111g._capture(tmp_path))  # T-0155 O-8: the persona sentence only
    assert now["dry_run_lines"] == base["t0111g"]["dry_run_lines"]
    for scenario in ("tool_turns", "loop_turns", "tools_off"):
        for section in t0107a.SECTIONS:
            _same(
                base["t0111g"][scenario][section], now[scenario][section], f"{scenario} {section!r}"
            )
    assert [h["user_text"] for h in now["tools_off"]["history"]] == ["what is on the screen"], (
        "premise"
    )


def _strip(tree: Any, g: str) -> Any:
    """``tree`` with every occurrence of ``g`` (plain or JSON-escaped) and the whitespace before it removed."""
    esc = json.dumps(g)[1:-1]
    plain = re.compile(r"\s*" + re.escape(g))
    escaped = re.compile(r"(?:\\n|\\t|\s)*" + re.escape(esc))
    if isinstance(tree, str):
        return escaped.sub("", plain.sub("", tree))
    if isinstance(tree, list):
        return [_strip(x, g) for x in tree]
    if isinstance(tree, dict):
        return {k: _strip(v, g) for k, v in tree.items()}
    return tree


def _system_texts(sections: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for r in sections["requests"]:
        for m in r["messages"]:
            if m["role"] == "system":
                content = m["content"]
                out.append(
                    content
                    if isinstance(content, str)
                    else "".join(p.get("text", "") for p in content)
                )
    return out


async def test_live_equal_f30365d_but_guidance(base: dict[str, Any], tmp_path: Path) -> None:
    """A5 (``live``) and C1/C4: a live loop turn (click, scroll, report) and a toolless live turn give f30365d's
    sections exactly, except that, once ``LIVE_TOOL_GUIDANCE`` exists, every system message ends with it; with
    those occurrences removed everything is equal, so the step payloads stay ``{action, status, screenshot}`` (C4)
    and no request body gains a key (A2 unset). The live ``--dry-run`` line is f30365d's."""
    now = observed(await _live_turns(tmp_path / "live"))  # T-0155 O-8: the persona sentence only
    assert _live_dry_run_line() == base["live_dry_run_line"]
    g = getattr(pb, "LIVE_TOOL_GUIDANCE", None)
    if g is not None:
        systems = _system_texts(now)
        assert systems and all(s.endswith(g) and s.count(g) == 1 for s in systems), (
            "T-0131 C1: every live system message ends with the guidance, once"
        )
        dumped = json.dumps(now)
        assert dumped.count(json.dumps(g)[1:-1]) >= len(systems), (
            "premise: the guidance is in the run"
        )
        now = _strip(now, g)
    for section in t0107a.SECTIONS:
        _same(base["live"][section], now[section], f"live {section!r}")
    steps = [
        r
        for r in now["log"]
        if r["kind"] == "llm.request" and r["data"].get("phase") == "tool_step"
    ]
    assert len(steps) == 2, f"premise: two step requests in the live loop turn; got {len(steps)}"
    assert [h["user_text"] for h in now["history"]] == [tsl.QUESTION, "what is on the page"]
    assert now["records"][-1]["extras"].get("frame_source") == "window", (
        "premise: the turns ran live"
    )


# --------------------------------------------------------------------------------------------- byte identity


def _prove_open_range() -> None:
    gr.require_commits(BASE_SHA, TIP_IMPL)
    assert gr.is_ancestor(BASE_SHA, TIP_IMPL), (
        f"{BASE_SHA[:7]} is not an ancestor of {TIP_IMPL[:7]}"
    )


def test_frozen_paths_over_closed_range() -> None:
    """A5: the machine, the speech gate, the dry-run executor, the contracts, the fakes, ``tools_v1.json``,
    ``services/vllm.sh`` and ``versions.lock``, ``computer/``, ``pages/`` and the HUD are byte-identical over
    ``BASE_SHA..TIP_IMPL`` (never the working tree), with no file added or removed under them. Power: the same
    predicate sees ``82da380..02ccff5``'s lock change."""
    _prove_open_range()
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    added_or_removed = gr.changed(BASE_SHA, TIP_IMPL, *FROZEN, diff_filter="ADR")
    assert added_or_removed == [], (
        f"T-0131 A5: added or removed under frozen paths: {added_or_removed}"
    )
    changed = gr.changed(BASE_SHA, TIP_IMPL, *FROZEN)
    assert changed == [], (
        f"T-0131 A5: frozen paths differ in {BASE_SHA[:7]}..{TIP_IMPL[:7]}: {changed}"
    )
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert gr.changed(t0111g.P2_PARENT, t0111g.P2, *FROZEN) == ["services/versions.lock"], "power"


def test_c3_pins_unedited_over_closed_range() -> None:
    """C3: the native-instructions parity row, the speaking-policy doc pins and the T-0107a docs pins are
    byte-identical over ``BASE_SHA..TIP_IMPL``."""
    _prove_open_range()
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *C3_MODULES).stdout.split()
    assert sorted(listed) == sorted(C3_MODULES), f"premise: BASE has every module; {listed}"
    assert gr.changed(BASE_SHA, TIP_IMPL, *C3_MODULES) == []


def test_tools_sha_unchanged() -> None:
    """A5: the tools sha stays ``bbfc53cc…`` at BASE and at TIP_IMPL, and the loaded set's version is it."""
    _prove_open_range()
    for rev in (BASE_SHA, TIP_IMPL):
        blob = gr._git("show", f"{rev}:local_astra/toolcall/tools_v1.json")
        assert blob.returncode == 0, blob.stderr
        assert hashlib.sha256(blob.stdout.encode()).hexdigest() == TOOLS_SHA, rev
    assert tools.load().version == TOOLS_SHA
