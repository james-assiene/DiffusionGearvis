"""T-0135 group A2, A9 and A10 (negative): with ``recover_thought_calls`` not passed, sessions equal the literal BASE
8887909 (``Merge T-0134``) in every mode, the D-5 shape included; the frozen paths of A3 and A9 are byte-identical over
a closed range; docs/02 gains exactly one §5 paragraph (docs/tasks/T-0135.md).

The differential and byte-identity rows are green at BASE and must stay green (no ``pending`` marker): until the
implementation lands the working tree *is* BASE's code, and the byte-identity rows compare two fixed commits.
``test_doc02_delta_t0135`` is red until the A10 paragraph is in ``TIP_IMPL``'s docs/02.

- **The differential** is T-0131's harness, imported, not copied: the base run is a subprocess over ``git archive
  8887909`` (T-0114's ``_base_tree`` with its ``BASE_SHA`` monkeypatched; ``local_astra`` first on ``sys.path``, the
  origin printed and checked), once per module. Sections: T-0111g's ``_capture`` (off and ``dry_run``: tool turns,
  loop turns, a tools-off turn), T-0131's live scenario, T-0132's live ReGround turn, and this task's D-5 turns
  (``_d5_live``: the T-0132 step-2 open thought ending in a ``type_at``; a step-1 open thought ending in a click; a
  closed thought quoting a call beside an answer-channel call; an open thought ending in a call next to an
  answer-channel call; a doubled close; prose after the close; ``_d5_dry_run``: the step-1 and step-2 shapes under
  ``dry_run``). No new keyword is passed anywhere. Reviewer mutation 7 (recovery with the keyword ``None``) turns the
  D-5 sections red.
- **Byte identity** (A3's and A9's lists, plus the files the task says it never touches): ``BASE_SHA..TIP_IMPL``,
  two literals, never the working tree. ``TIP_IMPL`` starts at BASE (no implementation yet), so the rows hold
  trivially until it lands. **TIP moves**: whoever commits implementation bumps it, and at the merge it closes to the
  last implementation commit.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / ``dry_run`` / ``live``, loop on
(8 steps) where a scenario says so, repair on (step 1 only), n = 1, deterministic.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_differential as t0107a
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0111g_differential as t0111g
from tests.unit import test_t0131_differential as t0131
from tests.unit import test_t0132_differential as t0132
from tests.unit._live_browser_support import FakeLiveBrowser, live_frame, live_rig
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins
from tests.unit._tool_turn_support import VALID_CLICK, answer, thought, tool, tool_rig
from tests.unit.test_t0117_differential import _PolicyMachine
from tests.unit.test_t0121_differential import _assert_equal, _with_history

#: T-0135 group A2's BASE: main at the ``Merge T-0134`` commit (docs/tasks/T-0135.md, "Sequencing"). A literal.
BASE_SHA = "8887909d58d92e8df3f88f952b79c2f2f2b2b1e8"
#: The A2 implementation tip: eb2dfc2 (the A10 docs/02 paragraph; the code is 16eb3c6, its parent). Was BASE.
TIP_IMPL = "eb2dfc21b93efef8a1b08857ea81e8c7ca4a99a6"
#: docs/02's own base for A10: e59aa69, the last main commit merged into the branch. It adds the T-0134 §5 paragraph
#: after 8887909 (``git diff --stat 8887909 e59aa69 -- docs/02-architecture.md``: 2 insertions), so "the rest of
#: docs/02 is unchanged" is measured against it, never against BASE_SHA. The code ranges keep BASE_SHA.
DOC_BASE = "e59aa695cff74b4f49b05016adafc0e770a5d2e1"
HEADER = "**Spec delta (T-0135"

#: A3's list, written out from the criterion
FROZEN_A3 = (
    "local_astra/toolcall/live.py",
    "local_astra/toolcall/native.py",
    "local_astra/toolcall/router.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/contracts",
    "tests/fakes",
)
#: A9's list, written out from the criterion
FROZEN_A9 = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/tool_executor.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/reground.py",
    "local_astra/computer",
    "local_astra/pages",
    "hud",
    "services",
    "local_astra/cli.py",
    "local_astra/config.py",
)
#: "Files T-0135 never touches" and "This task does not own" beyond A3/A9 (Sequencing, Owned files)
NOT_OWNED = (
    "local_astra/toolcall/loop.py",
    "tests/unit/_live_browser_support.py",
    "pyproject.toml",
    "uv.lock",
)

Q = '<|"|>'
TYPE_D5 = f"<|tool_call>call:type_at{{press_enter:true,text:{Q}ada@example.com{Q},x:305,y:448}}<tool_call|>"
SENT = "ZQX-T0135-DIFF"
DOUBLED = VALID_CLICK + "<tool_call|>"


def _open(text: str) -> Any:
    return thought(f"{SENT} {text}", open_=True)


#: (question, step-1 script, step scripts): every D-5 shape the rule names, with and without an answer-channel call
D5_TURNS: tuple[tuple[str, list[Any], tuple[list[Any], ...]], ...] = (
    ("fill in the form", [answer(tsl.S1), tool(VALID_CLICK)], ([_open(f"plan. {TYPE_D5}")],)),
    ("click it", [_open(f"plan. {VALID_CLICK}")], ()),
    (
        "and the next one",
        [thought(f"{SENT} I could {tsl.CLICK3}"), answer(tsl.S2), tool(tsl.CLICK2)],
        ([_open(f"maybe {tsl.CLICK3}"), tool(tsl.CLICK3)], [answer(tsl.S3)]),
    ),
    ("press it twice", [_open(f"plan. {DOUBLED}")], ()),
    ("type the address", [_open(f"plan. {TYPE_D5} Actually, I should scroll.")], ()),
)


async def _d5_live(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    browser = FakeLiveBrowser(
        [live_frame("onset")], post_frames=[live_frame(f"after-{i}") for i in range(1, 6)]
    )
    async with live_rig(tmp, browser, machine_cls=_PolicyMachine) as rig:
        for i, (question, script, steps) in enumerate(D5_TURNS):
            rig.llm.script = list(script)
            rig.llm.script_steps(*steps)
            await rig.ask(question)
            await t0107a._until_turns(rig, i + 1, f"the D-5 live turn {i + 1}")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    out = _with_history(rig, tmp, session_id, log_path)
    out["executed"] = [[i.call_id, dict(i.arguments)] for i in browser.invocations]
    return out


async def _d5_dry_run(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, policy=tsl.LOOP, machine_cls=_PolicyMachine) as rig:
        for i, (question, script, steps) in enumerate(D5_TURNS[:2]):
            rig.llm.script = list(script)
            rig.llm.script_steps(*steps)
            await rig.ask(question)
            await t0107a._until_turns(rig, i + 1, f"the D-5 dry_run turn {i + 1}")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    out = _with_history(rig, tmp, session_id, log_path)
    out["executed"] = [[i.call_id, dict(i.arguments)] for i in rig.executor.invocations]
    return out


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the 8887909 tree."""
    return {
        "t0111g": await t0111g._capture(tmp / "t0111g"),
        "live": await t0131._live_turns(tmp / "live"),
        "live_reground": await t0132._live_reground(tmp / "live_reground"),
        "d5_live": await _d5_live(tmp / "d5_live"),
        "d5_dry_run": await _d5_dry_run(tmp / "d5_dry_run"),
    }


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0135_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0135_base")
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
    _assert_equal(expected, got, f"T-0135 A9 {what} (base {BASE_SHA[:7]}; T-0121's helper):")


async def test_none_equal_8887909(base: dict[str, Any], tmp_path: Path) -> None:
    """A9 (mutation 7): with the keyword not passed, off/``dry_run`` sessions (machine steps, HUD messages, prompts,
    request bodies, session-log records, turn lines, history), the live loop and toolless turns, a live ReGround
    turn, and every D-5 turn (live and ``dry_run``; what executed included) equal 8887909's; no turn line gains
    ``recover_thought_calls`` and no ``tool.call`` record gains ``from_thought``."""
    now = observed(await _capture(tmp_path))  # T-0155 O-8: the persona sentence only
    for scenario in ("tool_turns", "loop_turns", "tools_off"):
        for section in t0107a.SECTIONS:
            _same(
                base["t0111g"][scenario][section],
                now["t0111g"][scenario][section],
                f"{scenario} {section!r}",
            )
    for name in ("live", "d5_live", "d5_dry_run"):
        for section in t0107a.SECTIONS:
            _same(base[name][section], now[name][section], f"{name} {section!r}")
    for name in ("d5_live", "d5_dry_run"):
        _same(base[name]["executed"], now[name]["executed"], f"{name} executed calls")
    for section in ("log", "records", "bodies", "pass2"):
        _same(
            base["live_reground"][section],
            now["live_reground"][section],
            f"live reground {section!r}",
        )
    # premises, by hand: the D-5 turns ran the paths they name at BASE
    ends = [r["extras"].get("tool_loop_end") for r in base["d5_live"]["records"]]
    assert ends == ["final", "final", "final", "final", "final"], ends
    assert [c for c, _ in base["d5_live"]["executed"]] == ["t1-call", "t3-call", "t3-call-s2"], (
        f"premise: BASE executes only answer-channel calls; {base['d5_live']['executed']}"
    )
    assert [c for c, _ in base["d5_dry_run"]["executed"]] == ["t1-call"], base["d5_dry_run"]
    dumped = json.dumps(now)
    assert "recover_thought_calls" not in dumped and '"from_thought"' not in dumped, (
        "T-0135 A8/A6: without the keyword no line or record names the feature"
    )
    assert SENT in json.dumps(now["d5_live"]["log"]), "premise: the thoughts are logged as at BASE"


def _prove_open_range() -> None:
    gr.require_commits(BASE_SHA, TIP_IMPL)
    assert gr.is_ancestor(BASE_SHA, TIP_IMPL), (
        f"{BASE_SHA[:7]} is not an ancestor of {TIP_IMPL[:7]}"
    )


def test_frozen_paths_over_closed_range() -> None:
    """A3 and A9: the toolcall reader (``live.py``, ``native.py``, ``router.py``), ``llm_client.py``, the contracts,
    the fakes, the machine, the speech gate, the dry-run executor, the prompt builder (``LIVE_TOOL_GUIDANCE``),
    ``tools_v1.json``, ``reground.py``, ``computer/``, ``pages/``, the HUD, ``services/``, ``cli.py`` and
    ``config.py`` are byte-identical over ``BASE_SHA..TIP_IMPL`` (never the working tree), with no file added or
    removed under them. Power: the same predicate sees ``82da380..02ccff5``'s lock change."""
    _prove_open_range()
    frozen = FROZEN_A3 + FROZEN_A9
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *frozen).stdout.split()
    for p in frozen:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    added_or_removed = gr.changed(BASE_SHA, TIP_IMPL, *frozen, diff_filter="ADR")
    assert added_or_removed == [], (
        f"T-0135 A3/A9: added or removed under frozen paths: {added_or_removed}"
    )
    changed = gr.changed(BASE_SHA, TIP_IMPL, *frozen)
    assert changed == [], (
        f"T-0135 A3/A9: frozen paths differ in {BASE_SHA[:7]}..{TIP_IMPL[:7]}: {changed}"
    )
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert gr.changed(t0111g.P2_PARENT, t0111g.P2, *frozen) == ["services/versions.lock"], "power"


def test_not_owned_unchanged_over_closed_range() -> None:
    """Owned files ("This task does not own"; Sequencing): ``toolcall/loop.py``, ``_live_browser_support.py``,
    ``pyproject.toml``, ``uv.lock`` and every ``scripts/gpu/t01*`` harness that exists at BASE are byte-identical
    over ``BASE_SHA..TIP_IMPL``."""
    _prove_open_range()
    gpu = [
        p
        for p in gr._git(
            "ls-tree", "-r", "--name-only", BASE_SHA, "--", "scripts/gpu"
        ).stdout.split()
        if re.match(r"scripts/gpu/t01", p)
    ]
    assert len(gpu) >= 3, f"premise: earlier tasks' harnesses exist at BASE; {gpu}"
    paths = (*NOT_OWNED, *gpu)
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *paths).stdout.split()
    assert sorted(listed) == sorted(paths), f"premise: every path exists at BASE; {listed}"
    changed = gr.changed(BASE_SHA, TIP_IMPL, *paths)
    assert changed == [], (
        f"T-0135 (not owned): changed in {BASE_SHA[:7]}..{TIP_IMPL[:7]}: {changed}"
    )


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_doc02_delta_t0135() -> None:
    """A10: docs/02 §5 gains exactly one paragraph starting ``**Spec delta (T-0135``, recording
    ``recover_thought_calls``, ``from_thought``, ``thought_open``, that the T-0088 split stands, that recovered calls
    are never repaired, and that ReGround pass 2 is excluded; removing it once leaves docs/02 equal to DOC_BASE
    e59aa69's paragraph for paragraph (the last main commit merged into the branch; BASE 8887909 predates main's T-0134
    paragraph, so no tip could match it). A second T-0135 paragraph fails the count; any other edited, added or removed
    paragraph fails the equality. Keywords, never wording. Every side is a fixed revision (DOC_BASE and
    ``TIP_IMPL``), never the working tree."""
    gr.require_commits(DOC_BASE, TIP_IMPL)
    assert gr.is_ancestor(DOC_BASE, TIP_IMPL), (
        f"{DOC_BASE[:7]} is not an ancestor of {TIP_IMPL[:7]}"
    )
    base_doc = _doc02_at(DOC_BASE)
    assert [p for p in t7d.paragraphs(base_doc) if p.lstrip().startswith(HEADER)] == [], (
        "premise: DOC_BASE has no T-0135 paragraph"
    )
    doc = _doc02_at(TIP_IMPL)
    s5 = t7d.section(doc, "## 5. ")
    paras = [p for p in t7d.paragraphs(s5) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, (
        f"T-0135 A10: one §5 paragraph starting {HEADER!r} in {TIP_IMPL[:7]}; found {len(paras)}"
    )
    para = paras[0]
    low = para.lower()
    need = {
        "recover_thought_calls": "recover_thought_calls" in para,
        "from_thought": "from_thought" in para,
        "thought_open": "thought_open" in para,
        "T-0088": "T-0088" in para,
        "repair": re.search(r"\brepair", low) is not None,
        "pass 2": "pass 2" in low or "pass-2" in low,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0135 A10: the docs/02 delta does not record {missing}"
    rest = t7d.paragraphs(doc)
    rest.remove(para)
    assert rest == t7d.paragraphs(base_doc), (
        "T-0135 A10: docs/02 changes only by the one delta paragraph"
    )
