"""T-0117 criteria 9, 10 and 11 (negative): with no thought marker nothing changes, against the literal base c7e04f0.

A separate module because the base run is a subprocess over ``git archive c7e04f0`` (its ``local_astra`` first on
``sys.path``, origin printed and checked), so under ``--dist loadfile`` it is a worker of its own. The tree is built
by T-0114's ``_base_tree`` with its base SHA set to c7e04f0 for the call (imported, not copied); the harness is the
tools-off golden's ``capture_sections`` and ``normalise`` plus T-0114's whole-log normalisation (pid, byte count,
git commit).

Every row here holds at the base and must stay green; none is pending.

Readings (listed in the test-author's report):

- criterion 10's file list is compared by git blob hash over T-0117's closed merge range (``git ls-tree`` at
  BASE = merge-base of ebff09e's parents and at TIP = ebff09e^2; T-0110b ruling, T-0094); it also covers ``llm_client.py`` and ``special_tokens.py``, which the task lists
  as not owned (``test_answer_cleaner_thoughts`` rebuilds the base cleaner over today's ``drop_special_literals``,
  so that file must be the base's). ``toolcall/loop.py`` is T-0120's and is not pinned here;
- "``thought_open=True`` commits still bypass the cleaner unchanged" is read at the machine: such a commit reaches it
  with its text and flag as the backend sent them;
- "``step_text`` is still ``router.route(cleaner.raw).spoken_prefix``" is read in the step-3 request of a three-step
  turn whose withheld step 2 carries a closed thought: the embedded text is that (hand-written) value.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``dry_run``, n = 1,
deterministic; Hypothesis bounded (``max_examples`` 200).
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.conductor.turn_machine import TurnPolicy
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.turn import TurnState
from local_astra.toolcall import router
from tests import _git_range as gr
from tests.fakes.search import FakeSearch
from tests.unit import test_session_tool_loop as tsl
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins
from tests.unit._tool_turn_support import (
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    VALID_CLICK,
    answer,
    tool,
    tool_rig,
)
from tests.unit.test_session_loop import POLICY, RecordingMachine
from tests.unit.test_session_tools_off_golden import StateMachine, capture_sections, normalise

REPO = Path(__file__).resolve().parents[2]
#: T-0117's base (docs/tasks/T-0117.md, "Branch"). A literal, never a moving ref (tests.md).
BASE_SHA = "c7e04f0"

LOOP = TurnPolicy(thinking=True, tool_loop=True)


def open_answer(text: str) -> Commit:
    return Commit(request_id="?", canvas_idx=0, segment="answer", text=text, thought_open=True)


# --- the criterion-9 scenarios -----------------------------------------------------------------------------

#: (question, script, repair continuation or None). Tools on, loop off, one session, in order.
LOOP_OFF_TURNS: tuple[tuple[str, tuple[Commit, ...], tuple[Commit, ...] | None], ...] = (
    ("and what does it say now", (answer("It says welcome back."),), None),
    ("click the sign in button", (answer(tsl.S1), tool(VALID_CLICK)), None),
    ("click the small one", (answer(tsl.S1), tool(REPAIRABLE_CLICK)), (tool(REPAIRED_CLICK),)),
    ("which key", (answer("Press <back> now. "),), None),
    ("which key again", (answer("Press <|tool"), answer("_call> now. ")), None),
    ("hold on", (answer("Hold <|tool"), open_answer("a draft"), answer("_call> on. ")), None),
    ("did you think so", (answer("I thought so, and I still do. "),), None),
    ("wait", (answer("Wait <|chan"),), None),
    ("are you done", (answer("Done <channel|>here. "),), None),
)


class _PolicyMachine(StateMachine):
    """The golden's full-state machine under ``tool_rig``'s policy argument (the golden's own is fixed to POLICY)."""

    def __init__(self, policy: Any = None) -> None:
        RecordingMachine.__init__(self, policy or POLICY)
        self.full = []


def _sections(rig: Any, tmp: Path, log_path: Path, session_id: str) -> dict[str, Any]:
    sections = capture_sections(rig, rig.log_records)
    per_process = {"log.started": {"pid": "<pid>"}, "log.stopped": {"bytes": "<bytes>"}}
    sections["log"] = [
        {"kind": r["kind"], "data": {**r["data"], **per_process.get(r["kind"], {})}}
        for r in rig.log_records
    ]
    out: dict[str, Any] = normalise(sections, session_id, tmp, log_path)
    for r in out["log"]:
        regime = r["data"].get("regime") if isinstance(r["data"], dict) else None
        if isinstance(regime, dict) and "git_commit" in regime:
            regime["git_commit"] = "<git>"
    return out


async def _loop_off(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, machine_cls=_PolicyMachine) as rig:
        for i, (question, script, continuation) in enumerate(LOOP_OFF_TURNS):
            rig.llm.script = list(script)
            if continuation is not None:
                rig.llm.script_continuation(*continuation)
            await rig.ask(question)
            await rig.until(
                lambda n=i + 1: rig.state.state is TurnState.IDLE and len(rig.lines()) == n,
                f"T-0117 criterion 9: loop-off turn {i + 1} ({question!r}) never ended",
            )
            await rig.advance(300)
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _sections(rig, tmp, log_path, session_id)


async def _loop_on(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(
        tmp,
        machine_cls=_PolicyMachine,
        policy=LOOP,
        search=FakeSearch(),
        script=[answer(tsl.PREFIX), tool(tsl.SEARCH_CALL)],
    ) as rig:
        rig.llm.script_steps([answer(tsl.REPORT)])
        await tsl.loop_turn(rig, 2)
        await rig.advance(300)
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _sections(rig, tmp, log_path, session_id)


#: Ruling F2 (reviewer A's mutation 11): tools off (``kit=None``), answers holding special literals split and
#: whole; a cleaner on this path would delete the literal (``Press  now.``) and change the records. One session, in
#: order. T-0121 (a named spec change, docs/tasks/T-0121.md "Owned files"): the third turn, ``Wait <|chan`` /
#: ``nel> there. ``, splices a channel open whose c7e04f0 result is the T-0121 bug (history holds the marker); it
#: left this comparison, and its expected behaviour (``Wait `` voiced and stored) is
#: ``test_session_thought_tools_off.test_t0117_tools_off_wait_row``. T-0124 re-scope R2 (docs/tasks/T-0124.md,
#: "Owned files"; a spec change to ruling F2): both remaining rows held a special literal, whose c7e04f0 result is
#: the T-0124 bug (history holds the literal and the post-stop text); their new behaviour is T-0124 criteria 1 and 4.
#: They are replaced by one no-literal turn whose c7e04f0 result is unchanged.
TOOLS_OFF_TURNS: tuple[tuple[str, tuple[Commit, ...]], ...] = (
    ("which key", (answer("Press <back> now. "),)),
)


async def _tools_off(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, kit=None, machine_cls=_PolicyMachine) as rig:
        for i, (question, script) in enumerate(TOOLS_OFF_TURNS):
            rig.llm.script = list(script)
            await rig.ask(question)
            await rig.until(
                lambda n=i + 1: rig.state.state is TurnState.IDLE and len(rig.lines()) == n,
                f"T-0117 ruling F2: tools-off turn {i + 1} ({question!r}) never ended",
            )
            await rig.advance(300)
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _sections(rig, tmp, log_path, session_id)


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario session. Runs in-process on the working tree and in a subprocess on the c7e04f0 tree."""
    return {
        "loop_off": await _loop_off(tmp / "off"),
        "loop_on": await _loop_on(tmp / "on"),
        "tools_off": await _tools_off(tmp / "tools_off"),
    }


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0117_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


async def test_no_marker_turns_equal_c7e04f0(tmp_path, monkeypatch):
    """Criterion 9 (negative; reviewer mutation 10): with no thought marker in any answer stream, tools on, the
    loop-off turns (toolless; prefix + valid call; repairable draft + repair; ``Press <back> now.``; a literal split
    across commits; a held tail across a ``thought_open=True`` commit; the word "thought"; a partial ``<|chan``; a
    stray ``<channel|>``) and a loop-on two-step search turn give the same machine steps, HUD messages, prompts,
    request bodies, session-log records and ``turns.jsonl`` lines as the session at c7e04f0."""
    gr.require_commits(BASE_SHA)
    monkeypatch.setattr(tsl, "BASE_SHA", BASE_SHA)
    base_dir = tsl._base_tree(tmp_path / "base_tree")
    out = tmp_path / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    base = json.loads(out.read_text())
    assert Path(base["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {base['origin']}, not from the {BASE_SHA} tree"
    )
    now = observed(await _capture(tmp_path / "now"))  # T-0155 O-8: the persona sentence only
    # T-0120b criterion 5 (pre-authorised re-scope): a loop-on turn line gains ``tool_compactions`` (0 here, nothing
    # is compacted) and nothing else; it is checked and removed from the line and its ``turn.record`` log copy.
    lines = [r["extras"] for r in now["loop_on"]["records"]] + [
        r["data"]["extras"] for r in now["loop_on"]["log"] if r["kind"] == "turn.record"
    ]
    assert lines, "premise: the loop-on scenario wrote a turn line"
    for extras in lines:
        assert extras.get("tool_compactions") == 0, (
            f"T-0120b criterion 5: the loop-on line carries tool_compactions 0; it holds {sorted(extras)}"
        )
        del extras["tool_compactions"]
    for scenario in ("loop_off", "loop_on", "tools_off"):
        for section in ("machine", "hud", "prompts", "requests", "log", "records"):
            expected, got = base["sections"][scenario][section], now[scenario][section]
            first = next(
                (i for i, (a, b) in enumerate(zip(expected, got, strict=False)) if a != b), None
            )
            assert got == expected, (
                f"T-0117 criterion 9: {scenario} {section!r} differs from {BASE_SHA} (lengths {len(expected)} "
                f"vs {len(got)}; first difference at {first}: "
                f"base {json.dumps(expected[first] if first is not None else None)[:800]} / "
                f"now {json.dumps(got[first] if first is not None else None)[:800]})"
            )
    # premise: the scenarios ran what they say (a repair happened, the loop ran two steps, nothing was noted)
    kinds = [r["kind"] for r in now["loop_off"]["log"]]
    assert kinds.count("llm.request") >= len(LOOP_OFF_TURNS), kinds.count("llm.request")
    statuses = [r["data"].get("status") for r in now["loop_off"]["log"] if r["kind"] == "tool.call"]
    assert "repairing" in statuses, statuses
    steps = [
        r
        for r in now["loop_on"]["log"]
        if r["kind"] == "llm.request" and r["data"].get("phase") == "tool_step"
    ]
    assert len(steps) == 1, steps
    # premise (ruling F2, re-scoped by T-0124 R2): the no-literal turn reached the tools-off machine as sent
    assert [
        s["event"]["text"]
        for s in now["tools_off"]["machine"]
        if s["event"].get("type") == "llm.commit" and not s["event"].get("thought_open")
    ] == ["Press <back> now. "], (
        "premise: `Press <back> now. ` reached the tools-off machine as sent"
    )


async def test_no_answer_cleaner_is_built_when_tools_are_off(tmp_path, monkeypatch):
    """Ruling F2 (reviewer A's mutation 11; criterion 10, first item): with ``kit=None`` no ``_AnswerCleaner`` is
    constructed, counted through a wrapper on the session module; with the default kit the same wrapper counts at
    least one (so the count is live)."""
    from local_astra.conductor import session as session_mod

    built: list[str] = []
    real = session_mod._AnswerCleaner

    class Counting(real):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            built.append("built")
            super().__init__()

    monkeypatch.setattr(session_mod, "_AnswerCleaner", Counting)
    script = [answer("Press <|tool_call> now. "), answer("Wait <|chan"), answer("nel> there. ")]
    async with tool_rig(tmp_path / "off", kit=None, script=list(script)) as rig:
        await rig.ask("which key")
        await rig.turn_done()
    assert built == [], f"T-0117 ruling F2: {len(built)} _AnswerCleaner built with tools off"
    async with tool_rig(tmp_path / "on", script=list(script)) as rig:
        await rig.ask("which key")
        await rig.turn_done()
    assert built, "premise: the wrapper counts the tools-on cleaner"


# =====================================================================================================
# Criterion 10: other paths unchanged
# =====================================================================================================

#: Criterion 10's files, plus the two the task lists as not owned that the unit rows rely on (module docstring).
UNOWNED = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/toolcall/special_tokens.py",
    "local_astra/contracts",
    "tests/fakes",
)


#: Criterion 10's scope row is closed over T-0117's own merge (tests.md "pin both ends", T-0094; T-0110b ruling
#: 2026-10-18): M = ebff09e, BASE = ``git merge-base M^1 M^2``, TIP = ``M^2``. The working tree is never read, so a
#: later task's legitimate edit of a fake (T-0110b's ``tests/fakes/audio.py``) cannot redden it.
T0117_MERGE = "ebff09e9c869f22636e92159e52cf34cd3914da1"
T0117_BASE = "4da5703e48ae89f58368e86eb653db8d5b3ebd89"
T0117_TIP = "76b1686f8796b6fbf9819a9b65e3ce28c8709d9c"
#: main when T-0110b closed the row (ea8d02f); ``prove_range``'s "later" premise (T-0117 merged after ``gr.LATER``).
T0117_LATER = "ea8d02fc9eb3dd68d6facae15c22954ad3e137bc"


def _ls_tree(rev: str, paths: tuple[str, ...]) -> dict[str, str]:
    """``{path: blob sha}`` of every file under ``paths`` at the commit ``rev`` (``git ls-tree -r``)."""
    listing = subprocess.run(
        ["git", "-C", str(REPO), "ls-tree", "-r", rev, "--", *paths],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    tree = {}
    for line in listing.splitlines():
        meta, path = line.split("\t", 1)
        tree[path] = meta.split()[2]
    return tree


def test_unowned_files_equal_c7e04f0():
    """Criterion 10, last item (reviewer mutation 11's neighbourhood): the machine, the gate, the client, the special
    token table, the contracts and the fakes are byte-identical (git blob hashes) across T-0117's own merge range
    ``T0117_BASE..T0117_TIP`` (c7e04f0 is an ancestor of that BASE and equal to it on these paths), with no file
    added or removed under the two directories."""
    gr.prove_range(T0117_BASE, T0117_TIP, T0117_MERGE, later=T0117_LATER)
    assert gr.is_ancestor(BASE_SHA, T0117_BASE), (
        f"premise: {BASE_SHA} is an ancestor of {T0117_BASE}"
    )
    base = _ls_tree(T0117_BASE, UNOWNED)
    assert set(UNOWNED[:4]) <= set(base), (
        f"premise: {T0117_BASE} has the four files; got {sorted(base)[:8]}"
    )
    assert base == _ls_tree(BASE_SHA, UNOWNED), (
        f"premise: {BASE_SHA} and {T0117_BASE} hold the same unowned files"
    )
    now = _ls_tree(T0117_TIP, UNOWNED)
    changed = sorted(p for p in base.keys() & now.keys() if base[p] != now[p])
    added, removed = sorted(now.keys() - base.keys()), sorted(base.keys() - now.keys())
    assert not (changed or added or removed), (
        f"T-0117 criterion 10: unowned files differ in {T0117_BASE[:7]}..{T0117_TIP[:7]}: changed {changed}, "
        f"added {added}, removed {removed}"
    )


async def test_thought_open_commits_bypass_the_cleaner(tmp_path):
    """Criterion 10 (I10, I15): a ``thought_open=True`` answer commit, even one holding a channel marker, reaches the
    machine with its text and flag exactly as sent; the voiced commits around it are still cleaned and released."""
    draft = "<|channel>draft SECRET-THOUGHT"
    async with tool_rig(
        tmp_path, script=[answer("Hello "), open_answer(draft), answer("there. ")]
    ) as rig:
        await rig.ask("say hello")
        await rig.turn_done()
        commits = [
            (e.text, e.thought_open)
            for e in rig.machine.events(E.LlmCommit)
            if e.request_id == "t1-gen" and e.segment == "answer"
        ]
        sources = rig.spoken_sources()
    assert (draft, True) in commits, f"T-0117 criterion 10: the open commit as sent; got {commits}"
    assert [t for t, open_ in commits if not open_] and "".join(
        t for t, open_ in commits if not open_
    ) == "Hello there. ", commits
    assert sources == "Hello there. ", sources


async def test_step_text_is_route_spoken_prefix(tmp_path):
    """Criterion 10 (T-0114, H-T111a-5): a withheld step 2 whose answer is ``S2 <|channel>thought SECRET<channel|>``
    is embedded in the step-3 request as ``S2 + " "`` (written by hand; it is ``router.route(raw).spoken_prefix``)."""
    raw2 = f"{tsl.S2} <|channel>thought SECRET-THOUGHT<channel|>"
    embedded = f"{tsl.S2} "
    assert router.route(raw2).spoken_prefix == embedded, "premise: the router's answer channel"
    async with tool_rig(tmp_path, policy=LOOP, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(raw2), tool(tsl.CLICK2)], [answer(tsl.S3)])
        await tsl.loop_turn(rig, 3)
        pass1 = tsl.sent(rig, "t1-gen")
        got = tsl.step_request(rig, 3)
    want = tsl.oracle(
        pass1,
        [
            (tsl.S1, "click_at", tsl.CLICK_ARGS, tsl.RECORDED_CLICK),
            (embedded, "click_at", tsl.CLICK2_ARGS, tsl.RECORDED_CLICK),
        ],
    )
    assert tsl.tail(got) == tsl.tail(want), (
        f"T-0117 criterion 10: the step-3 request embeds step 2 as route(raw).spoken_prefix\n"
        f"  want {tsl.tail(want)!r}\n  got  {tsl.tail(got)!r}"
    )


# =====================================================================================================
# Criterion 11: the router's routing is unchanged
# =====================================================================================================


def _base_router() -> types.ModuleType:
    source = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{BASE_SHA}:local_astra/toolcall/router.py"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    name = "t0117_base_router"
    module = types.ModuleType(name)
    module.__file__ = f"{BASE_SHA}:local_astra/toolcall/router.py"
    sys.modules[name] = module  # dataclasses resolve string annotations through sys.modules
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


_BASE_ROUTER: list[types.ModuleType] = []


def base_router() -> types.ModuleType:
    if not _BASE_ROUTER:
        _BASE_ROUTER.append(_base_router())
    return _BASE_ROUTER[0]


ROUTER_TOKENS = (
    "<|channel>",
    "<channel|>",
    "<|chan",
    "nel>",
    "<cc>",
    "</cc>",
    "<genui>",
    "<|tool_call>",
    "<tool_call|>",
    "<eos>",
    '<|"|>',
    '{"name": "click_at"',
    '{"name":"google_search","arguments":{}}',
    "}",
    " ",
    "\n",
    "a",
    "thought",
    "<",
    ">",
    "|",
)
router_texts = st.one_of(
    st.text(max_size=40),
    st.lists(st.sampled_from(ROUTER_TOKENS), max_size=12).map("".join),
)


def _fields(r: Any) -> dict[str, Any]:
    return {f.name: getattr(r, f.name) for f in dataclasses.fields(r)}


@settings(
    max_examples=200,
    deadline=None,
    report_multiple_bugs=False,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(router_texts)
def test_router_route_equals_c7e04f0(x):
    """Criterion 11 (negative; reviewer mutation 13): for any ``str``, ``route(x)``, ``route(x, thought_open=True)``
    and ``template_content(x)`` equal the c7e04f0 router's (loaded from ``git show`` into a temporary module)."""
    gr.require_commits(BASE_SHA)
    old = base_router()
    for kwargs in ({}, {"thought_open": True}):
        got, want = _fields(router.route(x, **kwargs)), _fields(old.route(x, **kwargs))
        assert got == want, (
            f"T-0117 criterion 11: route({x!r}, {kwargs}) now {got}, {BASE_SHA} {want}"
        )
    assert router.template_content(x) == old.template_content(x), x


def test_router_public_names_are_a_superset_of_c7e04f0s():
    """Criterion 11 / owned files: the router helper is additive; every c7e04f0 public name is still exported."""
    gr.require_commits(BASE_SHA)
    missing = set(base_router().__all__) - set(router.__all__)
    assert not missing, f"T-0117: router.__all__ lost {sorted(missing)}"
