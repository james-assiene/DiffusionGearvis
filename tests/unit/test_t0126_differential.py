"""T-0126 criteria 10 and 11 (negative): with no E literal (``<end_of_turn>``, ``<|endoftext|>``) nothing changes
against the literal base 4a25a85, the voice never changes, and the paths the task does not own are untouched.

A separate module because the base run is a subprocess over ``git archive 4a25a85`` (built by T-0114's
``_base_tree`` with its ``BASE_SHA`` monkeypatched for the call; its ``local_astra`` first on ``sys.path``, the origin
printed and checked), so under ``--dist loadfile`` it is a worker of its own. The base run is made once per module (a
module-scoped fixture) and every row compares its own in-process session with it. The harness is T-0124's and
T-0121's (``_PolicyMachine``, ``_with_history``, ``_machine_commits``, ``_assert_equal``, T-0124's tools-off
scenario), imported, not copied; the runner has the same shape over this module's ``_capture``.

Every row here holds at the base (the working tree is the base's code until the implementation lands) and must stay
green; none is pending.

Readings (listed in the test-author's report):

- ``_assert_equal``'s message names T-0121's base; every call here passes a ``what`` naming T-0126 and 4a25a85;
- the voice rows' "per turn" is keyed by the turn id of each ``TtsSpeak``'s utterance id; a turn with no
  ``TtsSpeak`` at all (criterion 7's row after the fix: nothing is submitted) voices ``""``, as a turn whose only
  speak has empty text does at the base;
- the allowance row's machine section "may differ only in the voiced commit split": every machine event other than
  that stream's ``thought_open=False`` answer commits is equal, and the voiced commits join to the same text and are
  either the base's or exactly ``["Tap ", "<xy> now. "]`` (the wider hold keeps ``<x`` back);
- the step-text row compares the step-2 request body the fake records (``step_bodies``, what the real client would
  send), normalised like every other section;
- criterion 11's source and file rows compare the closed range ``T0126_BASE..T0126_TIP`` = 4a25a85..b3dc62b (T-0094,
  as T-0124's were at c0395e8; the working tree is never read). The source row also compares the ``step_text=``
  line of ``_end_tools_on_stream`` (owned files: "Not edited: ... the ``step_text=`` line").

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``dry_run`` (loop off
unless a scenario says otherwise) or off where a scenario says so, n = 1, deterministic.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.turn_machine import TurnPolicy
from local_astra.contracts.llm import Commit
from local_astra.contracts.turn import TurnState
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
from tests.unit.test_session_thought_tools_off import open_answer
from tests.unit.test_session_tools_off_golden import normalise
from tests.unit.test_t0117_differential import _ls_tree, _PolicyMachine
from tests.unit.test_t0121_differential import _assert_equal, _machine_commits, _with_history
from tests.unit.test_t0124_differential import _tools_off as _t0124_tools_off

REPO = Path(__file__).resolve().parents[2]
#: T-0126's base (docs/tasks/T-0126.md, "Branch": from main at 4a25a85, literal). Never a moving ref (tests.md).
BASE_SHA = "4a25a85"

SECTIONS = ("machine", "hud", "prompts", "requests", "log", "records", "history")
SECRET = "SECRET-THOUGHT"
LOOP = TurnPolicy(thinking=True, tool_loop=True)

Turn = tuple[str, tuple[Commit, ...], tuple[Commit, ...] | None]

# --- the criterion-10 scenarios ----------------------------------------------------------------------------

#: tools on, loop off, one session, in order: (question, script, repair continuation or None). No E literal.
TOOLS_ON_TURNS: tuple[Turn, ...] = (
    ("and what does it say now", (answer("It says welcome back."),), None),
    ("which key", (answer("Press <|tool_call> now. "),), None),
    ("and then", (answer("Hi <eos> bye. "),), None),
    # T-0117's closed-block row (its note, no T-0126 note)
    (
        "what does it say",
        (answer(f"Sure, <|channel>thought {SECRET}<channel|>here it is. "),),
        None,
    ),
    ("compare them", (answer("x < y and y > z. "),), None),
    ("did you think so", (answer("I thought so, and I still do. "),), None),
    ("which one", (answer("Press <back> now. "),), None),
    ("which key again", (answer("Press <|tool"), answer("_call> now. ")), None),
    # I15: a held tail across a thought_open=True commit
    ("hold on", (answer("Press <b"), open_answer("a draft"), answer("ack> now. ")), None),
    ("click the sign in button", (answer(tsl.S1), tool(VALID_CLICK)), None),
    ("click the small one", (answer(tsl.S1), tool(REPAIRABLE_CLICK)), (tool(REPAIRED_CLICK),)),
)

#: tools off: T-0124's ``TOOLS_OFF_TURNS`` session (``test_t0124_differential._tools_off``) plus this turn, whose
#: T-0124 result ``Hi `` is unchanged
TOOLS_OFF_EOT = ("end of turn", (answer("Hi <end_of_turn> bye. "),), None)

#: the allowance row: ``x`` is a hold character only because of ``<|endoftext|>`` (it is in no tokenizer literal),
#: so the wider hold keeps ``<x`` back until ``y>`` arrives; 4a25a85 releases it at once. ``<xy>`` is no literal.
ALLOWANCE = (("hold on", (answer("Tap <x"), answer("y> now. ")), None),)

#: criteria 1-3 and 7's rows, tools on, loop off, one turn each: their voice is the base's
VOICE_TURNS: tuple[tuple[Commit, ...], ...] = (
    (answer("Press <end_of_turn> now. "),),
    (answer("Press <|endoftext|> now. "),),
    (answer("Hi <end_of_turn> bye. "),),
    (answer("Hi <end_of"), answer("_turn> bye. ")),
    (answer("Hi <|endof"), answer("text|> bye. ")),
    (answer("Hi <end_of<pad>_turn> bye. "),),
    (answer("Hi <end_of"), open_answer("draft"), answer("_turn> bye. ")),
    (answer("First. <end_of_turn>Second. "), answer("Third. ")),
    (answer("<end_of_turn> bye. "),),
)
#: criterion 5's rows, loop on: the two-step search turn whose report holds E, then a step-1 final answer
REPORT_RAW = "Found it: the café opens at nine. <end_of_turn>user: and tomorrow? "
STEP1_FINAL = (answer("Hi <end_of_turn> bye. "),)

#: H-T126-5: step 1 holds an E literal before a search call; the step-2 request embeds ``step_text``, unchanged
STEP_TEXT_SCRIPT = (answer("Let me look. <end_of_turn>junk "), tool(tsl.SEARCH_CALL))


async def _until_turns(rig: Any, n: int, what: str) -> None:
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n,
        f"T-0126 criterion 10: {what} never ended",
    )
    await rig.advance(300)


async def _session(tmp: Path, turns: Sequence[Turn], **kw: Any) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, machine_cls=_PolicyMachine, **kw) as rig:
        for i, (question, script, continuation) in enumerate(turns):
            rig.llm.script = list(script)
            if continuation is not None:
                rig.llm.script_continuation(*continuation)
            await rig.ask(question)
            await _until_turns(rig, i + 1, repr(question))
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path)


async def _tools_on(tmp: Path) -> dict[str, Any]:
    return await _session(tmp, TOOLS_ON_TURNS)


async def _loop_on(tmp: Path) -> dict[str, Any]:
    """T-0114's two-step search turn (as T-0117's differential runs it), with history."""
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
    return _with_history(rig, tmp, session_id, log_path)


async def _tools_off(tmp: Path) -> dict[str, Any]:
    return await _t0124_tools_off(tmp)


async def _tools_off_eot(tmp: Path) -> dict[str, Any]:
    return await _session(tmp, (TOOLS_OFF_EOT,), kit=None)


async def _allowance(tmp: Path) -> dict[str, Any]:
    return await _session(tmp, ALLOWANCE)


async def _voice(tmp: Path) -> dict[str, Any]:
    return await _session(tmp, [(f"row {i}", s, None) for i, s in enumerate(VOICE_TURNS)])


async def _voice_loop(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(
        tmp,
        machine_cls=_PolicyMachine,
        policy=LOOP,
        search=FakeSearch(),
        script=[answer(tsl.PREFIX), tool(tsl.SEARCH_CALL)],
    ) as rig:
        rig.llm.script_steps([answer(REPORT_RAW)])
        await tsl.loop_turn(rig, 2)
        await _until_turns(rig, 1, "the loop-on report turn")
        rig.llm.script = list(STEP1_FINAL)
        await rig.ask("and then")
        await _until_turns(rig, 2, "the loop-on step-1 final answer")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path)


async def _step_text(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(
        tmp,
        machine_cls=_PolicyMachine,
        policy=LOOP,
        search=FakeSearch(),
        script=list(STEP_TEXT_SCRIPT),
    ) as rig:
        rig.llm.script_steps([answer(tsl.REPORT)])
        await tsl.loop_turn(rig, 2)
        await rig.advance(300)
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
        bodies = list(rig.llm.step_bodies)
    out: dict[str, Any] = normalise({"step_bodies": bodies}, session_id, tmp, log_path)
    return out


SCENARIOS = {
    "tools_on": _tools_on,
    "loop_on": _loop_on,
    "tools_off": _tools_off,
    "tools_off_eot": _tools_off_eot,
    "allowance": _allowance,
    "voice": _voice,
    "voice_loop": _voice_loop,
    "step_text": _step_text,
}


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario session. Runs in-process on the working tree and in a subprocess on the 4a25a85 tree."""
    return {name: await run(tmp / name) for name, run in SCENARIOS.items()}


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0126_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The 4a25a85 run of every scenario, once per module."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0126_base")
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
        expected, got, f"T-0126 {what} (base {BASE_SHA}; message prefix below is T-0121's helper):"
    )


def _commits(sections: dict[str, Any]) -> list[tuple[str, bool, str]]:
    return [
        (e["text"], e["thought_open"], e["segment"])
        for s in sections["machine"]
        for e in [s["event"]]
        if e.get("type") == "llm.commit"
    ]


def _speaks(sections: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (c["source"], c["text"])
        for step in sections["machine"]
        for c in step["commands"]
        if c.get("type") == "tts.speak"
    ]


def _per_turn_voice(sections: dict[str, Any], turns: int) -> list[str]:
    """Each turn's ``TtsSpeak.text`` values joined with one space, whitespace collapsed; ``""`` for a turn with none."""
    found: dict[str, list[str]] = {}
    for step in sections["machine"]:
        for c in step["commands"]:
            if c.get("type") == "tts.speak":
                found.setdefault(c["utterance_id"].split("-")[0], []).append(c["text"])
    extra = sorted(set(found) - {f"t{k}" for k in range(1, turns + 1)})
    assert not extra, f"premise: speaks of unexpected turns {extra}"
    return [" ".join(" ".join(found.get(f"t{k}", [])).split()) for k in range(1, turns + 1)]


# =====================================================================================================
# Criterion 10
# =====================================================================================================


async def test_no_eot_turns_equal_4a25a85(base, tmp_path):
    """Criterion 10, main comparison, tools on (mutations 3, route tools-on through ``_LiteralCleaner``; 6, hold every
    ``<``-led tail; 12, a note when ``chars == 0``; 14, count I4's removals; 16, feed or reset on ``thought_open=True``
    commits; 17, refuse the call): loop off, a toolless turn, ``Press <|tool_call> now.``, ``Hi <eos> bye.``,
    T-0117's closed-block row, ``x < y and y > z.``, the word "thought", ``Press <back> now.``, the split
    ``Press <|tool`` / ``_call> now.``, the I15 row, a valid-call turn with a spoken prefix and a repairable-draft
    turn with its repair; and loop on, T-0114's two-step search turn: the same machine steps, HUD messages, prompts,
    request bodies, session-log records, ``turns.jsonl`` lines and history as 4a25a85."""
    now = observed(  # T-0155 O-8: the persona sentence only
        {
            "tools_on": await _tools_on(tmp_path / "on"),
            "loop_on": await _loop_on(tmp_path / "loop"),
        }
    )
    for scenario, sections in now.items():
        for section in SECTIONS:
            _same(
                base[scenario][section], sections[section], f"criterion 10: {scenario} {section!r}"
            )
    # premises: the scenarios ran what they say
    commits = _commits(now["tools_on"])
    assert ("a draft", True, "answer") in commits, (
        "premise (I15): the open commit reached the machine"
    )
    stored = [h["assistant_spoken"] for h in now["tools_on"]["history"]]
    log = now["tools_on"]["log"]
    statuses = [r["data"].get("status") for r in log if r["kind"] == "tool.call"]
    assert "repairing" in statuses and "dry_run" in statuses, statuses
    assert [r for r in log if r["kind"] == "note" and r["data"].get("chars") == 42], (
        "premise: T-0117's note ran for the closed block"
    )
    requests = json.dumps(now["tools_on"]["prompts"], ensure_ascii=False)
    assert "Press  now. " in requests and "Hi  bye. " in requests, (
        f"premise: the I4 rows were stored as ``Press  now. `` / ``Hi  bye. `` (replayed in later prompts); "
        f"last history {stored}"
    )
    steps = [r for r in now["loop_on"]["log"]
             if r["kind"] == "llm.request" and r["data"].get("phase") == "tool_step"]  # fmt: skip
    assert len(steps) == 1, steps


async def test_tools_off_unchanged_4a25a85(base, tmp_path):
    """Criterion 10, main comparison, tools off (mutation: route tools-off through the new cut, or reach into
    ``_LiteralCleaner``): T-0124's ``TOOLS_OFF_TURNS`` session and ``Hi <end_of_turn> bye.`` give the same sections as
    4a25a85, and the latter's T-0124 result ``Hi `` is unchanged."""
    # T-0155 O-8: the persona sentence only
    off = observed(await _tools_off(tmp_path / "off"))
    eot = observed(await _tools_off_eot(tmp_path / "eot"))
    for section in SECTIONS:
        _same(base["tools_off"][section], off[section], f"criterion 10: tools off {section!r}")
        _same(
            base["tools_off_eot"][section], eot[section], f"criterion 10: tools off, E {section!r}"
        )
    assert [h["assistant_spoken"] for h in eot["history"]] == ["Hi "], eot["history"]


async def test_hold_allowance_row(base, tmp_path):
    """Criterion 10, the allowance row (T-0121 ruling (a); guard too wide: the held ``<x`` must still be released):
    tools on, ``Tap <x`` / ``y> now.``. The ``TtsSpeak`` (source, text) sequence, history and ``turns.jsonl``
    (``canvases`` excused) equal 4a25a85's; the machine section differs at most in the voiced commit split."""
    now = await _allowance(tmp_path / "now")
    old = base["allowance"]
    _same(_speaks(old), _speaks(now), "criterion 10 (allowance): TtsSpeak (source, text)")
    _same(old["history"], now["history"], "criterion 10 (allowance): history")
    assert _speaks(now), "premise: something was voiced"
    assert [h["assistant_spoken"] for h in now["history"]] == ["Tap <xy> now. "], now["history"]

    def without_canvases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if k != "canvases"} for r in records]

    assert len(now["records"]) == len(old["records"]) == 1, (old["records"], now["records"])
    _same(without_canvases(old["records"]), without_canvases(now["records"]),
          "criterion 10 (allowance): turns.jsonl (all fields but canvases)")  # fmt: skip

    def voiced(e: dict[str, Any]) -> bool:
        return (
            e.get("type") == "llm.commit"
            and e.get("segment") == "answer"
            and not e.get("thought_open")
            and e.get("request_id") == "t1-gen"
        )

    others = [[s["event"] for s in x["machine"] if not voiced(s["event"])] for x in (old, now)]
    _same(
        others[0],
        others[1],
        "criterion 10 (allowance): machine events other than the voiced commits",
    )
    texts = [[e["text"] for e in _machine_commits(x) if voiced(e)] for x in (old, now)]
    assert "".join(texts[1]) == "".join(texts[0]) == "Tap <xy> now. ", texts
    assert texts[1] in (texts[0], ["Tap ", "<xy> now. "]), (
        f"T-0126 criterion 10 (allowance): the commit split may only move ``<x``; base {texts[0]}, now {texts[1]}"
    )


async def test_eot_rows_voice_equals_4a25a85(base, tmp_path):
    """Criterion 10, voice rows (H-T126-1, cut and not delete; mutation 1, delete E literals, which voices ``now`` /
    ``bye`` / ``Second``): the rows of criteria 1-3, 5 and 7 voice what 4a25a85 voices, per turn (``TtsSpeak.text``
    joined with one space, whitespace collapsed)."""
    now, loop = await _voice(tmp_path / "off"), await _voice_loop(tmp_path / "loop")
    old_v, now_v = (_per_turn_voice(x, len(VOICE_TURNS)) for x in (base["voice"], now))
    _same(old_v, now_v, "criterion 10: the E rows' voice (loop off)")
    old_l, now_l = (_per_turn_voice(x, 2) for x in (base["voice_loop"], loop))
    _same(old_l, now_l, "criterion 10: the E rows' voice (loop on)")
    # premises, by hand: what the gate voices for these rows (the cut is where the gate trims)
    assert old_v[:3] == ["Press", "Press", "Hi"], old_v
    assert old_v[-1] == "", f"premise: criterion 7's row voices nothing at the base; {old_v}"
    assert not [v for v in old_v + old_l if "now" in v.split() or "bye" in v or "tomorrow" in v], (
        old_v,
        old_l,
    )


async def test_step2_request_equals_4a25a85(base, tmp_path):
    """Criterion 10, step-text row (H-T126-5; mutation 10, cut ``step_text``): with the loop on, step 1
    ``answer("Let me look. <end_of_turn>junk ")`` plus a search call: the step-2 request body equals 4a25a85's."""
    now = observed(await _step_text(tmp_path / "now"))  # T-0155 O-8: the persona sentence only
    assert len(base["step_text"]["step_bodies"]) == 1, "premise: one step request at the base"
    _same(
        base["step_text"]["step_bodies"],
        now["step_bodies"],
        "criterion 10: the step-2 request body",
    )
    assert "junk" in json.dumps(now["step_bodies"]), (
        "premise: step_text is the routed raw stream (H-T126-5: unchanged), so it still embeds the post-cut text"
    )


# =====================================================================================================
# Criterion 11
# =====================================================================================================

#: criterion 11's untouched definitions in ``session.py``
CLASSES = ("_ThoughtCleaner", "_LiteralCleaner")
METHODS = ("_run_generate", "_stream_commits", "_run_step", "_judge_tool")


def _definitions(source: str) -> dict[str, str]:
    """The source segment of each class in ``CLASSES`` and method in ``METHODS``, plus the ``step_text=`` line(s) of
    ``_end_tools_on_stream`` (as ``step_text``)."""
    tree = ast.parse(source)
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef) and node.name in CLASSES) or (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in METHODS
        ):
            seg = ast.get_source_segment(source, node)
            assert seg is not None, node.name
            found.setdefault(node.name, []).append(seg)
        elif isinstance(node, ast.FunctionDef) and node.name == "_end_tools_on_stream":
            seg = ast.get_source_segment(source, node)
            assert seg is not None, node.name
            lines = [ln.strip() for ln in seg.splitlines() if "step_text=" in ln]
            found.setdefault("step_text", []).append("\n".join(lines))
    assert all(len(v) == 1 for v in found.values()), {k: len(v) for k, v in found.items()}
    return {k: v[0] for k, v in found.items()}


#: Criterion 11's rows are closed over T-0126's own range (tests.md "pin both ends", T-0094; orchestrator ruling
#: 2026-10-19, reviewer N2): BASE = 4a25a85 (the task's literal base), TIP = b3dc62b (the branch's last commit
#: touching non-test files: the docs/02 §5 correction and the rulings). The working tree is never read, so a later
#: task's legitimate edit of a guarded path cannot redden these rows. There is no merge commit yet, so
#: ``gr.prove_range`` (which needs M) cannot run; the rows check its pre-merge half (both commits present, BASE an
#: ancestor of TIP). At merge, whoever merges replaces this with ``gr.prove_range(T0126_BASE, T0126_TIP, M, later=...)``
#: if TIP is M^2 (T-0094).
T0126_BASE = "4a25a85a59a7e4105ff3cb69b8b660b2c905c5ce"
T0126_TIP = "b3dc62b366cf83cc7172308c96ede4d479f0da46"


def _prove_open_range() -> None:
    gr.require_commits(T0126_BASE, T0126_TIP)
    assert gr.is_ancestor(T0126_BASE, T0126_TIP), f"{T0126_BASE} is not an ancestor of {T0126_TIP}"
    assert gr.is_ancestor(BASE_SHA, T0126_BASE) and gr.is_ancestor(T0126_BASE, BASE_SHA), (
        f"premise: T0126_BASE is the task's literal base {BASE_SHA}"
    )


def _show(rev: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "show", f"{rev}:{path}"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout


def test_untouched_sources_equal_4a25a85():
    """Criterion 11 (owned files, "not edited"; mutation: fix ``_run_generate``/``_stream_commits``/``_run_step``
    instead of the cleaner, or cut ``step_text``, or route tools on through ``_LiteralCleaner``): the source of
    ``_ThoughtCleaner``, ``_LiteralCleaner``, ``_run_generate``, ``_stream_commits``, ``_run_step``, ``_judge_tool``
    and the ``step_text=`` line of ``session.py`` is the same, character for character, at ``T0126_BASE`` (4a25a85)
    and ``T0126_TIP``."""
    _prove_open_range()
    old = _definitions(_show(T0126_BASE, "local_astra/conductor/session.py"))
    assert set(old) == {*CLASSES, *METHODS, "step_text"}, sorted(old)
    assert old["step_text"] == "step_text=router.route(cleaner.raw).spoken_prefix,", old[
        "step_text"
    ]
    now = _definitions(_show(T0126_TIP, "local_astra/conductor/session.py"))
    assert set(now) == set(old), sorted(now)
    changed = sorted(k for k in old if old[k] != now[k])
    assert not changed, (
        f"T-0126 criterion 11: changed in {T0126_BASE[:7]}..{T0126_TIP[:7]}: {changed}"
    )


#: criterion 11's files, compared over T-0126's closed range ``T0126_BASE..T0126_TIP``
UNOWNED = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/toolcall/router.py",
    "local_astra/toolcall/special_tokens.py",
    "local_astra/toolcall/live.py",
    "local_astra/toolcall/loop.py",
    "local_astra/contracts",
    "tests/fakes",
)


def test_unowned_files_equal_4a25a85():
    """Criterion 11, last item: the machine, the gate, the client, the router, the special-token table, ``live.py``,
    ``loop.py``, the contracts and the fakes are byte-identical (git blob hashes) at ``T0126_BASE`` (4a25a85) and
    ``T0126_TIP``, with no file added or removed under the two directories."""
    _prove_open_range()
    old = _ls_tree(T0126_BASE, UNOWNED)
    assert set(UNOWNED[:7]) <= set(old), (
        f"premise: {T0126_BASE[:7]} has the seven files; got {sorted(old)[:8]}"
    )
    now = _ls_tree(T0126_TIP, UNOWNED)
    changed = sorted(p for p in old.keys() & now.keys() if old[p] != now[p])
    added, removed = sorted(now.keys() - old.keys()), sorted(old.keys() - now.keys())
    assert not (changed or added or removed), (
        f"T-0126 criterion 11: unowned files differ in {T0126_BASE[:7]}..{T0126_TIP[:7]}: changed {changed}, "
        f"added {added}, removed {removed}"
    )
