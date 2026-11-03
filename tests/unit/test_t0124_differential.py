"""T-0124 criteria 10 and 11 (negative): with no special literal on the tools-off path, and on the tools-on path
for the literals ``_AnswerCleaner`` deletes (I4), nothing changes against the literal base 8b3f6d0.

T-0126 re-scope R1 (docs/tasks/T-0126.md, "Owned files"; a spec change to T-0124 criteria 10/11's tools-on kept rows):
the tools-on rows that held ``<end_of_turn>`` / ``<|endoftext|>`` (``TOOLS_ON_KEPT_TURNS``, their own session and
the by-hand history 8b3f6d0 stored for them) are removed. Their 8b3f6d0 result is the T-0126 bug (the two gate trim
literals that are not tokenizer literals reach history with tools on); their new behaviour is T-0126 criteria 1-2
(``test_session_eot_tools_on.py``). Reviewer B's mutant (tools on routed through ``_LiteralCleaner``) is *not*
caught by this module's ``Press <|tool_call> now. `` row: ``_AnswerCleaner`` deletes the whole literal first, so a
chained second stage sees ``Press  now. `` and changes nothing (T-0126 reviewer B, N1). It is caught by T-0126
criterion 11's source row (``test_t0126_differential.test_untouched_sources_equal_4a25a85``: ``_run_generate`` is
unchanged over T-0126's closed range 4a25a85..b3dc62b) and, behaviourally, by a tail held after a cut
(``test_session_eot_tools_on.test_held_tail_after_a_cut_is_stored``: ``A <|to<end_of_turn> b. `` stores ``A <|to``;
the mutant holds ``<|to`` and never flushes it, storing ``A ``).

A separate module because the base run is a subprocess over ``git archive 8b3f6d0`` (built by T-0114's
``_base_tree`` with its ``BASE_SHA`` monkeypatched for the call; its ``local_astra`` first on ``sys.path``, the origin
printed and checked), so under ``--dist loadfile`` it is a worker of its own. The base run is made once per module (a
module-scoped fixture) and every row compares its own in-process session with it. The harness is T-0117's
(``_PolicyMachine`` and ``_sections``: the tools-off golden's ``capture_sections`` and ``normalise`` plus T-0114's
whole-log normalisation), imported, not copied. T-0114's ``_BASE_RUNNER`` itself is not reused: it is bound to
``test_session_tool_loop._loop_off_capture``; this module's runner has the same shape over its own capture (as
T-0121's differential does).

Every row here holds at the base and must stay green; none is pending.

Readings (listed in the test-author's report):

- the allowance row's machine section "may differ only in the commit split (``Press ``, the open commit, ``<back>
  now. ``)": every machine event other than that stream's ``thought_open=False`` answer commits is equal (in order,
  ``t_ms`` included); the voiced answer commits join to the same text and are either the base's or exactly
  ``["Press ", "<back> now. "]``;
- "criterion 1's rows compare their voiced texts with base" is one tools-off session with one turn per row (criteria
  1, 2, 4 and 5's rows: each states its voice is the base's); the per-turn voice is the turn's ``TtsSpeak.text``
  joined with one space, whitespace collapsed;
- criterion 11's "the source text of ``_ThoughtCleaner``, ``_AnswerCleaner`` and ``_end_tools_on_stream`` equals
  8b3f6d0's" is compared as the AST source segment of the class / method definition in ``session.py`` at 8b3f6d0
  and at ``T0124_TIP`` (``git show``);
- criterion 11's two rows compare the closed range ``T0124_BASE..T0124_TIP`` = 8b3f6d0..c9fc3c8 (T-0094; the working
  tree is never read).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools off unless a scenario
says otherwise (then ``dry_run``, loop off), n = 1, deterministic.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.turn import TurnState
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit._pre_t0155_persona import observed
from tests.unit._t0177_pins import without_vision_frames  # T-0177 B-P1
from tests.unit._tool_turn_support import answer, tool_rig
from tests.unit.test_session_loop import thought
from tests.unit.test_session_thought_tools_off import open_answer
from tests.unit.test_t0117_differential import _ls_tree, _PolicyMachine
from tests.unit.test_t0121_differential import (
    _assert_equal,
    _machine_commits,
    _with_history,
)

REPO = Path(__file__).resolve().parents[2]
#: T-0124's base (docs/tasks/T-0124.md, "Branch": from main at 8b3f6d0, literal). Never a moving ref (tests.md).
BASE_SHA = "8b3f6d0"

SECTIONS = ("machine", "hud", "prompts", "requests", "log", "records", "history")
SECRET = "SECRET-THOUGHT"

# --- the criterion-10 scenarios ----------------------------------------------------------------------------

#: (how the turn is asked, question, script). Tools off, one session, in order. No special literal in any answer
#: stream (``<back>``, ``<|chan`` and ``<channel|>`` are not literals L cuts or deletes once T has run).
TOOLS_OFF_TURNS: tuple[tuple[str, str, tuple[Commit, ...]], ...] = (
    ("typed", "and what does it mean", (answer("A typed answer here. "),)),
    ("ask", "compare them", (answer("x < y and y > z. "),)),
    ("ask", "did you think so", (answer("I thought so, and I still do. "),)),
    ("ask", "which key", (answer("Press <back> now. "),)),
    # T-0121's closed-block row: T-0121's note, no T-0124 note
    ("ask", "what does it say", (answer(f"Here <|channel>thought {SECRET}<channel|> it is. "),)),
    ("ask", "are you done", (answer("Done <channel|>here. "),)),
    (
        "ask",
        "think first",
        (thought("Let me think about it."), answer("First part. "), answer("Second part. ")),
    ),
    # criterion 11: a commit with empty raw text is still submitted
    ("ask", "say it", (answer(""), answer("Said. "))),
    # never completed: T-0121's held ``<|chan``, released at the end
    ("ask", "wait", (answer("Wait <|chan"),)),
    (
        "barge",
        "tell me more",
        (answer("This answer is long enough to still be playing at the cut. "),),
    ),
)
AFTER_BARGE = (answer("After the barge in. "),)

#: tools on, loop off: literals ``_AnswerCleaner`` deletes (I4); the tools-on path is untouched. Neither row tells
#: reviewer B's mutant (tools on routed through the new stage) apart: C deletes the literal before a second stage
#: sees it (T-0126 N1; see the module docstring for the rows that do). The kept-literal rows (``<end_of_turn>`` /
#: ``<|endoftext|>``) left this module with T-0126 R1 (module docstring).
TOOLS_ON_TURNS: tuple[tuple[str, tuple[Commit, ...]], ...] = (
    ("which key", (answer("Press <|tool_call> now. "),)),
    ("what does it say", (answer(f"Sure, <|chan<eos>nel>thought {SECRET}<channel|>ok. "),)),
)

#: the allowance row (T-0121 ruling (a)): ``<b`` can still grow into a literal, so the new stage holds it across the
#: thought_open=True commit; base releases it at once
HOLD_TURN = (answer("Press <b"), open_answer("a draft"), answer("ack> now. "))

#: criteria 1, 2, 4 and 5's rows, one turn each, tools off: their voice is the base's
VOICE_TURNS: tuple[tuple[Commit, ...], ...] = (
    *(
        (answer(f"Press {lit} now. "),)
        for lit in (
            "<eos>",
            "<turn|>",
            "<|tool_call>",
            "<|tool_response>",
            "<|think|>",
            "<end_of_turn>",
            "<|endoftext|>",
        )
    ),
    (answer("Hi <pad>there. "),),
    (answer('Say <|"|>yes<|"|>. '),),
    (answer(f"Sure, <|chan<eos>nel>thought {SECRET}<channel|>ok. "),),
    (answer("Press <|tool"), answer("_call> now. ")),
    (answer("Hi <end_of"), answer("_turn> bye. ")),
    (answer("First. <eos>Second. "), answer("Third. ")),
)


async def _until_turns(rig: Any, n: int, what: str) -> None:
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n,
        f"T-0124 criterion 10: {what} never ended",
    )
    await rig.advance(300)


async def _tools_off(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, kit=None, machine_cls=_PolicyMachine) as rig:
        for how, question, script in TOOLS_OFF_TURNS:
            before = len(rig.lines())
            rig.llm.script = list(script)
            if how == "typed":
                rig.submit(E.UserText(t_ms=rig.clock.now_ms, text=question))
            else:
                await rig.ask(question)
            if how == "barge":
                await rig.until(
                    lambda: (
                        rig.state.state is TurnState.SPEAKING
                        and any(s and "long enough" in s for _, s in rig.audio.plays)
                    ),
                    "T-0124 criterion 10: the barge-in turn never reached the device",
                    limit_ms=10_000,
                )
                await rig.advance(600, step=100)
                now = rig.clock.now_ms
                rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
                await rig.until(
                    lambda: rig.state.state is TurnState.LISTENING, "the takeover never relistened"
                )
                rig.llm.script = list(AFTER_BARGE)
                await rig.words("and then")
                await rig.end_of_turn()
            await _until_turns(rig, before + 1, repr(question))
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path)


async def _turns(tmp: Path, turns: Any, **kw: Any) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, machine_cls=_PolicyMachine, **kw) as rig:
        for i, (question, script) in enumerate(turns):
            rig.llm.script = list(script)
            await rig.ask(question)
            await _until_turns(rig, i + 1, repr(question))
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path)


async def _tools_on(tmp: Path) -> dict[str, Any]:
    return await _turns(tmp, TOOLS_ON_TURNS)


async def _hold(tmp: Path) -> dict[str, Any]:
    return await _turns(tmp, [("hold on", HOLD_TURN)], kit=None)


async def _voice(tmp: Path) -> dict[str, Any]:
    return await _turns(tmp, [(f"row {i}", s) for i, s in enumerate(VOICE_TURNS)], kit=None)


SCENARIOS = {
    "tools_off": _tools_off,
    "tools_on": _tools_on,
    "hold": _hold,
    "voice": _voice,
}


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario session. Runs in-process on the working tree and in a subprocess on the 8b3f6d0 tree."""
    return {name: await run(tmp / name) for name, run in SCENARIOS.items()}


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0124_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The 8b3f6d0 run of every scenario, once per module."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0124_base")
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


# =====================================================================================================
# Criterion 10
# =====================================================================================================


async def test_no_literal_tools_off_equals_8b3f6d0(base, tmp_path):
    """Criterion 10 (negative; mutations: hold every ``<``-led tail, cut or delete a non-literal, a note when
    ``chars == 0``, skip empty-raw commits, feed or reset on ``thought_open=True`` commits): tools off, a typed turn,
    ``x < y and y > z.``, the word "thought", ``Press <back> now.``, T-0121's closed block, a stray close, a
    ``thought`` segment then answers, an empty commit then an answer, ``Wait <|chan`` never completed and a barge-in
    turn give the same machine steps, HUD messages, prompts, request bodies, session-log records, ``turns.jsonl``
    lines and history as 8b3f6d0."""
    # T-0155 O-8: the persona sentence; T-0177 B-P1: the astra.vision.frame messages and their wire.out records
    now = without_vision_frames(observed(await _tools_off(tmp_path / "now")))
    for section in SECTIONS:
        _assert_equal(
            base["tools_off"][section], now[section], f"criterion 10: tools off {section!r}"
        )
    # premises: the scenarios ran what they say
    commits = _commits(now)
    assert ("Press <back> now. ", False, "answer") in commits, (
        "premise: the non-literal tag reached the machine as sent"
    )
    assert ("", False, "answer") in commits, (
        "premise (criterion 11): the empty-raw commit was submitted"
    )
    assert ("Let me think about it.", False, "thought") in commits, (
        "premise (criterion 11): the thought segment reached the machine as sent"
    )
    assert [r for r in now["log"] if r["kind"] == "note" and r["data"].get("chars") == 42], (
        "premise: T-0121's note ran for the closed block"
    )
    stopped = [s for s in now["machine"] if s["event"].get("type") == "playback.stopped"]
    assert stopped, "premise: the barge-in stopped playback"


async def test_tools_on_literal_turns_equal_8b3f6d0(base, tmp_path):
    """Criterion 10, tools on, loop off: ``Press <|tool_call> now.`` and criterion 2's string are identical to 8b3f6d0
    in every section (``_AnswerCleaner`` deletes, I4). This row does not catch reviewer B's mutant (tools on routed
    through the new stage): C has already deleted the literal, so the chained stage passes ``Press  now. `` through
    unchanged (T-0126 N1; the rows that catch it are named in the module docstring). The kept-literal rows
    (``<end_of_turn>`` / ``<|endoftext|>``) were removed by T-0126 R1 (module docstring)."""
    # T-0155 O-8: the persona sentence; T-0177 B-P1: the astra.vision.frame messages and their wire.out records
    now = without_vision_frames(observed(await _tools_on(tmp_path / "now")))
    for section in SECTIONS:
        _assert_equal(
            base["tools_on"][section], now[section], f"criterion 10: tools on {section!r}"
        )
    stored = [h["assistant_spoken"] for h in now["history"]]
    assert stored and stored[0] == "Press  now. ", (
        f"premise: tools on deletes the literal (I4); history {stored}"
    )


async def test_hold_allowance_row(base, tmp_path):
    """Criterion 10, the allowance row (T-0121 ruling (a); guard too wide: the held ``<b`` must still be released):
    tools off, ``Press <b`` + a ``thought_open=True`` ``a draft`` + ``ack> now.``. The ``TtsSpeak`` (source, text)
    sequence, history and ``turns.jsonl`` (``canvases`` excused) equal 8b3f6d0's; the machine section differs at most
    in the commit split (``Press ``, the open commit, ``<back> now. ``)."""
    now = await _hold(tmp_path / "now")
    old = base["hold"]
    _assert_equal(_speaks(old), _speaks(now), "criterion 10 (hold): TtsSpeak (source, text)")
    _assert_equal(old["history"], now["history"], "criterion 10 (hold): history")
    assert _speaks(now), "premise: something was voiced"

    def without_canvases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if k != "canvases"} for r in records]

    assert len(now["records"]) == len(old["records"]) == 1, (old["records"], now["records"])
    _assert_equal(
        without_canvases(old["records"]),
        without_canvases(now["records"]),
        "criterion 10 (hold): turns.jsonl (all fields but canvases)",
    )

    def voiced(e: dict[str, Any]) -> bool:
        return (
            e.get("type") == "llm.commit"
            and e.get("segment") == "answer"
            and not e.get("thought_open")
            and e.get("request_id") == "t1-gen"
        )

    others = [
        [s["event"] for s in sections["machine"] if not voiced(s["event"])]
        for sections in (old, now)
    ]
    _assert_equal(
        others[0],
        others[1],
        "criterion 10 (hold): machine events other than the voiced answer commits",
    )
    texts = [
        [e["text"] for e in _machine_commits(sections) if voiced(e)] for sections in (old, now)
    ]
    assert "".join(texts[1]) == "".join(texts[0]) == "Press <back> now. ", texts
    assert texts[1] in (texts[0], ["Press ", "<back> now. "]), (
        f"T-0124 criterion 10 (hold): the commit split may only move ``<b``; base {texts[0]}, now {texts[1]}"
    )
    opens = [(e["text"], e["thought_open"]) for e in _machine_commits(now) if e.get("thought_open")]
    assert opens == [("a draft", True)], (
        f"T-0124 criterion 11: the thought_open=True commit reaches the machine as sent; got {opens}"
    )


async def test_literal_rows_voice_equals_8b3f6d0(base, tmp_path):
    """Criterion 10, last item (mutation: delete trim literals instead of cutting, which voices ``now``): each of
    criteria 1, 2, 4 and 5's rows voices what 8b3f6d0 voices, per turn (``TtsSpeak.text`` joined with one space,
    whitespace collapsed)."""
    now = await _voice(tmp_path / "now")

    def per_turn(sections: dict[str, Any]) -> list[str]:
        turns: dict[str, list[str]] = {}
        for step in sections["machine"]:
            for c in step["commands"]:
                if c.get("type") == "tts.speak":
                    turns.setdefault(c["utterance_id"].split("-")[0], []).append(c["text"])
        return [
            " ".join(" ".join(v).split())
            for _, v in sorted(turns.items(), key=lambda kv: int(kv[0][1:]))
        ]

    old_v, now_v = per_turn(base["voice"]), per_turn(now)
    assert len(old_v) == len(VOICE_TURNS), f"premise: one voiced turn per row at the base; {old_v}"
    _assert_equal(old_v, now_v, "criterion 10: the literal rows' voice")
    assert old_v[0] == "Press", (
        f"premise: the base voices criterion 1's first row as 'Press'; {old_v}"
    )


# =====================================================================================================
# Criterion 11
# =====================================================================================================


def _definitions(source: str) -> dict[str, str]:
    """The source segments of ``class _ThoughtCleaner``, ``class _AnswerCleaner`` and every ``_end_tools_on_stream``
    definition in ``session.py`` (decorators excluded, as ``ast.get_source_segment`` gives them)."""
    tree = ast.parse(source)
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef) and node.name in ("_ThoughtCleaner", "_AnswerCleaner")
        ) or (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == "_end_tools_on_stream"
        ):
            seg = ast.get_source_segment(source, node)
            assert seg is not None, node.name
            found.setdefault(node.name, []).append(seg)
    assert all(len(v) == 1 for v in found.values()), {k: len(v) for k, v in found.items()}
    return {k: v[0] for k, v in found.items()}


#: Criterion 11's rows are closed over T-0124's own range (tests.md "pin both ends", T-0094; orchestrator ruling
#: 2026-10-18): BASE = 8b3f6d0 (the task's literal base), TIP = c9fc3c8 (the branch's last commit touching non-test
#: files, the docs/02 §5 wording). The working tree is never read, so a later task's legitimate edit of a guarded path
#: cannot redden these rows. There is no merge commit yet, so ``gr.prove_range`` (which needs M) cannot run; the rows
#: check its pre-merge half (both commits present, BASE an ancestor of TIP). At merge, whoever merges replaces this
#: with ``gr.prove_range(T0124_BASE, T0124_TIP, M, later=...)`` if TIP is M^2 (T-0094).
T0124_BASE = "8b3f6d0e97ba1728059bc2a683c4e9a5c945a185"
T0124_TIP = "c9fc3c8137c5adc47745eb4b1e0514b01042a0e5"


def _prove_open_range() -> None:
    gr.require_commits(T0124_BASE, T0124_TIP)
    assert gr.is_ancestor(T0124_BASE, T0124_TIP), f"{T0124_BASE} is not an ancestor of {T0124_TIP}"
    assert gr.is_ancestor(BASE_SHA, T0124_BASE) and gr.is_ancestor(T0124_BASE, BASE_SHA), (
        f"premise: T0124_BASE is the task's literal base {BASE_SHA}"
    )


def _show(rev: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "show", f"{rev}:{path}"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout


def test_cleaner_sources_unchanged():
    """Criterion 11 (mutation: edit ``_ThoughtCleaner`` instead of chaining; reach into ``_AnswerCleaner`` or the
    tools-on end of stream): the source of ``_ThoughtCleaner``, ``_AnswerCleaner`` and ``_end_tools_on_stream`` in
    ``session.py`` is the same, character for character, at ``T0124_BASE`` (8b3f6d0) and ``T0124_TIP``."""
    _prove_open_range()
    old = _definitions(_show(T0124_BASE, "local_astra/conductor/session.py"))
    assert set(old) == {"_ThoughtCleaner", "_AnswerCleaner", "_end_tools_on_stream"}, sorted(old)
    now = _definitions(_show(T0124_TIP, "local_astra/conductor/session.py"))
    assert set(now) == set(old), sorted(now)
    changed = sorted(k for k in old if old[k] != now[k])
    assert not changed, (
        f"T-0124 criterion 11: changed in {T0124_BASE[:7]}..{T0124_TIP[:7]}: {changed}"
    )


#: Criterion 11's files, compared over T-0124's closed range ``T0124_BASE..T0124_TIP``
UNOWNED = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/toolcall/router.py",
    "local_astra/toolcall/special_tokens.py",
    "local_astra/contracts",
    "tests/fakes",
)


def test_unowned_files_equal_8b3f6d0():
    """Criterion 11, last item: the machine, the gate, the client, the router, the special-token table, the contracts
    and the fakes are byte-identical (git blob hashes) at ``T0124_BASE`` (8b3f6d0) and ``T0124_TIP``, with no file
    added or removed under the two directories."""
    _prove_open_range()
    old = _ls_tree(T0124_BASE, UNOWNED)
    assert set(UNOWNED[:5]) <= set(old), (
        f"premise: {T0124_BASE[:7]} has the five files; got {sorted(old)[:8]}"
    )
    now = _ls_tree(T0124_TIP, UNOWNED)
    changed = sorted(p for p in old.keys() & now.keys() if old[p] != now[p])
    added, removed = sorted(now.keys() - old.keys()), sorted(old.keys() - now.keys())
    assert not (changed or added or removed), (
        f"T-0124 criterion 11: unowned files differ in {T0124_BASE[:7]}..{T0124_TIP[:7]}: changed {changed}, "
        f"added {added}, removed {removed}"
    )
