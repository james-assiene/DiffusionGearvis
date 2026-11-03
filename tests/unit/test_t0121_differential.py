"""T-0121 criteria 9 and 10 (negative): with no channel marker on the tools-off path, and on the tools-on path
whatever it holds, nothing changes against the literal base 791dc54.

A separate module because the base run is a subprocess over ``git archive 791dc54`` (built by T-0114's
``_base_tree`` with its ``BASE_SHA`` monkeypatched for the call, as T-0117 does; its ``local_astra`` first on
``sys.path``, the origin printed and checked), so under ``--dist loadfile`` it is a worker of its own. The base run is
made once per module (a module-scoped fixture) and every row compares its own in-process session with it. The harness
is T-0117's (``_PolicyMachine`` and ``_sections``: the tools-off golden's ``capture_sections`` and ``normalise`` plus
T-0114's whole-log normalisation), imported, not copied. T-0114's ``_BASE_RUNNER`` itself is not reused: it is bound
to ``test_session_tool_loop._loop_off_capture``; this module's runner has the same shape over its own capture.

Every row here holds at the base and must stay green; none is pending.

Readings (listed in the test-author's report):

- the ``Wait <|chan`` row's machine section "may differ only in that the held ``<|chan`` arrives as its own commit
  before ``llm.done``": every machine event other than that stream's ``thought_open=False`` answer commits is equal
  (in order, ``t_ms`` included); the answer commits join to the same text, and are either the base's or exactly
  ``["Wait ", "<|chan"]``; and each precedes the stream's ``llm.done``;
- criterion 10's "``thought_open=True`` and non-answer commits reach the machine as sent" is checked twice: inside the
  differential (an I15 turn, a ``thought`` segment turn) and in its own row, with channel markers inside those commits
  so a cleaner fed them would change the voice; "a tools-off commit with empty raw text is still submitted as at base"
  is a differential turn (``answer("")`` then an answer) plus a premise that the machine received the empty commit;
- the unowned-files row adds ``toolcall/router.py`` to T-0117's list (criterion 10 names it).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools off unless a scenario
says otherwise (then ``dry_run``), n = 1, deterministic.
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
from local_astra.contracts.turn import TurnState
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit._pre_t0155_persona import observed
from tests.unit._t0177_pins import without_vision_frames  # T-0177 B-P1
from tests.unit._tool_turn_support import answer, tool_rig
from tests.unit.test_session_loop import thought
from tests.unit.test_t0117_differential import (
    _ls_tree,
    _PolicyMachine,
    _sections,
    open_answer,
)

REPO = Path(__file__).resolve().parents[2]
#: T-0121's base (docs/tasks/T-0121.md, "Branch": code identical to 791dc54). A literal, never a moving ref.
BASE_SHA = "791dc54"

SECTIONS = ("machine", "hud", "prompts", "requests", "log", "records", "history")
SECRET = "SECRET-THOUGHT"

# --- the criterion-9 scenarios -----------------------------------------------------------------------------

#: (how the turn is asked, question, script). Tools off, one session, in order.
#: T-0124 re-scope R1 (docs/tasks/T-0124.md, "Owned files"; a spec change to criterion 9 for streams holding a special
#: literal): the ``which key`` (``Press <|tool_call> now. ``) and ``which key again`` (split) turns left this
#: comparison, because their 791dc54 result is the T-0124 bug (history holds the literal and the post-stop text);
#: their new behaviour is T-0124 criteria 1 and 4 (``test_session_literals_tools_off``). The ``hold on`` turn
#: (``Press <b`` + a thought_open=True commit + ``ack> now. ``) left too: it is T-0124 criterion 10's allowance row
#: (``test_t0124_differential.test_hold_allowance_row``). Its I15 premise below (``a draft`` reached the machine)
#: would then have no subject, so the turn is replaced, as R2 does for T-0117, by a no-literal I15 turn whose
#: 791dc54 result is unchanged (reported to the orchestrator: R1 as written removes the turn and keeps the premise).
TOOLS_OFF_TURNS: tuple[tuple[str, str, tuple[Commit, ...]], ...] = (
    ("typed", "and what does it mean", (answer("A typed answer here. "),)),
    ("ask", "compare them", (answer("x < y and y > z. "),)),
    ("ask", "did you think so", (answer("I thought so, and I still do. "),)),
    # I15: a thought_open=True commit between two voiced ones (no literal, no held tail)
    ("ask", "hold on", (answer("Hold "), open_answer("a draft"), answer("on. "))),
    (
        "ask",
        "think first",
        (thought("Let me think about it."), answer("First part. "), answer("Second part. ")),
    ),
    # criterion 10: a commit with empty raw text
    ("ask", "say it", (answer(""), answer("Said. "))),
    (
        "barge",
        "tell me more",
        (answer("This answer is long enough to still be playing at the cut. "),),
    ),
)
AFTER_BARGE = (answer("After the barge in. "),)

#: tools on, loop off: a closed thought in an answer (T-0117's path, which T-0121 does not touch)
TOOLS_ON_TURN = (answer(f"Sure, <|channel>thought {SECRET}<channel|>here it is. "),)

#: tools off: a partial open marker that never completes. ``canvas_idx`` 3, not the default 0, so a held-tail commit
#: carrying 0 (or any index other than its source commit's) is visible (N16)
WAIT_CANVAS = 3
WAIT_TURN = (answer("Wait <|chan", canvas_idx=WAIT_CANVAS),)


def _with_history(rig: Any, tmp: Path, session_id: str, log_path: Path) -> dict[str, Any]:
    out = _sections(rig, tmp, log_path, session_id)
    out["history"] = [h.model_dump(mode="json") for h in rig.state.history]
    return out


async def _until_turns(rig: Any, n: int, what: str) -> None:
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n,
        f"T-0121 criterion 9: {what} never ended",
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
                    "T-0121 criterion 9: the barge-in turn never reached the device",
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


async def _tools_on(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, machine_cls=_PolicyMachine) as rig:
        rig.llm.script = list(TOOLS_ON_TURN)
        await rig.ask("what does it say")
        await _until_turns(rig, 1, "the tools-on turn")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path)


async def _wait(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, kit=None, machine_cls=_PolicyMachine) as rig:
        rig.llm.script = list(WAIT_TURN)
        await rig.ask("wait")
        await _until_turns(rig, 1, "the Wait turn")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path)


SCENARIOS = {"tools_off": _tools_off, "tools_on": _tools_on, "wait": _wait}


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario session. Runs in-process on the working tree and in a subprocess on the 791dc54 tree."""
    return {name: await run(tmp / name) for name, run in SCENARIOS.items()}


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0121_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The 791dc54 run of every scenario, once per module."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0121_base")
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


def _assert_equal(expected: list[Any], got: list[Any], what: str) -> None:
    first = next((i for i, (a, b) in enumerate(zip(expected, got, strict=False)) if a != b), None)
    assert got == expected, (
        f"T-0121 {what} differs from {BASE_SHA} (lengths {len(expected)} vs {len(got)}; first difference at "
        f"{first}: base {json.dumps(expected[first] if first is not None else None)[:800]} / "
        f"now {json.dumps(got[first] if first is not None else None)[:800]})"
    )


def _machine_commits(sections: dict[str, Any], request_id: str = "t1-gen") -> list[dict[str, Any]]:
    return [
        step["event"]
        for step in sections["machine"]
        if step["event"].get("type") == "llm.commit"
        and step["event"].get("request_id") == request_id
    ]


# =====================================================================================================
# Criterion 9
# =====================================================================================================


async def test_no_marker_tools_off_equals_791dc54(base, tmp_path):
    """Criterion 9 (negative; mutations: hold every ``<``-led tail, delete other literals, route through
    ``_AnswerCleaner``, a note when ``chars == 0``, skip empty-raw commits, feed or reset on ``thought_open=True``
    commits): tools off, a typed turn, ``x < y and y > z.``, the word "thought", ``Hold `` + a
    ``thought_open=True`` commit + ``on.`` (I15; T-0124 R1), a ``thought`` segment then
    answers, an empty commit then an answer, and a barge-in turn give the same machine steps, HUD messages, prompts,
    request bodies, session-log records, ``turns.jsonl`` lines and history as 791dc54."""
    # T-0155 O-8: the persona sentence; T-0177 B-P1: the astra.vision.frame messages and their wire.out records
    now = without_vision_frames(observed(await _tools_off(tmp_path / "now")))
    for section in SECTIONS:
        _assert_equal(
            base["tools_off"][section], now[section], f"criterion 9: tools off {section!r}"
        )
    # premises: the scenarios ran what they say
    commits = [(e["text"], e["thought_open"], e["segment"]) for s in now["machine"] for e in [s["event"]]
               if e.get("type") == "llm.commit"]  # fmt: skip
    assert ("", False, "answer") in commits, (
        "premise (criterion 10): the empty-raw commit was submitted"
    )
    assert ("a draft", True, "answer") in commits, (
        "premise (I15): the thought_open=True commit reached the machine"
    )
    assert ("Let me think about it.", False, "thought") in commits, (
        "premise: the thought segment reached the machine"
    )
    stopped = [s for s in now["machine"] if s["event"].get("type") == "playback.stopped"]
    assert stopped, "premise: the barge-in stopped playback"


async def test_tools_on_marker_turn_equals_791dc54(base, tmp_path):
    """Criterion 9, tools on, loop off (mutation: route the tools-on path through the new cleaner): ``Sure,
    <|channel>thought SECRET-THOUGHT<channel|>here it is.`` holds a marker and is still identical to 791dc54 in every
    section, T-0117's note included (the tools-on path is untouched)."""
    # T-0155 O-8: the persona sentence; T-0177 B-P1: the astra.vision.frame messages and their wire.out records
    now = without_vision_frames(observed(await _tools_on(tmp_path / "now")))
    for section in SECTIONS:
        _assert_equal(base["tools_on"][section], now[section], f"criterion 9: tools on {section!r}")
    assert [r for r in now["log"] if r["kind"] == "note" and r["data"].get("chars") == 42], (
        "premise: T-0117's tools-on note ran"
    )


async def test_unfinished_partial_marker_voice_and_history_equal_791dc54(base, tmp_path):
    """Criterion 9, the separate row (guard too wide: the held ``<|chan`` must still be released at the end): tools
    off, ``Wait <|chan`` never completed. The ``TtsSpeak`` (source, text) sequence, history and ``turns.jsonl`` equal
    791dc54's, ``turns.jsonl``'s ``canvases`` excepted, which must be exactly the base's + 1 (ruling (a): it counts
    ``LlmCommit`` events, and the held tail is one more); the machine section differs at most in that the held
    ``<|chan`` arrives as its own commit before ``llm.done``."""
    now = await _wait(tmp_path / "now")
    old = base["wait"]

    def speaks(sections: dict[str, Any]) -> list[tuple[str, str]]:
        return [
            (c["source"], c["text"])
            for step in sections["machine"]
            for c in step["commands"]
            if c.get("type") == "tts.speak"
        ]

    _assert_equal(speaks(old), speaks(now), "criterion 9 (Wait <|chan): TtsSpeak (source, text)")
    _assert_equal(old["history"], now["history"], "criterion 9 (Wait <|chan): history")

    # Ruling (a), 2026-10-18 (docs/tasks/T-0121.md, Rulings): ``canvases`` counts ``LlmCommit`` events and
    # criterion 9 allows the held ``<|chan`` to arrive as its own commit, so that one field is excused here --
    # exactly as the machine section below is -- and every other ``turns.jsonl`` field stays compared.
    def without_canvases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if k != "canvases"} for r in records]

    assert len(now["records"]) == len(old["records"]) == 1, (old["records"], now["records"])
    assert all("canvases" in r for r in old["records"] + now["records"]), (
        "premise: every turns.jsonl line carries canvases"
    )
    _assert_equal(
        without_canvases(old["records"]),
        without_canvases(now["records"]),
        "criterion 9 (Wait <|chan): turns.jsonl (all fields but canvases)",
    )
    # the excuse is no wider than the one held-tail commit: exactly base + 1
    assert [r["canvases"] for r in now["records"]] == [r["canvases"] + 1 for r in old["records"]], (
        f"T-0121 criterion 9 (Wait <|chan): canvases may exceed the base only by the held-tail commit; "
        f"base {[r['canvases'] for r in old['records']]}, now {[r['canvases'] for r in now['records']]}"
    )
    assert speaks(now), "premise: something was voiced"

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
        "criterion 9 (Wait <|chan): machine events other than the answer commits",
    )
    texts = [
        [e["text"] for e in _machine_commits(sections) if voiced(e)] for sections in (old, now)
    ]
    assert "".join(texts[1]) == "".join(texts[0]) == "Wait <|chan", texts
    assert texts[1] in (texts[0], ["Wait ", "<|chan"]), (
        f"T-0121 criterion 9: the held <|chan may only arrive as its own commit; base {texts[0]}, now {texts[1]}"
    )
    # N16: the held tail is part of its source commit, so it carries that commit's ``canvas_idx``, as the base's
    # single commit does (the only allowed difference is the extra commit, not its index)
    idxs = [
        [e["canvas_idx"] for e in _machine_commits(sections) if voiced(e)]
        for sections in (old, now)
    ]
    assert idxs[0] == [WAIT_CANVAS], (
        f"premise: the base's one commit carries the scripted index; {idxs[0]}"
    )
    assert idxs[1] == [WAIT_CANVAS] * len(texts[1]), (
        f"T-0121 criterion 9 (N16): every voiced commit, the held tail included, carries its source commit's "
        f"canvas_idx {WAIT_CANVAS}, as at {BASE_SHA}; base {idxs[0]}, now {idxs[1]}"
    )
    events = [s["event"] for s in now["machine"]]
    done = next(
        i
        for i, e in enumerate(events)
        if e.get("type") == "llm.done" and e.get("request_id") == "t1-gen"
    )
    assert all(i < done for i, e in enumerate(events) if voiced(e)), (
        "every answer commit precedes llm.done"
    )


# =====================================================================================================
# Criterion 10
# =====================================================================================================


async def test_thought_open_and_non_answer_commits_reach_the_machine_as_sent(tmp_path):
    """Criterion 10 (I10, I15; mutation: feed ``thought_open=True`` commits to the cleaner): tools off, a
    ``thought_open=True`` answer commit and a ``thought`` segment commit, each holding a channel marker, reach the
    machine with their text and flags as sent, and the voiced commits around them are voiced whole: a cleaner fed
    the open marker would cut ``there.``."""
    draft = f"<|channel>draft {SECRET}"
    musing = f"<|channel>x<channel|> {SECRET}"
    script = [answer("Hello "), open_answer(draft), thought(musing), answer("there. ")]
    async with tool_rig(tmp_path, kit=None, script=script) as rig:
        await rig.ask("say hello")
        await rig.turn_done()
        commits = [
            (e.segment, e.text, e.thought_open)
            for e in rig.machine.events(E.LlmCommit)
            if e.request_id == "t1-gen"
        ]
        sources = rig.spoken_sources()
        stored = rig.state.history[-1].assistant_spoken
    assert ("answer", draft, True) in commits and ("thought", musing, False) in commits, (
        f"T-0121 criterion 10: the open and thought commits as sent; got {commits}"
    )
    voiced = "".join(t for seg, t, open_ in commits if seg == "answer" and not open_)
    assert voiced == "Hello there. ", commits
    assert sources == "Hello there. " and stored == "Hello there. ", (sources, stored)


#: Criterion 10's files, compared over T-0121's merge range (BASE holds them as at 791dc54).
UNOWNED = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/toolcall/router.py",
    "local_astra/toolcall/special_tokens.py",
    "local_astra/contracts",
    "tests/fakes",
)


#: Criterion 10's scope row is closed over T-0121's own merge (tests.md "pin both ends", T-0094; T-0110b ruling
#: 2026-10-18): M = 1248525, BASE = ``git merge-base M^1 M^2``, TIP = ``M^2``. The working tree is never read, so a
#: later task's legitimate edit of a fake (T-0110b's ``tests/fakes/audio.py``) cannot redden it.
T0121_MERGE = "12485258e5a867b56afa280c2d407f5cde78a769"
T0121_BASE = "1814db90fdfb710548f00f15add749af36cf5ddd"
T0121_TIP = "6076edf34bc76109eb1bbc251af193c07002f7b9"
#: main when T-0110b closed the row (ea8d02f); ``prove_range``'s "later" premise (T-0121 merged after ``gr.LATER``).
T0121_LATER = "ea8d02fc9eb3dd68d6facae15c22954ad3e137bc"


def test_unowned_files_equal_791dc54():
    """Criterion 10, last item: the machine, the gate, the client, the router, the special-token table, the contracts
    and the fakes are byte-identical (git blob hashes) across T-0121's own merge range ``T0121_BASE..T0121_TIP``
    (791dc54 is an ancestor of that BASE and equal to it on these paths), with no file added or removed under the
    two directories."""
    gr.prove_range(T0121_BASE, T0121_TIP, T0121_MERGE, later=T0121_LATER)
    assert gr.is_ancestor(BASE_SHA, T0121_BASE), (
        f"premise: {BASE_SHA} is an ancestor of {T0121_BASE}"
    )
    old = _ls_tree(T0121_BASE, UNOWNED)
    assert set(UNOWNED[:5]) <= set(old), (
        f"premise: {T0121_BASE} has the five files; got {sorted(old)[:8]}"
    )
    assert old == _ls_tree(BASE_SHA, UNOWNED), (
        f"premise: {BASE_SHA} and {T0121_BASE} hold the same unowned files"
    )
    now = _ls_tree(T0121_TIP, UNOWNED)
    changed = sorted(p for p in old.keys() & now.keys() if old[p] != now[p])
    added, removed = sorted(now.keys() - old.keys()), sorted(old.keys() - now.keys())
    assert not (changed or added or removed), (
        f"T-0121 criterion 10: unowned files differ in {T0121_BASE[:7]}..{T0121_TIP[:7]}: changed {changed}, "
        f"added {added}, removed {removed}"
    )
