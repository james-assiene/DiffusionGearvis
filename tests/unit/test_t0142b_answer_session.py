"""T-0142b A3-A5 (session side): with ``degeneracy_guard=True`` a request whose committed *answer* collapses into
repetition is cut off before the tripping commit reaches the machine; the request is aborted and the turn ends with a
HUD message and nothing new spoken (docs/tasks/T-0142b.md).

Every row runs T-0142's session rig over the fakes, imported, not copied (``test_t0142_session.guard_rig``: the live
kit over a ``FakeLiveBrowser``, ``dry_run`` or tools off; ``LateLlm`` keeps serving an aborted stream when asked).

**The interfaces these rows pin** (A3-A5; the readings are listed in the test-author's report):

- R-3: ``session.DEGENERATE_ANSWER_MESSAGE == "degenerate answer: request aborted"`` (A3.4, verbatim), a module
  constant distinct from T-0142's ``DEGENERATE_MESSAGE``.
- R-4: the answer-trip ``note`` has exactly T-0142's keys (``what, request_id, attempt_id, canvas_idx, unit,
  repeats, kind``) with ``what == "degenerate answer: request aborted"`` (A3.1, verbatim); ``request_id`` is the
  machine id, ``attempt_id`` the engine id the fake saw, ``canvas_idx`` that of the tripping commit, ``repeats`` at
  least ``MIN_REPEATS_ANSWER`` (6) and really present in the attempt's raw answer text so far (a check).
- R-5: exactly one ``abort`` of that engine id reaches the fake *before* ``E.LlmError`` is submitted (the order of
  ``backend.abort`` and ``Session.submit`` calls, recorded by T-0142's wrapper).
- R-6: "logged as dropped" is T-0142's R-6: a commit the session received from that attempt after the trip is in the
  log as its ``llm.commit`` record and is marked dropped (``"dropped": true`` on it, or a ``note`` whose ``what``
  contains ``dropped`` with the attempt's ``attempt_id``). Whether the session reads on after the abort is not
  pinned.
- R-7: the tripping commit is never submitted: no ``E.LlmCommit`` of that attempt's answer at or after the tripping
  canvas reaches the machine (A3.3, "the tripping commit is not submitted"), so A4 does not depend on the speech
  gate's timing.
- R-8: "the turn's ``hud.error`` carries the message": the machine emits ``C.HudError(message=DEGENERATE_ANSWER_
  MESSAGE)`` and the HUD publisher receives an error message carrying it (the existing ``turn_error`` path).
- R-9: "queued audio stops through the existing error path": after the trip, the audio fake is flushed after its last
  ``play`` (measured at BASE for a T-0142 *thought* trip after voiced answer text: the same holds there).
- R-10: "history holds only what the fake TTS reports as played": the turn's ``assistant_spoken`` is empty or a
  prefix of the ``played_text`` the audio fake reported through a ``PlaybackStopped``/``PlaybackDrained`` event, and
  never holds text of the tripping commit.

Every row fails at BASE 0248cfd on a named ``AssertionError`` (``DEGENERATE_ANSWER_MESSAGE`` missing), never on an
import error.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps) in live
and ``dry_run``, 1280x800 frames, ReGround off unless a row says otherwise, n = 1, deterministic.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0142_session as t42
from tests.unit._tool_turn_support import (
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    VALID_CLICK,
    answer,
    thought,
    tool,
)
from tests.unit.test_session_loop import POLICY
from tests.unit.test_session_reground import click
from tests.unit.test_session_reground import frames as jpeg_browser

GUARD = t42.GUARD
ANSWER_NOTE = "degenerate answer: request aborted"
THOUGHT_NOTE = t42.NOTE
NOTE_KEYS = t42.NOTE_KEYS
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "t0142b" / "answers.jsonl"
TOK = re.compile(r"\w+|[^\w\s]+")
#: S2#5's step 1, as the task writes it
STEP1_SENTENCE = "I will add the blue mug to the cart."
STEP1 = [answer(STEP1_SENTENCE), tool(VALID_CLICK)]
#: 6 x ". the" after a clean clause: trips at 6 (by hand), never at 8
COLLAPSE = "Let me check the cart. the. the. the. the. the. the."
LATE = "ZQX-LATE-T0142B your order is placed."


# --------------------------------------------------------------------------------------------- seams


def answer_message() -> str:
    s = importlib.import_module("local_astra.conductor.session")
    value = getattr(s, "DEGENERATE_ANSWER_MESSAGE", None)
    assert value == "degenerate answer: request aborted", (
        "T-0142b A3.4: session.DEGENERATE_ANSWER_MESSAGE = 'degenerate answer: request aborted' does not exist yet"
        f" (got {value!r})"
    )
    assert value != s.DEGENERATE_MESSAGE, "T-0142b A3.4: its own message, not T-0142's"
    return value


def collapsed_row() -> dict[str, Any]:
    rows = [json.loads(x) for x in FIXTURE.read_text().splitlines() if x.strip()]
    (row,) = [r for r in rows if r["class"] == "collapsed"]
    return row


S25 = collapsed_row()


def s25_step2() -> list[Any]:
    """S2#5's step 2 verbatim: its thought commits, then its 3 answer commits (canvas 1, 2, 3)."""
    out = [
        thought(c["text"], c["canvas_idx"], open_=bool(c["thought_open"])) for c in S25["thoughts"]
    ]
    out += [answer(c["text"], c["canvas_idx"]) for c in S25["answers"]]
    return out


def answer_notes(rig: Any) -> list[dict[str, Any]]:
    return [d for d in rig.kind("note") if d.get("what") == ANSWER_NOTE]


def thought_notes(rig: Any) -> list[dict[str, Any]]:
    return [d for d in rig.kind("note") if d.get("what") == THOUGHT_NOTE]


def speaks(rig: Any) -> list[str]:
    return [c.text for c in rig.machine.commands(C.TtsSpeak)]


def run_present(text: str, unit: str, repeats: int) -> bool:
    toks = TOK.findall(text.casefold())
    block = TOK.findall(unit.casefold()) * repeats
    return bool(block) and any(
        toks[i : i + len(block)] == block for i in range(len(toks) - len(block) + 1)
    )


def check_note(
    note: dict[str, Any], *, request_id: str, attempt_id: str, canvas_idx: int, text: str
) -> None:
    """R-4: the keys, the ids, and that the reported unit really repeats ``repeats`` (>= 6) times in ``text``."""
    assert set(note) == NOTE_KEYS, f"T-0142b A3.1 (R-4): the note's keys; got {sorted(note)}"
    assert (note["request_id"], note["attempt_id"], note["canvas_idx"]) == (
        request_id,
        attempt_id,
        canvas_idx,
    ), note
    assert note["kind"] in ("ngram", "char_run"), note
    if note["kind"] == "ngram":
        assert note["repeats"] >= 6 and run_present(text, note["unit"], note["repeats"]), note
    else:
        assert note["repeats"] >= 64 and note["unit"] * note["repeats"] in text, note


def machine_commits(rig: Any, request_id: str, segment: str | None = None) -> list[Any]:
    return [
        e
        for e in rig.machine.events(E.LlmCommit)
        if e.request_id == request_id and (segment is None or e.segment == segment)
    ]


def hud_errors(rig: Any) -> tuple[list[str], list[str]]:
    """(machine ``HudError`` messages, HUD-published messages that carry a ``message`` and are errors)."""
    machine = [c.message for c in rig.machine.commands(C.HudError)]
    published = [
        str(getattr(m, "message", ""))
        for m in rig.published.messages
        if "error" in str(getattr(m, "type", "")).lower()
    ]
    return machine, published


def abort_before_error(order: list[Any], attempt: str, request_id: str) -> None:
    """R-5: exactly one abort of ``attempt`` precedes the ``LlmError`` for ``request_id``."""
    assert ("submit", "LlmError", request_id) in order, (
        f"no LlmError submitted for {request_id}: {order}"
    )
    error_at = order.index(("submit", "LlmError", request_id))
    before = [x for x in order[:error_at] if x == ("abort", attempt)]
    assert len(before) == 1, f"T-0142b A3.2 (R-5): one abort of {attempt} before the error; {order}"


# =====================================================================================================
# A3/A4: the logged collapse
# =====================================================================================================


async def test_logged_collapse_voices_nothing_of_it(tmp_path: Path) -> None:
    """A3/A4 (adversarial; the regression this task exists for; mutants: judging after submission, ``K = 8``): S2#5
    replayed (step 1's sentence and call, then step 2's thought and its 3 answer commits verbatim). The turn speaks
    exactly step 1's sentence and nothing with ``Wait`` or ``the.``; one answer note names the attempt and canvas 1;
    one abort of that engine id precedes ``LlmError(t1-gen-s2, DEGENERATE_ANSWER_MESSAGE)``; ``tool_loop_end ==
    "degenerate"``; the turn ends before ``llm_timeout_ms``; the turn's ``hud.error`` carries the message.

    T-0179 B-11 (supersedes H-T142b-1's "no retry"): nothing of the answer was submitted, so the step is retried once;
    S2#5's step 2 is scripted for both attempts, each trips on its first answer commit (one answer note and one abort
    each, both before the ``LlmError``), and the error and the ``hud.error`` carry ``DEGENERATE_RETRIED_MESSAGE``."""
    answer_message()
    message = t42.retried_message()
    assert S25["attempt_id"] == "5008519a4296-t1-gen-s2-4", "premise: the fixture's collapse"
    assert [c["canvas_idx"] for c in S25["answers"]] == [1, 2, 3], "premise: answer canvases"
    async with t42.guard_rig(tmp_path, script=STEP1) as rig:
        await t42.run(rig, s25_step2(), s25_step2())
        extras = dict(rig.record("t1").extras)
        said, order = speaks(rig), list(rig.order)
        sources = [c.source for c in rig.machine.commands(C.TtsSpeak)]
        errs = t42.errors(rig, "t1-gen-s2")
        answers_in = machine_commits(rig, "t1-gen-s2", "answer")
        machine_err, published_err = hud_errors(rig)
    both = t42.attempts(rig, "t1-gen-s2")
    assert len(both) == 2, f"T-0179 B-11: the tripped step and its one retry; {both}"
    step2_at = [t for kind, rid, t in rig.llm.calls if kind == "generate" and rid == both[0]]
    assert said == [STEP1_SENTENCE], (
        f"T-0142b A4: the turn speaks exactly step 1's sentence; got {said}"
    )
    assert not [s for s in said + sources if "Wait" in s or "the." in s], said
    assert answers_in == [], (
        f"T-0142b A3.3 (R-7): answer commits of the tripped attempt reached the machine: {answers_in}"
    )
    found = answer_notes(rig)
    assert len(found) == 2, f"T-0142b A3.1: one answer-trip note per attempt; got {found}"
    for note, attempt in zip(found, both, strict=True):
        check_note(
            note,
            request_id="t1-gen-s2",
            attempt_id=attempt,
            canvas_idx=1,
            text=S25["answers"][0]["text"],
        )
        abort_before_error(order, attempt, "t1-gen-s2")
    assert thought_notes(rig) == [], "premise: S2#5's thought is clean"
    assert [e.message for e in errs] == [message], (
        f"T-0142b A3.4 / T-0179 B-11: one LlmError with DEGENERATE_RETRIED_MESSAGE; {errs}"
    )
    assert t42.dones(rig, "t1-gen-s2") == [], (
        "T-0142b A3.3: no llm.done of the tripped attempt reaches the machine"
    )
    assert extras.get("tool_loop_end") == "degenerate", extras.get("tool_loop_end")
    assert step2_at, f"premise: step 2 was requested; {rig.llm.calls}"
    assert errs[0].t_ms - step2_at[-1] < POLICY.llm_timeout_ms, (
        "T-0142b: the turn ends before the LLM timeout"
    )
    assert message in machine_err, f"T-0142b A4 (R-8): the machine's hud.error; {machine_err}"
    assert message in published_err, (
        f"T-0142b A4 (R-8): the HUD error carries the message; {published_err}"
    )


async def test_late_commits_dropped(tmp_path: Path) -> None:
    """A3.3 (mutant: drop only answer commits after the trip): an engine that keeps serving the aborted stream delivers
    an answer, a thought and a call after the tripping answer; none reaches the machine, TTS, history, the HUD or the
    executor, no ``llm.done`` of the attempt reaches the machine, and each one received is logged as dropped (R-6).

    T-0179 B-11: the step is retried once; the same late-serving collapse is scripted for both attempts, so only the
    clean thought before each trip reaches the machine (once per attempt, B-8 reading accepted) and every never-voiced
    and dropped assertion holds over both attempts."""
    answer_message()
    llm = t42.LateLlm(late=True)
    step2 = [
        thought("I check the cart.", 0),
        answer(COLLAPSE, 1),
        answer(LATE, 2),
        thought(" and then more planning.", 2, open_=True),
        tool(tsl.CLICK3, canvas_idx=2),
    ]
    async with t42.guard_rig(tmp_path, llm=llm, script=STEP1) as rig:
        await t42.run(rig, step2, step2, [answer(tsl.S3)])
        extras = dict(rig.record("t1").extras)
        said, hist, hud = " ".join(speaks(rig)), t42.history_text(rig), t42.hud_dump(rig)
        reached = machine_commits(rig, "t1-gen-s2")
        executed = [i.call_id for i in rig.browser.invocations]
    both = t42.attempts(rig, "t1-gen-s2")
    assert len(both) == 2, f"T-0179 B-11: the tripped step and its one retry; {both}"
    assert len(answer_notes(rig)) == 2, "T-0142b A3: each attempt tripped on its answer"
    assert [(e.segment, e.text) for e in reached] == [("thought", "I check the cart.")] * 2, (
        f"T-0142b A3.3: only the commit before each trip reached the machine; {[(e.segment, e.text) for e in reached]}"
    )
    for text in ("ZQX-LATE", "planning", "Let me check"):
        assert text not in said and text not in hist, f"T-0142b A4: {text!r} voiced or stored"
    assert "ZQX-LATE" not in hud and "planning" not in hud, (
        "T-0142b A3.3: a late commit reached the HUD"
    )
    assert t42.dones(rig, "t1-gen-s2") == []
    assert executed == ["t1-call"], f"T-0142b A3.3: the late call executed; {executed}"
    assert extras.get("tool_loop_end") == "degenerate", extras.get("tool_loop_end")
    for rid, _segment, text in llm.after_abort:
        if rid not in both:
            continue
        assert t42.logged_commit(rig, rid, text), (
            f"T-0142b A3.3: a received commit is logged: {text[:40]!r}"
        )
        assert t42.dropped_logged(rig, rid, text), (
            f"T-0142b A3.3 (R-6): not logged as dropped: {text[:40]!r}"
        )


async def test_earlier_commit_voiced_then_stopped(tmp_path: Path) -> None:
    """A4 (R-9, R-10): tools off, answer commit 0 is clean prose and is voiced before commit 1 arrives; commit 1 alone
    completes a trip. Commit 0's sentences are voiced, commit 1's never; the turn ends on the answer message; queued
    audio is flushed after the last play; history holds only what the audio fake reported as played."""
    message = answer_message()
    c0 = "Your cart has the blue mug. It costs nine dollars. The red mug is eight."
    c1 = " blah. blah. blah. blah. blah. blah."
    async with t42.guard_rig(tmp_path, mode="off", script=[answer(c0, 0), answer(c1, 1)]) as rig:
        rig.llm.hold_after(1)
        await rig.ask("what is in my cart")
        await rig.until(
            lambda: bool(speaks(rig)), "premise: commit 0 was never voiced", limit_ms=10_000
        )
        await rig.advance(1500)
        before = list(speaks(rig))
        rig.llm.release()
        await rig.turn_done(limit_ms=30_000)
        await rig.advance(8000)
        said = speaks(rig)
        calls = list(rig.audio.calls)
        errs = t42.errors(rig, "t1-gen")
        played = [
            e.played_text
            for e in [
                *rig.machine.events(E.PlaybackStopped),
                *rig.machine.events(E.PlaybackDrained),
            ]
        ]
        hist = [h.assistant_spoken for h in rig.state.history]
    assert before and before[0] == "Your cart has the blue mug.", (
        f"premise: commit 0 voiced first; {before}"
    )
    assert all(s in c0 for s in said), f"T-0142b A4: only commit 0's text is voiced; {said}"
    assert not [s for s in said if "blah" in s], said
    assert [e.message for e in errs] == [message], errs
    (note,) = answer_notes(rig)
    assert note["canvas_idx"] == 1 and note["request_id"] == "t1-gen", note
    plays = [i for i, c in enumerate(calls) if c == "play"]
    flushes = [i for i, c in enumerate(calls) if c == "flush"]
    assert flushes and plays and flushes[-1] > plays[-1], (
        f"T-0142b A4 (R-9): queued audio is cut; {calls}"
    )
    assert len(hist) == 1, hist
    spoken = hist[0]
    assert "blah" not in spoken, f"T-0142b A4: history holds the tripping commit; {spoken!r}"
    assert spoken == "" or any(p.startswith(spoken) for p in played), (
        f"T-0142b A4 (R-10): history {spoken!r} is not what the audio reported as played {played}"
    )


async def test_thought_and_answer_separate_thought_first(tmp_path: Path) -> None:
    """A3 (adversarial; mutation 5, one shared accumulator judged on the answer): a thought ending with 5 ``the`` and
    an answer opening with 3 are 8 together, but each segment is judged on its own (5 < 8 for thought, 3 < 6 for
    answer): nothing trips."""
    answer_message()
    step2 = [
        thought("I check the page and" + " the" * 5, 0, open_=True),
        answer(" the" * 3 + " cart is ready.", 1),
    ]
    async with t42.guard_rig(tmp_path, script=STEP1) as rig:
        await t42.run(rig, step2)
        extras = dict(rig.record("t1").extras)
    assert answer_notes(rig) == [] and thought_notes(rig) == [], rig.kind("note")
    assert extras.get("tool_loop_end") == "final" and extras.get(GUARD) == {
        "on": True,
        "tripped": 0,
    }, extras


async def test_thought_and_answer_separate_answer_first(tmp_path: Path) -> None:
    """A3 (adversarial; mutation 5, one shared accumulator judged on the thought): an answer ending with 5 ``ha`` and a
    later thought opening with 3 are 8 together (the thought rule), but each segment is judged on its own: nothing
    trips."""
    answer_message()
    step2 = [answer("Done, ha" + " ha" * 4, 0), thought(" ha" * 3 + " next.", 1, open_=True)]
    async with t42.guard_rig(tmp_path, script=STEP1) as rig:
        await t42.run(rig, step2)
        extras = dict(rig.record("t1").extras)
    assert answer_notes(rig) == [] and thought_notes(rig) == [], rig.kind("note")
    assert extras.get(GUARD) == {"on": True, "tripped": 0}, extras


async def test_straddle_trips_in_session(tmp_path: Path) -> None:
    """A3 (adversarial; mutation 2, each commit judged alone): 3 repeats closing answer canvas 0 and 3 opening canvas 1
    trip at canvas 1 on the attempt's concatenated raw answer text; canvas 1 is never voiced."""
    message = answer_message()
    c0 = "Let me check the cart. the. the. the."
    c1 = " the. the. the. Then I click."
    async with t42.guard_rig(tmp_path, script=STEP1) as rig:
        await t42.run(rig, [answer(c0, 0), answer(c1, 1), tool(tsl.CLICK3, canvas_idx=1)])
        extras = dict(rig.record("t1").extras)
        executed = [i.call_id for i in rig.browser.invocations]
        said = " ".join(speaks(rig))
    (attempt,) = t42.attempts(rig, "t1-gen-s2")
    found = answer_notes(rig)
    assert len(found) == 1, f"T-0142b A3: the straddling collapse trips; notes {found}"
    check_note(found[0], request_id="t1-gen-s2", attempt_id=attempt, canvas_idx=1, text=c0 + c1)
    assert found[0]["repeats"] == 6, found[0]
    assert "Then I click" not in said, said
    assert [e.message for e in t42.errors(rig, "t1-gen-s2")] == [message]
    assert executed == ["t1-call"] and extras.get("tool_loop_end") == "degenerate", (
        executed,
        extras,
    )


async def _off_trip(
    tmp_path: Path, script: list[Any]
) -> tuple[Any, list[Any], list[Any], list[str]]:
    """Tools off, one turn over ``script``: (rig, the abort/submit order, ``t1-gen``'s errors, its engine ids).

    T-0179 B-11: every engine id of ``t1-gen`` is returned (the fake replays ``script`` for a degenerate retry, so a
    trip with nothing of the answer submitted gives two); each row states how many it expects."""
    async with t42.guard_rig(tmp_path, mode="off", script=script) as rig:
        await rig.ask("what is in my cart")
        await rig.turn_done(limit_ms=30_000)
        await rig.advance(300)
        order = list(rig.order)
        errs = t42.errors(rig, "t1-gen")
    return rig, order, errs, t42.attempts(rig, "t1-gen")


def _retried_twice(
    rig: Any, order: list[Any], errs: list[Any], both: list[str], *, canvas_idx: int, text: str
) -> None:
    """T-0179 B-11: the tripped generate and its one retry both trip the same way (a note and an abort each, before
    the ``LlmError``), which carries ``DEGENERATE_RETRIED_MESSAGE``."""
    assert len(both) == 2, f"T-0179 B-11: the tripped generate and its one retry; {both}"
    found = answer_notes(rig)
    assert len(found) == 2, f"T-0142b: each attempt's answer commit is judged; answer notes {found}"
    for note, attempt in zip(found, both, strict=True):
        check_note(note, request_id="t1-gen", attempt_id=attempt, canvas_idx=canvas_idx, text=text)
        abort_before_error(order, attempt, "t1-gen")
    assert [e.message for e in errs] == [t42.retried_message()], errs


async def test_thought_open_answer_commit_judged(tmp_path: Path) -> None:
    """Orchestrator ruling (2026-10-23: "every answer-segment commit's raw text is judged, including one with
    ``thought_open=True``"; mutant: the answer predicate gains ``and not commit.thought_open``): after an open thought,
    an answer commit flagged ``thought_open=True`` whose raw text is ``"Ok."`` plus 8 x ``" the."`` (8 back-to-back
    ``. the`` by hand, >= 6) trips: one answer note at its canvas, one abort before the ``LlmError``.

    T-0179 B-11: nothing of the answer was submitted, so the generate is retried once; the replayed script trips again
    at the same commit, and the ``LlmError`` carries ``DEGENERATE_RETRIED_MESSAGE``."""
    answer_message()
    text = "Ok." + " the." * 8
    script = [
        thought("Let me look at the cart", 0, open_=True),
        answer(text, 0).model_copy(update={"thought_open": True}),
    ]
    assert script[1].thought_open and script[1].segment == "answer", "premise: the commit's flags"
    rig, order, errs, both = await _off_trip(tmp_path, script)
    _retried_twice(rig, order, errs, both, canvas_idx=0, text=text)


async def test_answer_that_cleans_to_nothing_judged(tmp_path: Path) -> None:
    """Orchestrator ruling ("including one that cleans to nothing") and the implementation note "the cleaners must not
    change what is judged" (mutant: the answer judged on the cleaned text, still before submit): tools off, the raw
    answer commit ``"<|channel>thought\\n" + " the." * 8 + "<channel|>"`` is a closed thought block, which the T-0121
    cleaner removes entirely, so the cleaned text is empty; its raw text has 7 back-to-back ``the .`` (the 8th ``.``
    fuses with ``<`` as ``.<``; by hand), >= 6, so it trips. Nothing of it is voiced.

    T-0179 B-11: nothing was submitted, so the generate is retried once and the replayed script trips again (judged on
    its raw text both times); the ``LlmError`` carries ``DEGENERATE_RETRIED_MESSAGE``."""
    answer_message()
    text = "<|channel>thought\n" + " the." * 8 + "<channel|>"
    rig, order, errs, both = await _off_trip(tmp_path, [answer(text, 0)])
    _retried_twice(rig, order, errs, both, canvas_idx=0, text=text)
    assert not [s for s in speaks(rig) if "the." in s], speaks(rig)


async def test_concatenated_with_no_separator(tmp_path: Path) -> None:
    """A3 ("concatenated across canvases with no separator"; mutant: commits joined with ``' '``): the unit is split
    mid-token across two commits. Joined raw, ``"Ok. the. the. the. th" + "e. the. the. the. the."`` is ``"Ok."`` plus
    8 x ``" the."``: 8 back-to-back ``. the`` (>= 6), trip at canvas 1. Joined with a space it reads ``... th e. ...``:
    the run breaks at ``th e``, leaving at most 3 before and 4 after (< 6, by hand), and each commit alone has 3 and 4:
    no trip."""
    message = answer_message()
    c0, c1 = "Ok. the. the. the. th", "e. the. the. the. the."
    assert c0 + c1 == "Ok." + " the." * 8, "premise: the joined text"
    rig, order, errs, ids = await _off_trip(tmp_path, [answer(c0, 0), answer(c1, 1)])
    # T-0179 B-11: commit 0 (clean) was submitted before the trip, so no retry: one request, today's message
    (attempt,) = ids
    found = answer_notes(rig)
    assert len(found) == 1, f"T-0142b A3: the raw join with no separator trips; notes {found}"
    check_note(found[0], request_id="t1-gen", attempt_id=attempt, canvas_idx=1, text=c0 + c1)
    abort_before_error(order, attempt, "t1-gen")
    assert [e.message for e in errs] == [message], errs


async def test_tools_off_answer_trips(tmp_path: Path) -> None:
    """A3 (tools off, the conversational turn): a collapsing answer gives the A3 sequence for ``t1-gen`` (one note,
    one abort before ``LlmError(t1-gen, DEGENERATE_ANSWER_MESSAGE)``), no ``tool_loop_end`` key, nothing of the
    tripping commit or later voiced or stored, and ``tripped: 1`` on the turn line.

    T-0179 B-11: nothing of the answer was submitted, so the generate is retried once; the replayed script trips again,
    giving two notes and two aborts before the one ``LlmError``, ``tripped: 2``, and ``DEGENERATE_RETRIED_MESSAGE`` on
    the error and the ``hud.error``. Every never-voiced assertion stays."""
    answer_message()
    message = t42.retried_message()
    script = [
        thought("Let me look.", 0),
        answer("Here is the page. " + COLLAPSE, 0),
        answer(LATE, 1),
    ]
    async with t42.guard_rig(tmp_path, mode="off", script=script) as rig:
        await rig.ask("what is on this screen")
        await rig.turn_done(limit_ms=30_000)
        await rig.advance(300)
        extras = dict(rig.record("t1").extras)
        said, hist, order = " ".join(speaks(rig)), t42.history_text(rig), list(rig.order)
        machine_err, published_err = hud_errors(rig)
    both = t42.attempts(rig, "t1-gen")
    assert len(both) == 2, f"T-0179 B-11: the tripped generate and its one retry; {both}"
    found = answer_notes(rig)
    assert len(found) == 2, found
    for note, attempt in zip(found, both, strict=True):
        check_note(
            note,
            request_id="t1-gen",
            attempt_id=attempt,
            canvas_idx=0,
            text="Here is the page. " + COLLAPSE,
        )
        abort_before_error(order, attempt, "t1-gen")
    assert [e.message for e in t42.errors(rig, "t1-gen")] == [message]
    assert machine_commits(rig, "t1-gen", "answer") == [], (
        "T-0142b A3.3 (R-7): the tripping commit was submitted"
    )
    assert "tool_loop_end" not in extras, extras
    assert extras.get(GUARD) == {"on": True, "tripped": 2}, extras.get(GUARD)
    for text in ("Here is the page", "ZQX-LATE", "the. the."):
        assert text not in said and text not in hist, f"T-0142b A4: {text!r} voiced or stored"
    assert message in machine_err and message in published_err, (machine_err, published_err)


async def test_drafts_never_judged(tmp_path: Path) -> None:
    """A8 (adversarial; judging anything but answer commits, or thought at 6): clean answer commits beside a thought
    with 7 back-to-back ``the`` and a call whose draft text repeats (``la`` x 20) never trip; the turn runs to
    ``final`` and the turn line says ``tripped: 0``. (The session takes no denoising-tap input: the draft channel it
    can see is the ``tool`` commit text.)"""
    answer_message()
    q = '<|"|>'
    typed = "la " * 20
    type_call = f"<|tool_call>call:type_at{{press_enter:false,text:{q}{typed.strip()}{q},x:305,y:448}}<tool_call|>"
    step2 = [
        thought("I check the field" + " the" * 7 + " first.", 0),
        answer("Typing it now.", 0),
        tool(type_call),
    ]
    step3 = [thought("Done typing.", 0), answer("All set.", 0)]
    async with t42.guard_rig(tmp_path, script=STEP1) as rig:
        await t42.run(rig, step2, step3)
        extras = dict(rig.record("t1").extras)
        executed = [i.call_id for i in rig.browser.invocations]
    assert answer_notes(rig) == [] and thought_notes(rig) == [], rig.kind("note")
    assert executed == ["t1-call", "t1-call-s2"], executed
    assert extras.get("tool_loop_end") == "final", extras.get("tool_loop_end")
    assert extras.get(GUARD) == {"on": True, "tripped": 0}, extras.get(GUARD)


async def test_next_turn_runs_after_answer_trip(tmp_path: Path) -> None:
    """docs/02 §4 (one live generate plus one prefill; ``llm.abort`` idempotent): after an answer trip the engine ids
    are released and the next turn is answered and voiced.

    T-0179 B-11: the tripped generate is retried once and the replayed script trips again (two answer notes), and the
    next turn still runs (the retry's engine id released as well)."""
    answer_message()
    async with t42.guard_rig(tmp_path, mode="off", script=[answer(COLLAPSE, 0)]) as rig:
        await rig.ask("what is in my cart")
        await rig.turn_done(limit_ms=30_000)
        rig.llm.script = [answer("Second answer here.")]
        await rig.ask("and now")
        await rig.until(
            lambda: len(rig.lines()) >= 2, "the turn after a trip never ended", limit_ms=30_000
        )
        await rig.advance(600)
        said = " ".join(speaks(rig))
        t42.assert_concurrency(rig)
    assert len(answer_notes(rig)) == 2, answer_notes(rig)
    assert len(t42.attempts(rig, "t1-gen")) == 2, t42.attempts(rig, "t1-gen")
    assert "Second answer" in said, said
    assert rig.record("t2").extras.get(GUARD) == {"on": True, "tripped": 0}


# =====================================================================================================
# Out of scope: ReGround pass 2, tool_repair
# =====================================================================================================

PASS2_ANSWER_COLLAPSE = [
    thought("The button is at the top.", 0),
    answer("Here." + " the." * 20, 0),
    tool(click(900, 100), canvas_idx=1),
]


async def test_out_of_scope_pass2(tmp_path: Path) -> None:
    """A3 out of scope (mutation 8, a pass-2 answer trip ending the turn): a ReGround pass 2 whose *answer* text
    collapses gives, with the guard on, the outcome and records it gives with the guard off (as at BASE, where the
    guard judges thought only): pass 2 regrounds to (900, 100), the turn goes on; no answer note, no abort of the
    pass-2 id."""
    answer_message()
    off, off_extras = await t42._pass2_turn(tmp_path / "off", False, PASS2_ANSWER_COLLAPSE)
    on, on_extras = await t42._pass2_turn(tmp_path / "on", True, PASS2_ANSWER_COLLAPSE)
    want, got = t42._reground_view(off, off_extras), t42._reground_view(on, on_extras)
    assert [r["outcome"] for r in want["regrounds"]] == ["ok"], (
        f"premise: pass 2 regrounds; {want['regrounds']}"
    )
    assert got == want, f"T-0142b A3: a pass-2 answer is out of scope\noff {want}\non  {got}"
    assert answer_notes(on) == [] and thought_notes(on) == []
    (p2,) = [d for d in on.kind("llm.request") if d.get("phase") == "reground"]
    assert p2["attempt_id"] not in on.llm.aborted, on.llm.aborted
    assert on_extras.get(GUARD) == {"on": True, "tripped": 0}, on_extras.get(GUARD)


def _repair_view(rig: Any, extras: dict[str, Any]) -> dict[str, Any]:
    return {
        "requests": [(d.get("phase"), d.get("request_id")) for d in rig.kind("llm.request")],
        "tool_calls": [(r.get("call_id"), r.get("status")) for r in rig.tool_records()],
        "executed": [(i.call_id, dict(i.arguments)) for i in rig.executor.invocations],
        "spoken": [c.source for c in rig.machine.commands(C.TtsSpeak)],
        "errors": [(e.request_id, e.message) for e in rig.machine.events(E.LlmError)],
        "aborted": [a.rsplit("-", 2)[-2:] for a in rig.llm.aborted],
        "turn": {k: extras.get(k) for k in ("tool_loop_end", "tool_steps", "tool_steps_executed")},
        "notes": [d.get("what") for d in rig.kind("note")],
    }


async def _repair_turn(tmp_path: Path, guard: bool) -> tuple[Any, dict[str, Any]]:
    async with t42.guard_rig(
        tmp_path, mode="dry_run", guard=guard, script=[answer(tsl.S1), tool(REPAIRABLE_CLICK)]
    ) as rig:
        rig.llm.script_continuation(answer("Fixed." + " the." * 20, 0), tool(REPAIRED_CLICK))
        await t42.run(rig, [answer(tsl.S3)])
        extras = dict(rig.record("t1").extras)
    return rig, extras


async def test_out_of_scope_repair(tmp_path: Path) -> None:
    """A3 out of scope: a ``tool_repair`` stream carrying collapsing answer text gives, with the guard on, the outcome
    and records it gives with the guard off: the repair runs, no answer note, nothing aborted for it."""
    answer_message()
    off, off_extras = await _repair_turn(tmp_path / "off", False)
    on, on_extras = await _repair_turn(tmp_path / "on", True)
    want, got = _repair_view(off, off_extras), _repair_view(on, on_extras)
    assert ("tool_repair", "t1-repair") in want["requests"], (
        f"premise: a repair ran; {want['requests']}"
    )
    assert got == want, f"T-0142b A3: a repair's answer is out of scope\noff {want}\non  {got}"
    assert answer_notes(on) == []
    assert on_extras.get(GUARD) == {"on": True, "tripped": 0}, on_extras.get(GUARD)


# =====================================================================================================
# A5: the turn line
# =====================================================================================================


@pytest.mark.parametrize("guard", [True, False])
async def test_turn_line_shape(tmp_path: Path, guard: bool) -> None:
    """A5 (R-a): after an answer trip the turn line's dict is exactly ``{"on": true, "tripped": 1}`` (no extra key);
    with the guard off the same turn has no key, no note and no answer-message error (A8)."""
    message = answer_message()
    async with t42.guard_rig(tmp_path, guard=guard, script=STEP1) as rig:
        await t42.run(rig, [answer(COLLAPSE, 0)], [answer(tsl.S3)])
        extras = dict(rig.record("t1").extras)
    if guard:
        assert extras.get(GUARD) == {"on": True, "tripped": 1}, (
            f"T-0142b A5: got {extras.get(GUARD)}"
        )
        assert list(extras[GUARD]) == ["on", "tripped"], extras[GUARD]
    else:
        assert GUARD not in extras, extras
        assert answer_notes(rig) == []
        assert [e for e in rig.machine.events(E.LlmError) if e.message == message] == []


async def test_tripped_counts_both_segments(tmp_path: Path) -> None:
    """A5 ("``tripped`` counts the requests that tripped on either segment"): a pass-2 *thought* trip (T-0142's
    fallback) and then a step-2 *answer* trip in one turn are each counted; the answer trip ends the turn.

    T-0179 B-11: pass 2 is never retried, the step is retried once and trips again on its answer (scripted for both
    attempts): one thought note, two answer notes, ``tripped: 3``."""
    answer_message()
    async with t42.guard_rig(tmp_path, reground=True, browser=jpeg_browser()) as rig:
        rig.llm.script_pass2([thought(t42.COLLAPSE, 0, open_=True), tool(click(900, 100))])
        await t42.run(rig, [answer(COLLAPSE, 0)], [answer(COLLAPSE, 0)])
        extras = dict(rig.record("t1").extras)
    assert len(thought_notes(rig)) == 1 and len(answer_notes(rig)) == 2, rig.kind("note")
    assert extras.get(GUARD) == {"on": True, "tripped": 3}, extras.get(GUARD)
    assert extras.get("tool_loop_end") == "degenerate", extras.get("tool_loop_end")
