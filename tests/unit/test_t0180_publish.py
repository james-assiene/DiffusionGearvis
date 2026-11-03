"""T-0180 group B: the session publishes each request attempt's committed thought as ``astra.thought``
(docs/tasks/T-0180.md B1-B7; B8 is ``test_t0180_differential.py``, B9 the narrowed leak rows).

Every row runs the session over the fakes (``_tool_turn_support.tool_rig``: tools off, ``dry_run`` or ``live``; T-0142's
``guard_rig``, T-0179's ``retry_rig`` and T-0172's ``reask_rig``, imported, not copied) with the typed HUD ``Recorder``
as publisher, and reads two things only: the ``astra.thought`` messages published, and (the B1 oracle) the session
log's ``llm.commit`` records with ``segment == "thought"``. Expected texts, steps, kinds and ends are also written out
by hand from each row's script.

**Readings where the task is silent** (each listed in the test-author's report):

- ``request_id`` is the ``llm.commit`` record's ``request_id`` (the machine id: ``t1-gen``, ``t1-gen-s<k>``, the repair
  id, the pass-2 key) and ``attempt_id`` the record's ``attempt_id`` (the engine id); ``turn_id`` is ``t<seq>``.
- "One end per attempt" (B3): an attempt's messages are zero or more ``open`` ones, then exactly one final one, last.
  Every message's ``text`` is the concatenation of that attempt's first ``k >= 1`` thought commits (capped), so no
  message carries text the log does not hold, and the last carries all of it.
- ``step`` of a repair and of a ReGround pass 2 is the loop step whose call they serve (C3's ``Step 3 · repair``,
  ``Step 2 · ReGround``); a re-ask and a T-0179 retry carry their step's number.
- B4 beats B1 when the session logs a commit it received after the abort: the published text stops at the abort.
- ``omitted_chars`` is 0 everywhere but the cap row.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, n = 1 per row, deterministic.
No measurement.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.protocol import THOUGHT_MAX_CHARS, AstraThought
from local_astra.contracts.turn import TurnState
from tests.fakes.search import FakeSearch
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0142_session as t42
from tests.unit import test_t0172_session_reask as t72
from tests.unit import test_t0179_degenerate_retry as t79
from tests.unit._tool_turn_support import (
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    VALID_CLICK,
    answer,
    thought,
    tool,
    tool_rig,
)
from tests.unit.test_session_reground import click, regrounds
from tests.unit.test_session_reground import frames as jpeg_browser
from tests.unit.test_session_reset import do_reset, one_cleared

THOUGHT = "astra.thought"
FINAL_ENDS = frozenset({"done", "stopped_repeating", "interrupted", "error"})
QUESTION_OFF = "what is on this screen"


# --------------------------------------------------------------------------------------------- reading the rig


def published_thoughts(rig: Any) -> list[Any]:
    return [m for m in rig.published.messages if getattr(m, "type", None) == THOUGHT]


def by_attempt(messages: Sequence[Any]) -> dict[str, list[Any]]:
    """Published thoughts grouped by ``attempt_id``, in first-seen order."""
    out: dict[str, list[Any]] = {}
    for m in messages:
        out.setdefault(m.attempt_id, []).append(m)
    return out


def logged_thoughts(rig: Any) -> dict[str, dict[str, Any]]:
    """The B1 oracle: per attempt (first-seen order), the ``request_id`` and the list of thought commit texts, read
    from the session log's ``llm.commit`` records (``rig.log_records``, filled after ``aclose``)."""
    out: dict[str, dict[str, Any]] = {}
    for r in rig.log_records:
        d = r.get("data") or {}
        if r["kind"] == "llm.commit" and d.get("segment") == "thought":
            got = out.setdefault(d["attempt_id"], {"request_id": d["request_id"], "texts": []})
            got["texts"].append(d["text"])
    return out


def dump(messages: Sequence[Any]) -> str:
    return json.dumps([m.model_dump(mode="json") for m in messages])


def non_thought_dump(rig: Any) -> str:
    return dump([m for m in rig.published.messages if getattr(m, "type", None) != THOUGHT])


def history_dump(rig: Any) -> str:
    return json.dumps([h.model_dump(mode="json") for h in rig.state.history])


def assert_blocks(
    rig: Any,
    want: Sequence[tuple[int, str, str, str]],
    *,
    turn_id: str = "t1",
    oracle: bool = True,
    what: str = "",
) -> list[str]:
    """``want``: one ``(step, kind, text, end)`` per attempt that committed a thought, in first-seen order, written
    by hand. Checks B1/B3/B6 on every message of ``turn_id``; returns the attempt ids. ``oracle``: also check each
    attempt's text and request id against the logged ``llm.commit`` thoughts (B1)."""
    msgs = [m for m in published_thoughts(rig) if m.turn_id == turn_id]
    groups = by_attempt(msgs)
    got = [(g[-1].step, g[-1].kind, g[-1].text, g[-1].end) for g in groups.values()]
    assert got == list(want), (
        f"T-0180 B1/B3/B6 {what}: one block per attempt with a thought commit, (step, kind, final text, end) "
        f"written by hand\n want {list(want)}\n got  {got}\n (every published astra.thought: "
        f"{[(m.turn_id, m.attempt_id, m.step, m.kind, m.end, m.text[:40]) for m in published_thoughts(rig)]})"
    )
    logged = logged_thoughts(rig)
    for attempt, group in groups.items():
        first = group[0]
        for m in group:
            assert isinstance(m, AstraThought), type(m)
            assert (m.turn_id, m.request_id, m.step, m.kind) == (
                first.turn_id,
                first.request_id,
                first.step,
                first.kind,
            ), f"T-0180 B6 {what}: an attempt's labels never change; {group}"
            assert m.replay is False, f"T-0180: a live publication has replay=False; {m}"
            assert m.omitted_chars == 0, f"T-0180 B1 {what}: nothing omitted under the cap; {m}"
        ends = [m.end for m in group]
        assert ends[-1] in FINAL_ENDS and all(e == "open" for e in ends[:-1]), (
            f"T-0180 B3 {what}: zero or more 'open', then exactly one final message, last; {attempt}: {ends}"
        )
        assert attempt in logged, (
            f"T-0180 B1 {what}: {attempt} was published but logged no thought llm.commit"
        )
        texts = logged[attempt]["texts"]
        assert first.request_id == logged[attempt]["request_id"], (
            f"T-0180 B1 {what}: request_id is the llm.commit record's ({logged[attempt]['request_id']!r}); "
            f"got {first.request_id!r}"
        )
        prefixes = {"".join(texts[: k + 1]) for k in range(len(texts))}
        stray = [m.text for m in group if m.text not in prefixes]
        assert not stray, (
            f"T-0180 B1/B2 {what}: every text is the concatenation of the attempt's first k logged thought commits "
            f"(never a draft, never a partial commit); {attempt} published {stray[:2]}"
        )
        if oracle:
            assert group[-1].text == "".join(texts), (
                f"T-0180 B1 {what}: the last text is every logged thought commit of {attempt}, in order"
            )
    if oracle:
        assert list(groups) == list(logged_for_turn(rig, turn_id)), (
            f"T-0180 B1 {what}: one block for each attempt that logged a thought commit (prefill and captioner "
            f"excluded), in order; published {list(groups)}, logged {list(logged_for_turn(rig, turn_id))}"
        )
    return list(groups)


def logged_for_turn(rig: Any, turn_id: str) -> dict[str, dict[str, Any]]:
    """The oracle restricted to ``turn_id``'s requests (machine ids and pass-2 keys start with ``t<seq>-``)."""
    return {
        a: d
        for a, d in logged_thoughts(rig).items()
        if str(d["request_id"]).startswith(f"{turn_id}-")
    }


async def ask_off(rig: Any, script: Sequence[Commit], question: str = QUESTION_OFF) -> None:
    rig.llm.script = list(script)
    n = len(rig.lines())
    await rig.ask(question)
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) > n,
        "T-0180: the tools-off turn never ended",
    )
    await rig.advance(300)


def implicit(text: str, canvas_idx: int = 0) -> Commit:
    """T-0170: a thought that came from an implicit thought."""
    return Commit(  # type: ignore[call-arg]
        request_id="?", canvas_idx=canvas_idx, segment="thought", text=text, implicit_thought=True
    )


# =====================================================================================================
# B1: the source of truth, one block per attempt
# =====================================================================================================


async def test_tools_off_step1_publishes_the_logged_thought(tmp_path: Path) -> None:
    """B1/B3/B6, tools off: two thought commits (the second an open thought at the end of its canvas) then the
    answer: one block ``(1, step)`` ending ``done`` whose text is both commits, as logged; ``turn_id`` ``t1`` and
    ``request_id`` ``t1-gen``."""
    script = [
        thought("ZQX-OFF-1 Let me look at the page. ", 0),
        thought("It shows a cart.", 1, open_=True),
        answer("It is a shopping cart page.", 2),
    ]
    async with tool_rig(tmp_path, kit=None) as rig:
        await ask_off(rig, script)
    (attempt,) = assert_blocks(
        rig, [(1, "step", "ZQX-OFF-1 Let me look at the page. It shows a cart.", "done")]
    )
    assert by_attempt(published_thoughts(rig))[attempt][0].request_id == "t1-gen"


async def test_implicit_thought_is_published(tmp_path: Path) -> None:
    """B1: a T-0170 implicit thought is a thought commit and is included."""
    script = [implicit("ZQX-IMPLICIT the user wants the total. ", 0), answer("It is four.", 1)]
    async with tool_rig(tmp_path, kit=None) as rig:
        await ask_off(rig, script)
    assert_blocks(rig, [(1, "step", "ZQX-IMPLICIT the user wants the total. ", "done")])


async def test_no_thought_commit_publishes_nothing_and_turn_ids_follow(tmp_path: Path) -> None:
    """B1 ("an attempt with no thought commit publishes nothing"): turn 1 answers with no thought and publishes no
    ``astra.thought``; turn 2 thinks and publishes one block under ``t2`` (so the row is not vacuous)."""
    async with tool_rig(tmp_path, kit=None) as rig:
        await ask_off(rig, [answer("It is a page.", 0)])
        first = list(published_thoughts(rig))
        await ask_off(
            rig, [thought("ZQX-T2 think again.", 0), answer("Still a page.", 1)], "and now"
        )
    assert first == [], f"T-0180 B1: a turn with no thought commit published {first}"
    assert [m.turn_id for m in published_thoughts(rig)] and all(
        m.turn_id == "t2" for m in published_thoughts(rig)
    ), (
        "T-0180 B1: turn 2's thought is published, under t2 only; got "
        f"{[(m.turn_id, m.text) for m in published_thoughts(rig)]}"
    )
    assert_blocks(rig, [(1, "step", "ZQX-T2 think again.", "done")], turn_id="t2")


async def test_two_step_loop_one_block_per_step(tmp_path: Path) -> None:
    """B1/B6, ``dry_run`` loop: three steps, each thinking once; step 1 and 2 end on a tool call (``done``), step 3
    answers. Blocks ``(1, step)``, ``(2, step)``, ``(3, step)`` under ``t1-gen``, ``t1-gen-s2``, ``t1-gen-s3``."""
    async with tool_rig(
        tmp_path,
        policy=tsl.LOOP,
        script=[thought("ZQX-S1 open the menu. ", 0), answer(tsl.S1, 1), tool(VALID_CLICK)],
    ) as rig:
        rig.llm.script_steps(
            [thought("ZQX-S2 now the settings. ", 0), tool(tsl.CLICK2)],
            [thought("ZQX-S3 it is open.", 0), answer(tsl.S3, 1)],
        )
        await rig.ask(tsl.QUESTION)
        await rig.turn_done("T-0180: the loop turn never ended", limit_ms=40_000)
        await rig.advance(300)
    attempts = assert_blocks(
        rig,
        [
            (1, "step", "ZQX-S1 open the menu. ", "done"),
            (2, "step", "ZQX-S2 now the settings. ", "done"),
            (3, "step", "ZQX-S3 it is open.", "done"),
        ],
    )
    groups = by_attempt(published_thoughts(rig))
    assert [groups[a][0].request_id for a in attempts] == [
        tsl.step_id(1),
        tsl.step_id(2),
        tsl.step_id(3),
    ]


async def test_reask_is_its_own_block(tmp_path: Path) -> None:
    """B1/B6, T-0172: step 2 only announces a search and is re-asked; the re-ask thinks too. Blocks: step 1, step 2
    (``done``), and the re-ask as ``(2, reask)`` under step 2's machine id with its own attempt id."""
    async with t72.reask_rig(
        tmp_path,
        search=FakeSearch(),
        script=[thought("ZQX-RA1 search first. ", 0), tool(t72.FIRST_CALL)],
    ) as rig:
        await t72.run(
            rig,
            [thought("ZQX-RA2 the search failed. ", 0), answer(t72.ANN_PYTHON, 1)],
            reasks=[[thought("ZQX-RA3 say it plainly.", 0), answer(t72.R_STOPPED, 1)]],
        )
    attempts = assert_blocks(
        rig,
        [
            (1, "step", "ZQX-RA1 search first. ", "done"),
            (2, "step", "ZQX-RA2 the search failed. ", "done"),
            (2, "reask", "ZQX-RA3 say it plainly.", "done"),
        ],
    )
    groups = by_attempt(published_thoughts(rig))
    assert groups[attempts[2]][0].request_id == tsl.step_id(2), "the re-ask runs under step 2's id"
    assert attempts[1] != attempts[2], "a re-ask is a fresh attempt"


async def test_repair_is_its_own_block(tmp_path: Path) -> None:
    """B1/B6: the H-T84-2 repair of step 1's call (``_run_repair``) commits a thought: a ``(1, repair)`` block
    under the repair's request id, after step 1's own."""
    async with tool_rig(
        tmp_path,
        policy=tsl.LOOP,
        script=[thought("ZQX-RP1 click it. ", 0), answer(tsl.S1, 1), tool(REPAIRABLE_CLICK)],
        continuation=[thought("ZQX-RP2 drop the button.", 0), tool(REPAIRED_CLICK)],
    ) as rig:
        rig.llm.script_steps([answer(tsl.S3, 0)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done("T-0180: the repair turn never ended", limit_ms=40_000)
        await rig.advance(300)
    assert rig.llm.continuations(), "premise: a repair was sent"
    assert_blocks(
        rig,
        [
            (1, "step", "ZQX-RP1 click it. ", "done"),
            (1, "repair", "ZQX-RP2 drop the button.", "done"),
        ],
    )


async def test_reground_pass2_is_its_own_block(tmp_path: Path) -> None:
    """B1/B6: a ReGround pass 2 (``_pass2_stream``) that thinks gives a ``(1, reground)`` block under the pass-2
    key, after step 1's own; ``turn_id`` is the call's turn."""
    pass2 = [thought("ZQX-P2 the button is higher.", 0), tool(click(900, 100), canvas_idx=1)]
    async with t42.guard_rig(
        tmp_path,
        guard=t42.ABSENT,
        reground=True,
        browser=jpeg_browser(),
        script=[thought("ZQX-P1 the sign in button. ", 0), answer(tsl.S1, 1), tool(VALID_CLICK)],
    ) as rig:
        rig.llm.script_pass2(pass2)
        await t42.run(rig, [answer(tsl.S3)])
    (p2,) = [d for d in rig.kind("llm.request") if d.get("phase") == "reground"]
    attempts = assert_blocks(
        rig,
        [
            (1, "step", "ZQX-P1 the sign in button. ", "done"),
            (1, "reground", "ZQX-P2 the button is higher.", "done"),
        ],
    )
    assert attempts[1] == p2["attempt_id"], (attempts, p2)


async def test_captioner_never_publishes(tmp_path: Path) -> None:
    """B1: the idle captioner's stream (with a thought commit) publishes nothing; the turn after it does (so the
    row is not vacuous)."""
    from local_astra.contracts.memory import Episode
    from local_astra.contracts.perception import Source

    thumb = tmp_path / "thumb.jpg"
    thumb.write_bytes(b"\xff\xd8" + b"CAPTION-T0180|" * 8)
    caption = [thought("ZQX-CAPTION-THOUGHT a cart.", 0), answer("A cart page.", 1)]
    async with t42.guard_rig(tmp_path / "rig", mode="off", guard=t42.ABSENT, script=caption) as rig:
        await rig.session.memory.ingest(
            Episode(ts_ms=0, source=Source.SCREEN, thumbnail_path=str(thumb), phash="cap0180")
        )
        await rig.session.captioner.run_once()
        await rig.pump()
        after_caption = list(published_thoughts(rig))
        await ask_off(rig, [thought("ZQX-TURN-THOUGHT look.", 0), answer("A page.", 1)])
    assert any("caption" in rid for rid, _, _ in rig.llm.generates), (
        "premise: a caption was generated"
    )
    assert after_caption == [], f"T-0180 B1: the captioner published {after_caption}"
    assert "ZQX-CAPTION-THOUGHT" not in dump(rig.published.messages)
    assert_blocks(rig, [(1, "step", "ZQX-TURN-THOUGHT look.", "done")], oracle=False)


# =====================================================================================================
# B2: never a draft, never an answer
# =====================================================================================================


async def test_answer_splice_and_open_answer_never_contribute_tools_off(tmp_path: Path) -> None:
    """B2 (adversarial): the T-0124 splice ``<|chan<eos>nel>thought SECRET`` inside an answer commit and a
    ``thought_open=True`` answer commit never contribute; the real thought commit does, alone. Neither secret is in
    any published message, history or ``turns.jsonl``."""
    script = [
        thought("ZQX-REAL-OFF ", 0),
        answer("Sure, <|chan<eos>nel>thought ZQX-SPLICE-SECRET<channel|>ok. ", 1),
        Commit(
            request_id="?", canvas_idx=2, segment="answer", text="ZQX-OPEN-DRAFT", thought_open=True
        ),
        answer("Done. ", 3),
    ]
    async with tool_rig(tmp_path, kit=None) as rig:
        await ask_off(rig, script)
        history = history_dump(rig)
        turns = "\n".join(rig.lines())
    assert_blocks(rig, [(1, "step", "ZQX-REAL-OFF ", "done")])
    for secret in ("ZQX-SPLICE-SECRET", "ZQX-OPEN-DRAFT"):
        for name, text in (
            ("published", dump(rig.published.messages)),
            ("history", history),
            ("turns.jsonl", turns),
        ):
            assert secret not in text, f"T-0180 B2: {secret} (answer-segment text) reached {name}"


async def test_closed_thought_in_answer_never_contributes_tools_on(tmp_path: Path) -> None:
    """B2/H-T180-7 (adversarial): tools on, a closed thought block inside an answer commit (removed by the T-0117
    cleaner) is not a thought and is shown nowhere; the thought commit is published alone."""
    script = [
        thought("ZQX-REAL-ON ", 0),
        answer("Sure, <|channel>thought ZQX-CLOSED-SECRET<channel|>here it is. ", 1),
    ]
    async with tool_rig(tmp_path, script=script) as rig:
        await rig.ask("what does it say")
        await rig.turn_done("T-0180: the tools-on turn never ended")
        await rig.advance(300)
        history = history_dump(rig)
    assert_blocks(rig, [(1, "step", "ZQX-REAL-ON ", "done")])
    assert "ZQX-CLOSED-SECRET" not in dump(rig.published.messages), "T-0180 B2/H-T180-7"
    assert "ZQX-CLOSED-SECRET" not in history


# =====================================================================================================
# B3/B4/B5: one end per attempt
# =====================================================================================================


async def test_backend_error_ends_error(tmp_path: Path) -> None:
    """B3 ``error``: the backend raises after two thought commits; the attempt's block holds both and ends
    ``error``."""
    async with tool_rig(tmp_path, kit=None) as rig:
        rig.llm.fail_generates_after_script(RuntimeError("the engine fell over"))
        await ask_off(rig, [thought("ZQX-ERR one. ", 0), thought("two", 1, open_=True)])
    assert rig.machine.events(E.LlmError), "premise: the turn failed on the backend error"
    assert_blocks(rig, [(1, "step", "ZQX-ERR one. two", "error")])


@pytest.mark.parametrize("how", ["takeover", "disconnect"])
async def test_abort_ends_interrupted_and_nothing_after_is_published(
    tmp_path: Path, how: str
) -> None:
    """B3 ``interrupted`` / B4 (adversarial): the stream is held after one thought commit; a barge-in (onset and
    takeover) or ``ClientDisconnected`` aborts it; the fake keeps yielding (``LateLlm(late=True)``) a thought
    carrying ``ZQX-AFTER-ABORT``. The block holds only what came before the abort and ends ``interrupted``; the
    sentinel is in no published message."""
    script = [
        thought("ZQX-BEFORE-ABORT I start. ", 0, open_=True),
        thought("ZQX-AFTER-ABORT keep going.", 1, open_=True),
        answer("Never voiced.", 2),
    ]
    llm = t42.LateLlm(script, late=True)
    async with tool_rig(tmp_path, kit=None, llm=llm) as rig:
        rig.llm.hold_after(1)
        await rig.ask(QUESTION_OFF)
        await rig.until(
            lambda: any(e.segment == "thought" for e in rig.machine.events(E.LlmCommit)),
            "premise: the first thought commit reached the machine",
        )
        await rig.until(
            lambda: bool(published_thoughts(rig)),
            "T-0180 B1: no astra.thought was published for the first thought commit",
            limit_ms=2_000,
        )
        now = rig.clock.now_ms
        if how == "disconnect":
            rig.submit(E.ClientDisconnected(t_ms=now))
        else:
            rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
            rig.submit(E.UserTakeover(t_ms=now + 1, onset_ms=now))
        await rig.advance(500)
        rig.llm.release()
        await rig.advance(3_000)
    engines = [rid for rid, _, _ in rig.llm.generates if "-t1-gen" in rid]
    assert engines and engines[0] in rig.llm.aborted, (
        f"premise: the attempt was aborted; {engines}, {rig.llm.aborted}"
    )
    assert_blocks(rig, [(1, "step", "ZQX-BEFORE-ABORT I start. ", "interrupted")], oracle=False)
    assert "ZQX-AFTER-ABORT" not in dump(rig.published.messages), (
        "T-0180 B4: a commit received after the abort was published"
    )


async def test_degenerate_trip_then_retry_tools_off(tmp_path: Path) -> None:
    """B5/H-T180-3 and the T-0179 retry: the tripping thought commit is in the first block, which ends
    ``stopped_repeating``; the retry is its own ``(1, retry)`` block ending ``done``; the answer after the trip
    (``LATE``) is in no thought."""
    t42.degenerate_message()
    first = [thought(t79.TRIP_THOUGHT, 0, open_=True), answer(t79.LATE, 1)]
    second = [thought("ZQX-RETRY Let me look at the page.", 0), answer(t79.ANSWER, 1)]
    async with t79.retry_rig(tmp_path, mode="off", script=[answer("Second answer here.")]) as rig:
        rig.llm.script_generates(first, second)
        await t79.run_off(rig)
    assert t79.retry_notes(rig), "premise: T-0179 retried"
    assert_blocks(
        rig,
        [
            (1, "step", t79.TRIP_THOUGHT, "stopped_repeating"),
            (1, "retry", "ZQX-RETRY Let me look at the page.", "done"),
        ],
    )
    assert t79.LATE not in dump(published_thoughts(rig))


async def test_degenerate_trip_then_retry_live_step_k(tmp_path: Path) -> None:
    """B5/B6 live, the t10 shape: step 4's thought trips and is retried; blocks for steps 2, 3, 4 (stopped), and the
    step-4 retry; step 1 and the final step think nothing and have no block."""
    t42.degenerate_message()
    trip = [
        thought(t79.TRIP_THOUGHT, 0, open_=True),
        answer(t79.LATE, 1),
        tool(t79.CLICK_X, canvas_idx=1),
    ]
    clean = [thought(t79.CLEAN_THOUGHT, 0), tool(t79.CLICK4)]
    async with t79.retry_rig(tmp_path, script=t42.STEP1) as rig:
        await t79.run(rig, *t79.live_turn_steps(), trip, clean, [answer(t79.FINAL)])
    assert t79.retry_notes(rig), "premise: T-0179 retried"
    attempts = assert_blocks(
        rig,
        [
            (2, "step", "Step two plan.", "done"),
            (3, "step", "Step three plan.", "done"),
            (4, "step", t79.TRIP_THOUGHT, "stopped_repeating"),
            (4, "retry", t79.CLEAN_THOUGHT, "done"),
        ],
    )
    groups = by_attempt(published_thoughts(rig))
    assert {groups[a][0].request_id for a in attempts[2:]} == {tsl.step_id(4)}


@pytest.mark.parametrize("phase", ["generate", "tool_step"])
async def test_engine_empty_resend_is_a_retry_block(tmp_path: Path, phase: str) -> None:
    """B1/B6 and the group B ruling (2026-10-28): a T-0171 engine-empty resend is ``kind: "retry"``. The failed
    attempt delivered nothing, so it has no block (B1: "an attempt with no thought commit publishes nothing"); the
    resend's thought is one ``(step, retry)`` block under the same machine id, ending ``done``. ``tool_step``: step 1
    thinks too and keeps its ``(1, step)`` block; the resent step 2 is ``(2, retry)``."""
    from tests.fakes.llm import engine_empty_error

    if phase == "generate":
        cm = tool_rig(
            tmp_path,
            script=[thought("ZQX-EMPTY-RESEND look again. ", 0), answer("It is a login form. ", 1)],
        )
    else:
        cm = tool_rig(
            tmp_path,
            policy=tsl.LOOP,
            script=[
                thought("ZQX-EMPTY-S1 click it. ", 0),
                answer(tsl.PREFIX, 1),
                tool(VALID_CLICK),
            ],
        )
    async with cm as rig:
        if phase == "generate":
            rig.llm.fail_generates(engine_empty_error, None)
        else:
            rig.llm.script_steps([thought("ZQX-EMPTY-S2 report it. ", 0), answer(tsl.REPORT, 1)])
            rig.llm.fail_steps(engine_empty_error, None)
        await rig.ask(tsl.QUESTION)
        await rig.turn_done("T-0180: the engine-empty turn never ended", limit_ms=40_000)
        await rig.advance(300)
    machine = tsl.step_id(1) if phase == "generate" else tsl.step_id(2)
    engines = [rid for rid, _ in tsl.generates_for(rig, machine)]
    assert len(engines) == 2, f"premise: T-0171 resent {machine} once; {engines}"
    failed, resent = engines
    if phase == "generate":
        want = [(1, "retry", "ZQX-EMPTY-RESEND look again. ", "done")]
    else:
        want = [
            (1, "step", "ZQX-EMPTY-S1 click it. ", "done"),
            (2, "retry", "ZQX-EMPTY-S2 report it. ", "done"),
        ]
    attempts = assert_blocks(rig, want, what="(T-0171 resend)")
    assert failed not in attempts, f"T-0180 B1: the empty attempt {failed} published a block"
    assert attempts[-1] == resent, (attempts, resent)
    assert by_attempt(published_thoughts(rig))[resent][0].request_id == machine


# =====================================================================================================
# B7: the reset fence; B3's exception
# =====================================================================================================


@pytest.mark.parametrize("turns_before", [1, 2])
async def test_is_pre_reset_judges_a_thought_by_turn_id(tmp_path: Path, turns_before: int) -> None:
    """B7 (the group A review note): ``is_pre_reset`` judges ``astra.thought`` by ``turn_id``; ``t1``..``tn`` are
    dropped after a reset with n turns before it, ``t(n+1)`` is not (the too-wide direction)."""

    def probe(turn: str) -> AstraThought:
        return AstraThought(
            turn_id=turn,
            request_id=f"{turn}-gen",
            attempt_id=f"x-{turn}-gen-1",
            step=1,
            kind="step",
            text="ZQX-PROBE",
            omitted_chars=0,
            end="done",
        )

    async with tool_rig(tmp_path, kit=None, script=[answer("Fine. ")]) as rig:
        for i in range(turns_before):
            await ask_off(rig, [answer("Fine. ")], f"question number {i}")
        floor = rig.state.turn_seq
        assert floor == turns_before
        assert rig.session.is_pre_reset(probe(f"t{floor}")) is False, "premise: no reset, no filter"
        await do_reset(rig)
        one_cleared(rig)
        publish = rig.session._publish
        for seq in range(1, floor + 1):
            stale = probe(f"t{seq}")
            assert rig.session.is_pre_reset(stale) is True, f"T-0180 B7: t{seq} (floor t{floor})"
            n = len(rig.published.messages)
            publish(stale)
            assert rig.published.messages[n:] == [], f"T-0180 B7: a t{seq} thought was published"
        fresh = probe(f"t{floor + 1}")
        assert rig.session.is_pre_reset(fresh) is False, f"T-0180 B7 too wide: t{floor + 1}"


async def test_reset_mid_thought_publishes_nothing_after_cleared(tmp_path: Path) -> None:
    """B3's exception and B7 (adversarial): a reset lands while the attempt is held after one published thought;
    the fake keeps yielding. After ``astra.session.cleared`` no ``astra.thought`` is published at all (no final
    message either), and the post-reset sentinel is nowhere."""
    script = [
        thought("ZQX-RESET-BEFORE I start. ", 0, open_=True),
        thought("ZQX-RESET-AFTER more.", 1, open_=True),
        answer("Never voiced.", 2),
    ]
    llm = t42.LateLlm(script, late=True)
    async with tool_rig(tmp_path, kit=None, llm=llm) as rig:
        rig.llm.hold_after(1)
        await rig.ask(QUESTION_OFF)
        await rig.until(
            lambda: bool(published_thoughts(rig)),
            "T-0180 B1: no astra.thought was published for the first thought commit",
            limit_ms=5_000,
        )
        await do_reset(rig)
        rig.llm.release()
        await rig.advance(3_000)
        cleared = one_cleared(rig)
    at = rig.published.messages.index(cleared)
    before = [m for m in rig.published.messages[:at] if getattr(m, "type", None) == THOUGHT]
    after = [m for m in rig.published.messages[at + 1 :] if getattr(m, "type", None) == THOUGHT]
    assert before and before[0].text == "ZQX-RESET-BEFORE I start. ", before
    assert after == [], f"T-0180 B3/B7: thoughts published after the reset: {after}"
    assert "ZQX-RESET-AFTER" not in dump(rig.published.messages)


# =====================================================================================================
# The cap (A6, H-T180-5)
# =====================================================================================================


def long_thought(n: int) -> str:
    """``n`` code points, one of them outside the BMP (so a UTF-16 count is off by one), no repetition."""
    body = "".join(chr(0x61 + (i * 7 + i // 26) % 26) for i in range(n - 1))
    return "\U0001d11e" + body


async def test_cap_20000_gives_16000_and_4000_omitted(tmp_path: Path) -> None:
    """A6/H-T180-5 through the session: four 5000-code-point thought commits (20000) give, after the fourth, a text
    of the first 16000 code points and ``omitted_chars`` 4000; after the third (15000) the text is all of it and
    nothing is omitted."""
    assert THOUGHT_MAX_CHARS == 16000
    full = long_thought(20_000)
    parts = [full[i : i + 5000] for i in range(0, 20_000, 5000)]
    script = [thought(p, i, open_=True) for i, p in enumerate(parts)] + [answer("Done.", 4)]
    async with tool_rig(tmp_path, kit=None) as rig:
        await ask_off(rig, script)
    groups = by_attempt(published_thoughts(rig))
    assert len(groups) == 1, f"T-0180 B1: one attempt; {list(groups)}"
    (group,) = groups.values()
    last = group[-1]
    assert (len(last.text), last.omitted_chars, last.end) == (16_000, 4_000, "done"), (
        len(last.text),
        last.omitted_chars,
        last.end,
    )
    assert last.text == full[:16_000], "T-0180 A6: the first 16000 code points, no chunking"
    under = [m for m in group if m.text == full[:15_000]]
    assert under and all(m.omitted_chars == 0 for m in under), (
        "T-0180 B1/A6: after the third commit (15000 code points) one message carries all of it, nothing omitted"
    )


# =====================================================================================================
# Group B review rows (rulings B-R6..B-R8, 2026-10-28): every tap site pinned on its own
# =====================================================================================================
#
# B-R8 asks for a repair row at step >= 2. None is written: a repairable step >= 2 draft is refused "repair: step 1 only"
# before any repair (T-0114 criterion 8, session.py), so no repair runs at step >= 2 and the mutant `call.loop_step` -> 1
# in `_run_repair` is equivalent on every reachable input (reported to the orchestrator).


def _after_repair_thought(llm: Any, then: Any) -> None:
    """Make every repair (continuation, not step) request of ``llm`` do ``then`` once its scripted commits are
    yielded: ``HANG`` (held until the engine id is aborted) or an exception (the backend failing mid-stream).
    Wraps the instance's ``generate``/``abort``; the fake's own bookkeeping is untouched."""
    import asyncio

    from tests.fakes.llm import is_step_prompt
    from tests.unit._live_browser_support import HANG

    inner_generate, inner_abort = llm.generate, llm.abort
    hangs: dict[str, asyncio.Event] = {}

    async def generate(
        prompt: Any, request_id: str, options: Any = None, *, xargs: Any = None, cd: Any = None
    ) -> Any:
        async for c in inner_generate(prompt, request_id, options, xargs=xargs, cd=cd):
            yield c
        if getattr(prompt, "continue_final_message", False) and not is_step_prompt(prompt):
            if then == HANG:
                await hangs.setdefault(request_id, asyncio.Event()).wait()
                return
            raise then

    async def abort(request_id: str) -> None:
        await inner_abort(request_id)
        hangs.setdefault(request_id, asyncio.Event()).set()

    llm.generate, llm.abort = generate, abort


def _pass2_thought_first(rig: Any, text: str, then: Any) -> None:
    """Pass 2 commits the thought ``text`` (an open thought), then serves ``then`` from ``script_pass2``: ``HANG``
    (until aborted) or an exception (raised by the backend after the thought)."""
    llm = rig.llm
    inner = llm._pass2

    async def _pass2(prompt: Any, request_id: str) -> Any:
        yield thought(text, 0, open_=True).model_copy(update={"request_id": request_id})
        async for c in inner(prompt, request_id):
            yield c

    llm._pass2 = _pass2
    llm.script_pass2(then)


async def test_reground_pass2_at_step_two_is_labelled_step_two(tmp_path: Path) -> None:
    """B6/B-R8: ReGround pass 2 of step 2's click is ``(2, reground)``; step 1's pass 2 is ``(1, reground)``."""
    async with t42.guard_rig(
        tmp_path,
        guard=t42.ABSENT,
        reground=True,
        browser=jpeg_browser(),
        script=[thought("ZQX-GK1 the menu. ", 0), answer(tsl.S1, 1), tool(VALID_CLICK)],
    ) as rig:
        rig.llm.script_pass2(
            [thought("ZQX-GK1-P2 a bit higher.", 0), tool(click(900, 100), canvas_idx=1)],
            [thought("ZQX-GK2-P2 a bit lower.", 0), tool(click(300, 700), canvas_idx=1)],
        )
        await t42.run(
            rig,
            [thought("ZQX-GK2 the settings. ", 0), tool(tsl.CLICK2)],
            [answer(tsl.S3)],
        )
    p2 = [d for d in rig.kind("llm.request") if d.get("phase") == "reground"]
    assert [d.get("step") for d in p2] == [1, 2], f"premise: one pass 2 per step; {p2}"
    assert_blocks(
        rig,
        [
            (1, "step", "ZQX-GK1 the menu. ", "done"),
            (1, "reground", "ZQX-GK1-P2 a bit higher.", "done"),
            (2, "step", "ZQX-GK2 the settings. ", "done"),
            (2, "reground", "ZQX-GK2-P2 a bit lower.", "done"),
        ],
        what="(pass 2 at step 2)",
    )


async def test_repair_aborted_mid_thought_ends_interrupted(tmp_path: Path) -> None:
    """B3/B-R8 (adversarial): the repair commits a thought and then hangs; the repair timeout aborts it. Its block
    holds that thought and ends with exactly one final message, ``interrupted`` (no final at all, or ``done``/
    ``error``, fails)."""
    from tests.unit._live_browser_support import HANG

    async with tool_rig(
        tmp_path,
        policy=tsl.LOOP,
        script=[thought("ZQX-RC1 click it. ", 0), answer(tsl.S1, 1), tool(REPAIRABLE_CLICK)],
        continuation=[thought("ZQX-RC-REPAIR drop the", 0, open_=True)],
    ) as rig:
        _after_repair_thought(rig.llm, HANG)
        rig.llm.script_steps([answer(tsl.S3, 0)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done("T-0180: the aborted-repair turn never ended", limit_ms=40_000)
        await rig.advance(300)
    (engine, *_), *rest = rig.llm.continuations()
    assert not rest and engine in rig.llm.aborted, (
        f"premise: the one repair was aborted; {rig.llm.continuations()}, {rig.llm.aborted}"
    )
    assert_blocks(
        rig,
        [
            (1, "step", "ZQX-RC1 click it. ", "done"),
            (1, "repair", "ZQX-RC-REPAIR drop the", "interrupted"),
        ],
        what="(repair aborted)",
    )


async def test_repair_backend_error_mid_thought_ends_error(tmp_path: Path) -> None:
    """B3/B-R8: the repair's backend raises after one thought commit; the block holds it and ends ``error``."""
    async with tool_rig(
        tmp_path,
        policy=tsl.LOOP,
        script=[thought("ZQX-RE1 click it. ", 0), answer(tsl.S1, 1), tool(REPAIRABLE_CLICK)],
        continuation=[thought("ZQX-RE-REPAIR drop the", 0, open_=True)],
    ) as rig:
        _after_repair_thought(rig.llm, RuntimeError("the engine fell over mid-repair"))
        rig.llm.script_steps([answer(tsl.S3, 0)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done("T-0180: the failed-repair turn never ended", limit_ms=40_000)
        await rig.advance(300)
    assert [d for d in rig.kind("error") if d.get("what") == "tool.repair"], (
        "premise: the repair failed on the backend error"
    )
    assert_blocks(
        rig,
        [
            (1, "step", "ZQX-RE1 click it. ", "done"),
            (1, "repair", "ZQX-RE-REPAIR drop the", "error"),
        ],
        what="(repair error)",
    )


async def test_pass2_collapse_ends_stopped_repeating(tmp_path: Path) -> None:
    """B5/B-R8 (the T-0142 A3.5 rig): a pass 2 whose thought collapses is a ``(1, reground)`` block holding the
    tripping commit and ending ``stopped_repeating`` (not ``error``, not ``done``); the commit after the trip is
    in no thought."""
    t42.degenerate_message()
    after = " ZQX-AFTER-TRIP" + " the" * 20
    collapse = [
        thought(t42.COLLAPSE, 0, open_=True),
        thought(after, 1, open_=True),
        tool(click(900, 100), canvas_idx=1),
    ]
    async with t42.guard_rig(tmp_path, guard=True, reground=True, browser=jpeg_browser()) as rig:
        rig.llm.script_pass2(collapse)
        await t42.run(rig, [answer(tsl.S3)])
    (rec,) = regrounds(rig)
    assert rec["outcome"] == "pass2_timeout", f"premise (T-0142 R-7): pass 2 tripped; {rec}"
    assert t42.notes(rig), "premise: the trip note was logged"
    assert_blocks(rig, [(1, "reground", t42.COLLAPSE, "stopped_repeating")], what="(pass-2 trip)")
    assert "ZQX-AFTER-TRIP" not in dump(published_thoughts(rig))


@pytest.mark.parametrize("how", ["takeover", "timeout"])
async def test_pass2_cut_short_mid_thought_ends_interrupted(tmp_path: Path, how: str) -> None:
    """B3/B-R7/B-R8: pass 2 commits a thought, then hangs. ``takeover``: a barge-in cancels the execute;
    ``timeout``: ``REGROUND_TIMEOUT_MS`` expires (B-R7: the request was cut short, as a barge-in is). Either way the
    ``(1, reground)`` block holds the thought and ends ``interrupted``."""
    from tests.unit._live_browser_support import HANG
    from tests.unit.test_session_reground import reground_timeout_ms

    limit = reground_timeout_ms()
    async with t42.guard_rig(
        tmp_path, guard=t42.ABSENT, reground=True, browser=jpeg_browser()
    ) as rig:
        _pass2_thought_first(rig, "ZQX-P2-CUT the button is", HANG)
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await rig.until(
            lambda: any(m.kind == "reground" for m in published_thoughts(rig)),
            "premise: pass 2's thought was published",
            limit_ms=10_000,
        )
        if how == "takeover":
            now = rig.clock.now_ms
            rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
            await rig.pump()
            await rig.advance(500)
        else:
            await rig.turn_done(
                "T-0180: the timed-out pass-2 turn never ended", limit_ms=limit + 40_000
            )
            await rig.advance(300)
    assert rig.llm.pass2_ids and rig.llm.pass2_ids[0] in rig.llm.aborted, (
        "premise: pass 2 was aborted"
    )
    if how == "timeout":
        (rec,) = regrounds(rig)
        assert rec["outcome"] == "pass2_timeout", f"premise: pass 2 timed out; {rec}"
    want_steps = [(1, "reground", "ZQX-P2-CUT the button is", "interrupted")]
    got = [
        (m[-1].step, m[-1].kind, m[-1].text, m[-1].end)
        for m in by_attempt([m for m in published_thoughts(rig) if m.kind == "reground"]).values()
    ]
    assert got == want_steps, f"T-0180 B3/B-R7 ({how}): {got}"
    assert_blocks(rig, want_steps, what=f"(pass 2 {how})")


async def test_pass2_backend_error_mid_thought_ends_error(tmp_path: Path) -> None:
    """B3/B-R8: pass 2's backend raises after one thought commit; the ``(1, reground)`` block ends ``error``."""
    async with t42.guard_rig(
        tmp_path, guard=t42.ABSENT, reground=True, browser=jpeg_browser()
    ) as rig:
        _pass2_thought_first(
            rig, "ZQX-P2-ERR the button is", RuntimeError("vLLM returned HTTP 500")
        )
        await t42.run(rig, [answer(tsl.S3)])
    (rec,) = regrounds(rig)
    assert rec["outcome"] == "error", f"premise: pass 2 failed; {rec}"
    assert_blocks(
        rig, [(1, "reground", "ZQX-P2-ERR the button is", "error")], what="(pass 2 error)"
    )


async def test_over_long_resend_is_a_retry_block(tmp_path: Path) -> None:
    """B-R6: the turn generate is refused as over-long (HTTP 413, a scripted failure that delivered nothing); the one
    resend (``drop=1``) under a fresh attempt id thinks, and its block is ``(1, retry)``, ending ``done``. The
    refused attempt has no block."""
    from tests.fakes.llm import http_error

    script = [thought("ZQX-OVERLONG-RESEND shorter now. ", 0), answer("It is a page.", 1)]
    async with tool_rig(tmp_path, kit=None) as rig:
        await ask_off(rig, [answer("Fine. ")], "first question")
        rig.llm.fail_generates(http_error(413), None)
        await ask_off(rig, script, "second question")
    notes = [d for d in rig.kind("note") if "too long" in str(d.get("what"))]
    assert notes and notes[0].get("status") == 413, f"premise: the over-long resend ran; {notes}"
    engines = [rid for rid, _, _ in rig.llm.generates if "-t2-gen-" in rid]
    assert len(engines) == 2, f"premise: the refused attempt and its one resend; {engines}"
    refused, resent = engines
    attempts = assert_blocks(
        rig,
        [(1, "retry", "ZQX-OVERLONG-RESEND shorter now. ", "done")],
        turn_id="t2",
        what="(413 resend)",
    )
    assert attempts == [resent] == [notes[0]["attempt_id"]], (attempts, engines, notes)
    assert refused not in attempts


async def test_close_mid_thought_publishes_nothing_after_close_starts(tmp_path: Path) -> None:
    """B3's exception/B-R8 (adversarial): the session is closed while the attempt is held after one published
    thought commit. Once ``aclose`` starts no ``astra.thought`` is published (no final ``interrupted`` either)."""
    script = [
        thought("ZQX-CLOSE-BEFORE I start. ", 0, open_=True),
        thought("ZQX-CLOSE-AFTER more.", 1, open_=True),
        answer("Never voiced.", 2),
    ]
    llm = t42.LateLlm(script, late=True)
    published_at_close: list[Any] = []
    async with tool_rig(tmp_path, kit=None, llm=llm) as rig:
        rig.llm.hold_after(1)
        await rig.ask(QUESTION_OFF)
        await rig.until(
            lambda: bool(published_thoughts(rig)),
            "T-0180 B1: no astra.thought was published for the first thought commit",
            limit_ms=5_000,
        )
        published_at_close = list(published_thoughts(rig))
    assert [(m.text, m.end) for m in published_at_close] == [
        ("ZQX-CLOSE-BEFORE I start. ", "open")
    ], f"premise: one open message before close; {published_at_close}"
    assert published_thoughts(rig) == published_at_close, (
        "T-0180 B3: an astra.thought was published after the session started closing: "
        f"{published_thoughts(rig)[len(published_at_close) :]}"
    )
