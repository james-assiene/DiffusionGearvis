"""T-0126: the gate's end-of-turn literals are never stored or published with tools on (criteria 1-8).

Session-level rows over the T-0092c/T-0114 rig (``tests/unit/_tool_turn_support.tool_rig``, default kit, ``dry_run``),
the fake LLM, the recording executor, ``FakeSearch`` and the ``ScriptedClock``. The sinks are read with T-0117's
``snapshot`` and T-0124's sink helpers (``sinks``, ``assert_stored_exactly``, ``assert_no_literal``, ``released``,
``voice``), imported, not copied.

Every expected text is D(raw) written out by hand from the task's rule: C(raw) (T-0117 criterion 1: special literals
deleted to a fixpoint, closed thought blocks removed, the text from an open nothing closes cut, stray closes
deleted, then ``drop_special_literals``), then cut at the earliest ``<end_of_turn>`` or ``<|endoftext|>`` (E, the gate
trim literals that are not tokenizer literals). Nothing is read from ``_AnswerCleaner`` to get an expected value.

Pending rows (``pending(task="T-0126")``) are red at the base 4a25a85. Unmarked rows hold there: criterion 4's I4
rows and its closed-block case (C already removes the block and the E literal inside it; it is the stage-order
guard against "cut before C"), and the note rows where nothing is cut.

Readings where the task is silent (listed in the test-author's report):

- "``astra.spoken`` sources" are its texts, compared whitespace-normalised (T-0124's reading); "no ``now``" is a
  whole-word match;
- ``call.spoken_prefix`` (criterion 6) is read as the ``spoken_prefix`` argument ``_judge_tool`` receives (the
  ``_ToolCall`` is built from it; ``_judge_tool`` is unchanged by the task), and as the ``tool.call`` records' key;
- "the repair request's continued assistant text" is the text of the last message of the repair body (string
  content, or the join of its text parts);
- "the verdict, card states and execution equal 4a25a85's" (criterion 6) is compared against a control turn on the
  same tree whose prefix is ``Let me look. `` with no E literal: at the base the prefix text does not reach the
  verdict (the call is judged from the ``tool`` commit), so the control's verdict is the base's;
- criterion 7's "the ``outcome`` 4a25a85 records for a tools-on ``answer("<pad>")``" is a control turn run on the same
  tree (that path is I4's, unchanged by the task), and the value ``silent`` (``_TurnLog.outcome`` with nothing
  played), which the control row also checks;
- the note's "no value holds stream text" is checked as no 4-character substring of the cut text in any value but
  ``what``.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``dry_run``, loop off
unless a row says otherwise, n = 1 and deterministic per row. No measurement.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import speech_gate
from local_astra.conductor.turn_machine import TurnPolicy
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.turn import TurnState
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS
from tests.fakes.search import FakeSearch
from tests.unit._tool_turn_support import (
    CALL_ID,
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    VALID_CLICK,
    ToolRig,
    answer,
    assert_history_clean,
    assert_never_voiced,
    tool,
    tool_rig,
)
from tests.unit.test_session_literals_tools_off import (
    LITERALS,
    assert_no_literal,
    assert_stored_exactly,
    press,
    released,
    sinks,
    voice,
)
from tests.unit.test_session_thought_in_answer import NOTE_WHAT as THOUGHT_NOTE_WHAT
from tests.unit.test_session_thought_in_answer import answer_commits, snapshot
from tests.unit.test_session_thought_tools_off import open_answer
from tests.unit.test_session_tool_loop import PREFIX, SEARCH_CALL, loop_policy, loop_turn, notes

SECRET = "SECRET-THOUGHT"
GEN = "t1-gen"
NOTE_WHAT = "special literals in an answer commit were not voiced or stored"
NOTE_KEYS = frozenset({"what", "request_id", "step", "chars", "cut"})
LOOP_OFF = TurnPolicy(thinking=True)

#: E, derived as the task's implementation notes say (``_TRIM_TOKENS`` minus ``SPECIAL_TOKENS``)
EOT = tuple(t for t in speech_gate._TRIM_TOKENS if t not in SPECIAL_TOKENS)


def test_e_is_the_two_end_of_turn_strings():
    """Premise (the rule): E is exactly ``<end_of_turn>`` and ``<|endoftext|>``, written out by hand."""
    assert sorted(EOT) == sorted(("<end_of_turn>", "<|endoftext|>")), EOT


# --- the raw texts and D(raw), by hand ---------------------------------------------------------------------

#: criterion 1, second example
HI = "Hi <end_of_turn> bye. "
HI_D = "Hi "
#: criterion 3
STOP = ("First. <end_of_turn>Second. ", "Third. ")
STOP_D = "First. "
#: criterion 5: the step-2 report; the cut removes the invented next turn
REPORT_RAW = "Found it: the café opens at nine. <end_of_turn>user: and tomorrow? "
REPORT_D = "Found it: the café opens at nine. "
REPORT_CUT = "<end_of_turn>user: and tomorrow? "
#: criterion 6
CALL_RAW = "Let me look. <end_of_turn>junk "
CALL_D = "Let me look. "
CALL_CUT = "<end_of_turn>junk "


# --- driving a tools-on session --------------------------------------------------------------------------


async def run_on(
    tmp_path: Path,
    scripts: Sequence[Sequence[Commit]],
    *,
    policy: TurnPolicy = LOOP_OFF,
    steps: Sequence[Sequence[Commit]] = (),
    search: Any = None,
    continuation: Sequence[Commit] | None = None,
    trace: list[Any] | None = None,
    judged: list[str] | None = None,
) -> tuple[ToolRig, dict[str, Any]]:
    """One tools-on session (default kit), one turn per script; the snapshot is taken after the last turn.

    ``steps`` (one-turn sessions only) scripts the loop steps after step 1. ``trace`` receives ``("submit", event)``
    and ``("log", kind, data)`` in the order they happen (T-0121's ``run_off`` technique); ``judged`` receives the
    ``spoken_prefix`` argument of every ``_judge_tool`` call."""
    assert not steps or len(scripts) == 1, "steps are for one-turn sessions"
    async with tool_rig(tmp_path, policy=policy, search=search, continuation=continuation) as rig:
        if trace is not None:
            submit, record = rig.session.submit, rig.log.record

            def traced_submit(event: Any) -> None:
                trace.append(("submit", event))
                submit(event)

            def traced_record(kind: str, data: Any, **kw: Any) -> None:
                trace.append(("log", kind, data))
                record(kind, data, **kw)

            rig.session.submit = traced_submit
            rig.log.record = traced_record
        if judged is not None:
            judge = rig.session._judge_tool
            sig = inspect.signature(judge)

            def spy(*a: Any, **kw: Any) -> Any:
                judged.append(sig.bind(*a, **kw).arguments["spoken_prefix"])
                return judge(*a, **kw)

            rig.session._judge_tool = spy
        for i, script in enumerate(scripts):
            rig.llm.script = list(script)
            if steps:
                rig.llm.script_steps(*steps)
                await loop_turn(rig, len(steps) + 1)
            else:
                await rig.ask("what does it say")
            await rig.until(
                lambda n=i + 1: rig.state.state is TurnState.IDLE and len(rig.lines()) == n,
                f"T-0126: turn {i + 1} never ended",
            )
            await rig.advance(300)
        snap = snapshot(rig)
    return rig, snap


def eot_free(rig: ToolRig, snap: dict[str, Any], what: str, words: tuple[str, ...] = ()) -> None:
    """No member of ``SPECIAL_TOKENS ∪ _TRIM_TOKENS`` (and none of ``words``) in any sink (T-0124's helper)."""
    assert_no_literal(rig, snap, f"(T-0126) {what}", words=words)


def cumulative_prefixes(texts: list[str], expected: str, what: str) -> None:
    acc = ""
    for t in texts:
        acc += t
        assert expected.startswith(acc), (
            f"T-0126 {what}: the cumulative release {acc!r} is not a prefix of D(raw) {expected!r} (a prefix that "
            f"D(raw) later cuts was submitted); commits {texts}"
        )


# =====================================================================================================
# Criterion 1: each literal, loop off
# =====================================================================================================


@pytest.mark.parametrize("lit", EOT)
async def test_each_eot_literal_is_neither_stored_nor_published(tmp_path, lit):
    """Criterion 1 (mutations 1, delete instead of cut, which stores and voices ``now``; 7, cut in flush only; 9,
    cut ``spoken_prefix`` but not ``feed``): one commit ``Press {lit} now. `` stores, plays and publishes exactly
    ``Press `` in every sink; no literal and no ``now`` reaches any of them, turns.jsonl or the memory episodes; the
    voice is ``Press``, as at the base; the machine receives exactly ``["Press "]``."""
    rig, snap = await run_on(tmp_path, [[answer(press(lit))]])
    assert voice(rig) == "Press", (
        f"T-0126 criterion 1 ({lit}): the voice is unchanged: {voice(rig)!r}"
    )
    assert_stored_exactly(rig, snap, "Press ", f"(T-0126) criterion 1 ({lit})")
    eot_free(rig, snap, f"criterion 1 ({lit})", words=("now",))
    assert_never_voiced(rig, what=f" (T-0126 criterion 1, {lit})")
    assert_history_clean(rig)
    assert released(rig, "Press ", f"(T-0126) criterion 1 ({lit})") == ["Press "]


async def test_hi_end_of_turn_bye_stores_hi(tmp_path):
    """Criterion 1, last sentence: ``Hi <end_of_turn> bye. `` stores ``Hi `` and voices ``Hi``; no ``bye``."""
    rig, snap = await run_on(tmp_path, [[answer(HI)]])
    assert voice(rig) == "Hi", voice(rig)
    assert_stored_exactly(rig, snap, HI_D, "(T-0126) criterion 1 (Hi)")
    eot_free(rig, snap, "criterion 1 (Hi)", words=("bye",))
    assert released(rig, HI_D, "(T-0126) criterion 1 (Hi)") == [HI_D]


# =====================================================================================================
# Criterion 2: split, spliced, and across a thought commit
# =====================================================================================================

#: case -> commits; every one gives D(raw) = ``Hi `` (by hand: the E literal is whole in C(raw), cut there)
SPLIT_SPLICED: dict[str, tuple[str, ...]] = {
    "split end_of_turn": ("Hi <end_of", "_turn> bye. "),
    "split endoftext": ("Hi <|endof", "text|> bye. "),
    # C deletes ``<pad>`` (I4), which splices ``<end_of_turn>``; the cut then applies as if written in full
    "spliced": ("Hi <end_of<pad>_turn> bye. ",),
    # the same splice across commits: the hold must cover ``<end_of<pa``
    "spliced split": ("Hi <end_of<pa", "d>_turn> bye. "),
}


@pytest.mark.parametrize("case", list(SPLIT_SPLICED))
async def test_split_and_spliced_eot(tmp_path, case):
    """Criterion 2 (adversarial; mutations 2, cut before C, which misses the splice; 4, hold alphabet
    ``_LITERAL_CHARS`` only, which releases ``<end_of``/``<|endof`` at once because ``f``/``x`` are in no tokenizer
    literal; 5, no hold): stores ``Hi ``; no ``LlmCommit`` the machine receives holds an E literal or a prefix that
    D(raw) later cuts (each cumulative release is a prefix of D(raw)); the machine receives exactly ``["Hi "]``."""
    commits = SPLIT_SPLICED[case]
    rig, snap = await run_on(tmp_path, [[answer(t) for t in commits]])
    texts = released(rig, HI_D, f"(T-0126) criterion 2 ({case})")
    cumulative_prefixes(texts, HI_D, f"criterion 2 ({case})")
    assert texts == [HI_D], f"T-0126 criterion 2 ({case}): voiced commits {texts}, expected ['Hi ']"
    assert_stored_exactly(rig, snap, HI_D, f"(T-0126) criterion 2 ({case})")
    eot_free(rig, snap, f"criterion 2 ({case})", words=("bye",))


async def test_eot_split_across_a_thought_commit(tmp_path):
    """Criterion 2, I15 (mutation 16: feed ``thought_open=True`` commits to the cleaner, or flush or reset the hold on
    them): ``Hi <end_of`` / a ``thought_open=True`` commit ``draft`` / ``_turn> bye. `` stores ``Hi ``; the open
    commit reaches the machine as sent; the held ``<end_of`` is neither released nor forgotten by it."""
    rig, snap = await run_on(
        tmp_path, [[answer("Hi <end_of"), open_answer("draft"), answer("_turn> bye. ")]]
    )
    opens = [
        (e.text, e.thought_open)
        for e in rig.machine.events(E.LlmCommit)
        if e.request_id == GEN and e.thought_open
    ]
    assert opens == [("draft", True)], (
        f"T-0126 criterion 2 (I15): the open commit reaches the machine as sent; got {opens}"
    )
    texts = released(rig, HI_D, "(T-0126) criterion 2 (I15)")
    cumulative_prefixes(texts, HI_D, "criterion 2 (I15)")
    assert texts == [HI_D], texts
    assert_stored_exactly(rig, snap, HI_D, "(T-0126) criterion 2 (I15)")
    eot_free(rig, snap, "criterion 2 (I15)", words=("bye",))


# =====================================================================================================
# Criterion 3: stopped means stopped
# =====================================================================================================


async def test_nothing_after_the_cut_is_submitted(tmp_path):
    """Criterion 3: ``First. <end_of_turn>Second. `` / ``Third. `` stores ``First. `` and voices ``First.``; nothing
    after the cut reaches the machine, before or at ``llm.done`` (no ``Second``, no ``Third`` in any commit)."""
    rig, snap = await run_on(tmp_path, [[answer(t) for t in STOP]])
    assert voice(rig) == "First.", f"T-0126 criterion 3: the voice is unchanged: {voice(rig)!r}"
    every = [e.text for e in rig.machine.events(E.LlmCommit) if e.request_id == GEN]
    assert not [t for t in every if "Second" in t or "Third" in t], (
        f"T-0126 criterion 3: text after the cut was submitted: {every}"
    )
    assert released(rig, STOP_D, "(T-0126) criterion 3") == [STOP_D]
    assert_stored_exactly(rig, snap, STOP_D, "(T-0126) criterion 3")
    eot_free(rig, snap, "criterion 3", words=("Second", "Third"))


#: reviewer B's N1 row: C(raw) is the raw text (``<|to`` is no whole tokenizer literal, so C deletes nothing), and
#: D(raw) cuts it at ``<end_of_turn>``, leaving a tail ``<|to`` that is a prefix of ``<|tool_call>``. By hand.
HELD_RAW = "A <|to<end_of_turn> b. "
HELD_D = "A <|to"


async def test_held_tail_after_a_cut_is_stored(tmp_path):
    """Criteria 1, 3 and 9 at session level (reviewer B's N1; mutation 3, tools on routed through
    ``_LiteralCleaner``): ``A <|to<end_of_turn> b. `` stores exactly D(raw) = ``A <|to`` in the exact sinks
    (``TtsSpeak.source``, plays, ``played_text``, history, ``astra.spoken.played``), the machine receives exactly
    ``["A <|to"]``, and nothing after the cut (``b``) reaches any sink. The tail ``<|to`` is what a second, gate-rule
    stage would hold and never flush, so a chained ``_LiteralCleaner`` stores ``A `` here. The I4 row
    ``Press <|tool_call> now.`` cannot see that mutant: C deletes the whole literal before any second stage."""
    rig, snap = await run_on(tmp_path, [[answer(HELD_RAW)]])
    exact = sinks(rig, snap)["exact"]
    bad = {k: v for k, v in exact.items() if v != HELD_D}
    assert not bad, (
        f"T-0126 held tail: stored/played text must be exactly {HELD_D!r} = D(raw); got {bad}"
    )
    texts = released(rig, HELD_D, "(T-0126) held tail")
    assert texts == [HELD_D], f"T-0126 held tail: voiced commits {texts}, expected {[HELD_D]}"
    flat = {k: v for g in sinks(rig, snap).values() for k, v in g.items() if isinstance(v, str)}
    leaked = {k: v for k, v in flat.items() if "<end_of_turn>" in v or re.search(r"\bb\b", v)}
    assert not leaked, f"T-0126 held tail: the cut text reached {leaked}"


# =====================================================================================================
# Criterion 4: C first; I4 unchanged
# =====================================================================================================


def _note(request_id: str, step: int, chars: int) -> dict[str, Any]:
    return {"what": NOTE_WHAT, "request_id": request_id, "step": step, "chars": chars, "cut": True}


def _thought_note(request_id: str, step: int, chars: int, unclosed: bool) -> dict[str, Any]:
    return {
        "what": THOUGHT_NOTE_WHAT,
        "request_id": request_id,
        "step": step,
        "chars": chars,
        "unclosed": unclosed,
    }


#: case -> (raw, D(raw), T-0117's note as at the base, the T-0126 notes), by hand
C_FIRST: dict[str, tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]] = {
    # C removes the block, and the E literal inside it: nothing left to cut, no T-0126 note.
    # len("<|channel>a<end_of_turn>b<channel|>") = 10 + 1 + 13 + 1 + 10 = 35
    "block": (
        "Here <|channel>a<end_of_turn>b<channel|> ok. ",
        "Here  ok. ",
        [_thought_note(GEN, 1, 35, False)],
        [],
    ),
    # C truncates at the open nothing closes (T-0117: len("<|channel>thought SECRET-THOUGHT") = 10 + 8 + 14 = 32,
    # unclosed), leaving ``Okay <end_of_turn> ``; the cut then removes len("<end_of_turn> ") = 14
    "unclosed": (
        f"Okay <end_of_turn> <|channel>thought {SECRET}",
        "Okay ",
        [_thought_note(GEN, 1, 32, True)],
        [_note(GEN, 1, len("<end_of_turn> "))],
    ),
}


@pytest.mark.parametrize(
    "case",
    [
        # green at the base (C already removes the block): the stage-order guard (mutation 2 gives ``Here ``)
        pytest.param("block"),
        pytest.param("unclosed"),
    ],
)
async def test_c_runs_before_the_cut(tmp_path, case):
    """Criterion 4 (adversarial; mutation 2, cut before C: ``Here `` for the block, and no T-0117 note for the
    unclosed case; mutation 15, merge the count into T-0117's note): C runs first, so an E literal inside a closed
    block goes with the block (stores and voices ``Here  ok. ``, T-0117's note as at the base, no T-0126 note); an E
    literal before an unclosed open is cut after C (stores ``Okay ``, T-0117's note ``unclosed`` true, the T-0126
    note ``chars == len("<end_of_turn> ")``)."""
    raw, expected, thought_notes, eot_notes = C_FIRST[case]
    rig, snap = await run_on(tmp_path, [[answer(raw)]])
    assert_stored_exactly(rig, snap, expected, f"(T-0126) criterion 4 ({case})")
    released(rig, expected, f"(T-0126) criterion 4 ({case})")
    eot_free(rig, snap, f"criterion 4 ({case})")
    leaks = {k: v for k, v in snap.items() if isinstance(v, str) and k != "HUD" and SECRET in v}
    assert not leaks and SECRET not in snap["HUD"], (
        f"T-0126 criterion 4 ({case}): thought text in {list(leaks)}"
    )
    assert notes(rig, THOUGHT_NOTE_WHAT) == thought_notes, (
        f"T-0126 criterion 4 ({case}): T-0117's note is as at the base; got {notes(rig, THOUGHT_NOTE_WHAT)}"
    )
    assert notes(rig, NOTE_WHAT) == eot_notes, (
        f"T-0126 criterion 4 ({case}): T-0126 notes {notes(rig, NOTE_WHAT)}, expected {eot_notes}"
    )


#: raw -> (C(raw) = D(raw), the voice): I4 deletes tokenizer literals with tools on; no E literal, nothing cut
I4_ROWS = {
    "tool_call": ("Press <|tool_call> now. ", "Press  now. ", "Press now."),
    "eos": ("Hi <eos> bye. ", "Hi  bye. ", "Hi bye."),
}


@pytest.mark.parametrize("case", list(I4_ROWS))
async def test_i4_rows_unchanged(tmp_path, case):
    """Criterion 4, I4 (green at the base, no marker; mutation 3, cut at every ``_TRIM_TOKENS`` member or route
    tools-on through ``_LiteralCleaner``, which stores ``Press `` / ``Hi ``; mutation 14, count I4's deletions):
    ``Press <|tool_call> now. `` stores ``Press  now. `` and voices ``Press now.``; ``Hi <eos> bye. `` stores
    ``Hi  bye. ``; no T-0126 note."""
    raw, stored, spoken = I4_ROWS[case]
    rig, snap = await run_on(tmp_path, [[answer(raw)]])
    assert voice(rig) == spoken, voice(rig)
    assert_stored_exactly(rig, snap, stored, f"(T-0126) criterion 4, I4 ({case})")
    assert notes(rig, NOTE_WHAT) == [], notes(rig, NOTE_WHAT)


# =====================================================================================================
# Criterion 5: every voiced step, loop on
# =====================================================================================================


async def test_loop_on_step1_and_report_step(tmp_path):
    """Criterion 5 (mutation 11, note at step 1 only): under ``TurnPolicy(tool_loop=True)``, a step-1 final answer
    ``Hi <end_of_turn> bye. `` gives criterion 1's result; in a two-step search turn the step-2 report
    ``Found it: the café opens at nine. <end_of_turn>user: and tomorrow? `` voices and stores ``Found it: the café
    opens at nine. ``; turn 2's request (which replays history) holds neither ``<end_of_turn>`` nor ``tomorrow``; the
    T-0126 note carries ``"step": 2``."""
    # (a) step-1 final answer
    rig, snap = await run_on(tmp_path / "step1", [[answer(HI)]], policy=loop_policy())
    assert voice(rig) == "Hi", voice(rig)
    assert_stored_exactly(rig, snap, HI_D, "(T-0126) criterion 5 (step 1)")
    eot_free(rig, snap, "criterion 5 (step 1)", words=("bye",))
    assert notes(rig, NOTE_WHAT) == [_note(GEN, 1, len("<end_of_turn> bye. "))], notes(
        rig, NOTE_WHAT
    )

    # (b) the report step, then a second turn
    search = FakeSearch()
    async with tool_rig(
        tmp_path / "report",
        policy=loop_policy(),
        search=search,
        script=[answer(PREFIX), tool(SEARCH_CALL)],
    ) as rig:
        rig.llm.script_steps([answer(REPORT_RAW)])
        await loop_turn(rig, 2)
        snap = snapshot(rig)
        report = answer_commits(rig, "t1-gen-s2")
        rig.llm.script = [answer("Okay. ")]
        await rig.ask("and when does it close")
        await rig.until(lambda: len(rig.lines()) == 2, "turn 2 never ended")
        t2 = [p for rid, p, _ in rig.llm.generates if "-t2-gen" in rid]
    assert search.queries, "premise: the search ran (a two-step turn)"
    assert "".join(report) == REPORT_D, f"T-0126 criterion 5: the report's commits {report}"
    cumulative_prefixes(report, REPORT_D, "criterion 5 (report)")
    assert snap["TtsSpeak.source"] == PREFIX + REPORT_D, (
        f"T-0126 criterion 5: voiced {PREFIX + REPORT_D!r}; got {snap['TtsSpeak.source']!r}"
    )
    assert snap["history"] == f"{PREFIX} {REPORT_D}", (
        f"T-0126 criterion 5: history {PREFIX + ' ' + REPORT_D!r}; got {snap['history']!r}"
    )
    for key in ("TtsSpeak.source", "TtsSpeak.text", "astra.spoken", "audio plays", "history (all)",
                "turns.jsonl", "memory episodes"):  # fmt: skip
        v = snap[key] or ""
        bad = [s for s in (*LITERALS, "tomorrow", "user:") if s in v]
        assert not bad, f"T-0126 criterion 5: {bad} reached {key}: {v[:300]!r}"
    assert t2, "premise: turn 2 sent a request"
    for prompt in t2:
        dumped = json.dumps(prompt.model_dump(mode="json"), ensure_ascii=False)
        bad = [s for s in ("<end_of_turn>", "tomorrow") if s in dumped]
        assert not bad, f"T-0126 criterion 5: turn 2's request replays {bad}"
    assert notes(rig, NOTE_WHAT) == [_note("t1-gen-s2", 2, len(REPORT_CUT))], (
        f"T-0126 criterion 5: one note, for step 2; got {notes(rig, NOTE_WHAT)}"
    )


# =====================================================================================================
# Criterion 6: the call path, loop off (H-T126-3)
# =====================================================================================================


def _continued_text(body: dict[str, Any]) -> str:
    content = body["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


@pytest.mark.parametrize("draft", ["valid", "repairable"])
async def test_spoken_prefix_and_repair_seed_cut(tmp_path, draft):
    """Criterion 6 (H-T126-3; mutations 8, cut in ``feed`` but not in ``spoken_prefix``; 17, refuse the call after a
    cut): step 1 ``answer("Let me look. <end_of_turn>junk ")`` plus a call voices and stores ``Let me look. ``; every
    ``tool.call`` record's ``spoken_prefix`` and ``call.spoken_prefix`` are D(raw) = ``Let me look. ``; with a
    repairable draft the repair continues ``Let me look. <|tool_call>``; the card states and the execution equal the
    control turn's (the same call after a prefix with no E literal)."""
    if draft == "valid":
        call, continuation = VALID_CLICK, None
    else:
        call, continuation = REPAIRABLE_CLICK, [tool(REPAIRED_CLICK)]
    judged: list[str] = []
    rig, snap = await run_on(
        tmp_path / "eot", [[answer(CALL_RAW), tool(call)]], continuation=continuation, judged=judged
    )
    control, csnap = await run_on(
        tmp_path / "control", [[answer(CALL_D), tool(call)]], continuation=continuation
    )
    assert csnap["history"] == CALL_D, (
        f"premise (control): stores {CALL_D!r}; got {csnap['history']!r}"
    )
    assert control.statuses(CALL_ID) and control.statuses(CALL_ID)[-1] == "dry_run", (
        f"premise (control): the call ends dry_run; {control.statuses(CALL_ID)}"
    )
    if draft == "repairable":
        cseeds = [_continued_text(b) for b in control.llm.continuation_bodies()]
        assert cseeds == [f"{CALL_D}<|tool_call>"], (
            f"premise (control): the repair continues the spoken prefix plus the tag; got {cseeds}"
        )
    assert_stored_exactly(rig, snap, CALL_D, f"(T-0126) criterion 6 ({draft})")
    eot_free(rig, snap, f"criterion 6 ({draft})", words=("junk",))
    records = rig.tool_records()
    assert records, "premise: the call was judged (tool.call records written)"
    prefixes = [r.get("spoken_prefix") for r in records]
    assert all(p == CALL_D for p in prefixes), (
        f"T-0126 criterion 6: tool.call spoken_prefix is D(raw) {CALL_D!r} on every record; got {prefixes}"
    )
    assert judged and all(p == CALL_D for p in judged), (
        f"T-0126 criterion 6: call.spoken_prefix == D(raw) {CALL_D!r}; _judge_tool received {judged}"
    )
    # the verdict, card states and execution are the control's (the call is judged and executed as at the base)
    assert rig.statuses(CALL_ID) == control.statuses(CALL_ID), (
        f"T-0126 criterion 6: card states {rig.statuses(CALL_ID)}, control {control.statuses(CALL_ID)}"
    )
    assert rig.statuses(CALL_ID)[-1] == "dry_run", rig.statuses(CALL_ID)
    ran = [(i.name, dict(i.arguments)) for i in rig.executor.invocations]
    assert ran == [(i.name, dict(i.arguments)) for i in control.executor.invocations] and ran, (
        f"T-0126 criterion 6: execution {ran}, control {control.executor.invocations}"
    )
    if draft == "repairable":
        bodies = rig.llm.continuation_bodies()
        assert bodies, "premise: a repair request was sent"
        seeds = [_continued_text(b) for b in bodies]
        assert seeds == [f"{CALL_D}<|tool_call>"], (
            f"T-0126 criterion 6: the repair's continued assistant text is {CALL_D + '<|tool_call>'!r}; got {seeds}"
        )


# =====================================================================================================
# Criterion 7: all of it cut
# =====================================================================================================


@pytest.mark.parametrize("policy", ["loop off", "loop on"])
async def test_entirely_cut_answer_ends_the_turn(tmp_path, policy):
    """Criterion 7 (adversarial; M26: a commit that cleans to nothing is not submitted): with tools on,
    ``answer("<end_of_turn> bye. ")`` alone synthesises nothing, plays nothing but clips, stores ``""``, reaches IDLE
    and writes the ``outcome`` a tools-on ``answer("<pad>")`` writes (``silent``)."""
    pol = LOOP_OFF if policy == "loop off" else loop_policy()
    rig, snap = await run_on(tmp_path / "cut", [[answer("<end_of_turn> bye. ")]], policy=pol)
    control, _ = await run_on(tmp_path / "pad", [[answer("<pad>")]], policy=pol)
    pad_outcome = json.loads(control.lines()[-1])["outcome"]
    assert pad_outcome == "silent", (
        f"premise (control): a tools-on <pad> turn is silent; got {pad_outcome!r}"
    )
    assert rig.state.state is TurnState.IDLE, rig.state.state
    assert rig.tts.texts() == [], f"T-0126 criterion 7: synthesised {rig.tts.texts()}"
    played = [s for _, s in rig.audio.plays if s]
    assert played == [], (
        f"T-0126 criterion 7: played {played} (only clips, with no source, may play)"
    )
    assert rig.state.history, "T-0126 criterion 7: the turn is stored (with nothing spoken)"
    assert rig.state.history[-1].assistant_spoken == "", (
        f"T-0126 criterion 7: history {rig.state.history[-1].assistant_spoken!r}"
    )
    outcome = json.loads(rig.lines()[-1])["outcome"]
    assert outcome == pad_outcome, (
        f"T-0126 criterion 7: outcome {outcome!r}, the <pad> turn's {pad_outcome!r}"
    )
    # The observables above already hold at 4a25a85 (the gate latches an empty speak that is never synthesised);
    # what is red there is that the raw commit still reaches the machine and its TtsSpeak.source holds the literal
    # and ``bye``. The commit cleans to nothing, so it is not submitted (M26; the task's implementation notes).
    assert answer_commits(rig, GEN) == [], (
        f"T-0126 criterion 7 (M26): a commit that cleans to nothing is not submitted; got "
        f"{answer_commits(rig, GEN)}"
    )
    assert "".join(c.source for c in rig.speaks()) == "", [c.source for c in rig.speaks()]
    eot_free(rig, snap, "criterion 7", words=("bye",))


# =====================================================================================================
# Criterion 8: the note
# =====================================================================================================

#: case -> (one script per turn, loop steps or (), the expected T-0126 notes, the expected T-0117 notes, the cut
#: texts for the no-text check). ``chars`` = len(C(raw)) - len(D(raw)), by hand: never what C removed.
NOTE_CASES: dict[str, tuple[list[list[Commit]], tuple[list[Commit], ...], list[dict[str, Any]],
                            list[dict[str, Any]], tuple[str, ...]]] = {
    **{
        f"press {lit}": ([[answer(press(lit))]], (), [_note(GEN, 1, len(f"{lit} now. "))], [], (f"{lit} now. ",))
        for lit in EOT
    },
    # len("<end_of_turn> bye. ") = 13 + 6 = 19
    "hi": ([[answer(HI)]], (), [_note(GEN, 1, 19)], [], ("<end_of_turn> bye. ",)),
    "split": ([[answer("Hi <end_of"), answer("_turn> bye. ")]], (), [_note(GEN, 1, 19)], [],
              ("<end_of_turn> bye. ",)),
    # C deletes ``<pad>`` first (I4; not counted: mutation 14), then the cut removes 19
    "spliced": ([[answer("Hi <end_of<pad>_turn> bye. ")]], (), [_note(GEN, 1, 19)], [], ("<end_of_turn> bye. ",)),
    # T-0117's note first (32, unclosed), then T-0126's (14): two notes (mutation 15)
    "both": ([[answer(f"Okay <end_of_turn> <|channel>thought {SECRET}")]], (), [_note(GEN, 1, 14)],
             [_thought_note(GEN, 1, 32, True)], ("<end_of_turn> ", SECRET)),
    # D(raw) = ``A <|to`` ends in a tail the hold may keep to the end; the note comes after it.
    # len("<end_of_turn> b. ") = 13 + 4 = 17
    "held tail": ([[answer("A <|to<end_of_turn> b. ")]], (), [_note(GEN, 1, 17)], [], ("<end_of_turn> b. ",)),
    # before the call is judged: before the tool.call record and llm.done. len("<end_of_turn>junk ") = 18
    "call": ([[answer(CALL_RAW), tool(VALID_CLICK)]], (), [_note(GEN, 1, 18)], [], (CALL_CUT,)),
    # loop on: one note per stream, step 1 and the report step each with a cut (mutation 11)
    "both steps": ([[answer(CALL_RAW), tool(SEARCH_CALL)]], ([answer(REPORT_RAW)],),
                   [_note(GEN, 1, 18), _note("t1-gen-s2", 2, len(REPORT_CUT))], [], (CALL_CUT, REPORT_CUT)),
    # one note per stream: two turns. len("<|endoftext|> bye. ") = 13 + 6 = 19
    "two turns": ([[answer(press("<end_of_turn>"))], [answer("Hi <|endoftext|> bye. ")]], (),
                  [_note(GEN, 1, len("<end_of_turn> now. ")), _note("t2-gen", 1, 19)], [],
                  ("<end_of_turn> now. ", "<|endoftext|> bye. ")),
    # nothing cut: no T-0126 note (mutations 12, 14); green at the base
    "clean": ([[answer("Sure, here it is. ")]], (), [], [], ()),
    "i4 only": ([[answer("Press <|tool_call> now. "), answer("Hi <eos> bye. ")]], (), [], [], ()),
    "thought only": ([[answer(f"Sure, <|channel>thought {SECRET}<channel|>here it is. ")]], (), [],
                     [_thought_note(GEN, 1, 42, False)], ()),
    "c first": ([[answer("Here <|channel>a<end_of_turn>b<channel|> ok. ")]], (), [],
                [_thought_note(GEN, 1, 35, False)], ()),
}  # fmt: skip
NOTE_GREEN = frozenset({"clean", "i4 only", "thought only", "c first"})


def _grams(text: str, n: int = 4) -> set[str]:
    return {text[i : i + n] for i in range(len(text) - n + 1)}


@pytest.mark.parametrize("case", [pytest.param(c) for c in NOTE_CASES])
async def test_note_counts_chars_never_text(tmp_path, case):
    """Criterion 8 (H-T126-2; mutations 11-15): exactly one session-log note per stream where the cut removed
    anything, exactly ``{what, request_id, step, chars, cut: true}`` with hand-computed values; no value but ``what``
    holds a 4-character substring of the cut text; written after the stream's last answer commit (the held tail),
    after T-0117's note, before the ``tool.call`` record and ``llm.done``; none when nothing was cut and none for
    I4's deletions; T-0117's note unchanged; ``llm.commit`` keeps the raw text."""
    scripts, steps, expected, thought_expected, cut_texts = NOTE_CASES[case]
    trace: list[Any] = []
    search = FakeSearch() if steps else None
    rig, _ = await run_on(
        tmp_path,
        scripts,
        policy=loop_policy() if steps else LOOP_OFF,
        steps=steps,
        search=search,
        trace=trace,
    )
    found = notes(rig, NOTE_WHAT)
    assert found == expected, f"T-0126 criterion 8 ({case}): notes {found}, expected {expected}"
    assert notes(rig, THOUGHT_NOTE_WHAT) == thought_expected, (
        f"T-0126 criterion 8 ({case}): T-0117's note is unchanged; got {notes(rig, THOUGHT_NOTE_WHAT)}"
    )
    for note in found:
        assert set(note) == NOTE_KEYS, sorted(note)
        values = json.dumps([v for k, v in note.items() if k != "what"], ensure_ascii=False)
        for text in cut_texts:
            hits = sorted(g for g in _grams(text) if g in values)
            assert not hits, (
                f"T-0126 criterion 8 ({case}): the note holds stream text {hits}: {note}"
            )
        rid = note["request_id"]
        turn = rid.split("-")[0]
        mine = [i for i, t in enumerate(trace) if t[0] == "log" and t[1] == "note" and t[2] == note]
        assert len(mine) == 1, mine
        done = [i for i, t in enumerate(trace)
                if t[0] == "log" and t[1] == "llm.done" and t[2].get("request_id") == rid]  # fmt: skip
        commits = [i for i, t in enumerate(trace)
                   if t[0] == "submit" and isinstance(t[1], E.LlmCommit) and t[1].request_id == rid
                   and t[1].segment == "answer"]  # fmt: skip
        theirs = [i for i, t in enumerate(trace)
                  if t[0] == "log" and t[1] == "note" and t[2].get("what") == THOUGHT_NOTE_WHAT
                  and t[2].get("request_id") == rid]  # fmt: skip
        calls = [i for i, t in enumerate(trace)
                 if t[0] == "log" and t[1] == "tool.call" and str(t[2].get("call_id", "")).startswith(f"{turn}-")
                 and t[2].get("call_id") == (f"{turn}-call" if note["step"] == 1 else
                                             f"{turn}-call-s{note['step']}")]  # fmt: skip
        assert commits and done and commits[-1] < mine[0] < done[0], (
            f"T-0126 criterion 8 ({case}): the note comes after the stream's last answer commit and before "
            f"llm.done; commits {commits}, note {mine}, llm.done {done}"
        )
        assert all(i < mine[0] for i in theirs), (
            f"T-0126 criterion 8 ({case}): T-0117's note comes first; T-0117 {theirs}, T-0126 {mine}"
        )
        assert all(mine[0] < i for i in calls), (
            f"T-0126 criterion 8 ({case}): the note comes before the tool.call record; note {mine}, tool.call {calls}"
        )
        if case == "call":
            assert calls, "premise: the call was recorded"
    for k, script in enumerate(scripts, start=1):
        logged = "".join(
            r["text"]
            for r in rig.kind("llm.commit")
            if r.get("segment") == "answer" and r.get("request_id") == f"t{k}-gen"
        )
        assert logged == "".join(c.text for c in script if c.segment == "answer"), (
            "T-0126 criterion 8: llm.commit keeps the raw text verbatim"
        )
