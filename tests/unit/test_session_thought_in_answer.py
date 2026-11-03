"""T-0117: a thought block inside a ``thought_open=False`` answer commit is never voiced or stored (tools on).

Session-level rows for criteria 2-6, 8 and 12 of docs/tasks/T-0117.md, over the T-0092c/T-0114 rig
(``tests/unit/_tool_turn_support.tool_rig``, the fake LLM, the recording executor, ``FakeSearch``, the
``ScriptedClock``). Every expected text is written out by hand from criterion 1 (C(raw)) in the row; nothing is
read from ``_AnswerCleaner`` or from the router to get an expected value. Where a row also calls
``router.route``/``router.template_content``, it is to check that the hand-written expectation agrees with the
router's and the pinned template's thought rule (H-T117-3), never as the oracle of the session.

``test_the_word_thought_is_voiced`` (criterion 12) and the ``clean`` case of the note row hold at the base: they
are the guard-too-wide rows (``.claude/rules/tests.md``); the other rows need the fix.

Readings where the task is silent (listed in the test-author's report):

- "``TtsSpeak.text`` and the ``astra.spoken`` texts join to exactly X": those strings are the TTS-normalised
  spelling (trailing whitespace stripped, as on the base), so they are compared whitespace-normalised
  (``" ".join(s.split())``); ``TtsSpeak.source``, the audio plays' sources and history are compared byte-exact;
- the note's ``what`` is the criterion-8 literal, whose own word "thought" is not thought text: the
  "no 4-character substring of the thought" check runs over every other value of the note, and ``what`` is
  compared to the literal;
- "no later step request" (criterion 5): a two-step turn has no step after the report, so the row asks a second
  turn and checks its request (which replays history).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``dry_run``, n = 1 and
deterministic per row. No measurement.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.turn_machine import TurnPolicy
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.toolcall import router
from tests.fakes.search import FakeSearch
from tests.unit._tool_turn_support import (
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
from tests.unit.test_session_tool_loop import (
    PREFIX,
    SEARCH_CALL,
    history_text,
    loop_policy,
    loop_turn,
    notes,
    step_request,
)

SECRET = "SECRET-THOUGHT"
OPEN, CLOSE = "<|channel>", "<channel|>"
#: what criterion 2 names as never voiced or stored
LEAKS = (SECRET, "thought SECRET", OPEN, CLOSE)
NOTE_WHAT = "thought text in an answer commit was not voiced"
NOTE_KEYS = frozenset({"what", "request_id", "step", "chars", "unclosed"})

LOOP_OFF = TurnPolicy(thinking=True)

# --- the raw texts and their C(raw), written out by hand from criterion 1 ---------------------------------

BLOCK2 = f"{OPEN}thought {SECRET}{CLOSE}"
RAW2 = f"Sure, {BLOCK2}here it is. "
#: criterion 2: the closed block removed, nothing inserted, the bytes around it kept
CLEAN2 = "Sure, here it is. "

RAW3 = f"Let me look. {BLOCK2}"
CLEAN3 = "Let me look. "

RAW5 = f"Found it: {BLOCK2}the café opens at nine. "
CLEAN5 = "Found it: the café opens at nine. "

TAIL6 = f"{OPEN}thought {SECRET}"
RAW6 = f"Okay. {TAIL6}"
CLEAN6 = "Okay. "

#: criterion 6, second half: ``Okay. <|channel>x`` / ``y<channel|> fine. `` -> ``Okay.  fine. ``
RAW6B = ("Okay. <|channel>x", "y<channel|> fine. ")
CLEAN6B = "Okay.  fine. "


# --- running a turn and reading every sink ---------------------------------------------------------------


def _norm(s: str) -> str:
    return " ".join(s.split())


def answer_commits(rig: ToolRig, request_id: str) -> list[str]:
    """The voiced answer texts the machine received for ``request_id`` (what the cleaner released)."""
    return [
        e.text
        for e in rig.machine.events(E.LlmCommit)
        if e.request_id == request_id and e.segment == "answer" and not e.thought_open
    ]


def snapshot(rig: ToolRig) -> dict[str, Any]:
    """Every sink criterion 2 names, read inside the rig (the session log is read after ``aclose``)."""
    speaks = rig.speaks()
    spoken = [m for m in rig.published.messages if getattr(m, "type", None) == "astra.spoken"]
    return {
        "TtsSpeak.source": "".join(c.source for c in speaks),
        "TtsSpeak.text": _norm(" ".join(c.text for c in speaks)),
        "astra.spoken": _norm(" ".join(m.text for m in spoken)),
        "audio plays": "".join(s for _, s in rig.audio.plays if s),
        "history": rig.state.history[-1].assistant_spoken if rig.state.history else None,
        "history (all)": history_text(rig),
        "HUD": json.dumps([m.model_dump(mode="json") for m in rig.published.messages]),
        "turns.jsonl": "\n".join(rig.lines()),
        "memory episodes": repr(rig.memory.episodes),
    }


def assert_voiced_exactly(snap: dict[str, Any], expected: str, what: str) -> None:
    """The positive half: the words around the block are voiced and stored, byte for byte."""
    exact = {k: snap[k] for k in ("TtsSpeak.source", "audio plays", "history")}
    bad = {k: v for k, v in exact.items() if v != expected}
    assert not bad, f"T-0117 {what}: voiced/stored text must be exactly {expected!r}; got {bad}"
    normed = {k: snap[k] for k in ("TtsSpeak.text", "astra.spoken")}
    bad = {k: v for k, v in normed.items() if v != _norm(expected)}
    assert not bad, (
        f"T-0117 {what}: the TTS/HUD spelling must be {_norm(expected)!r} (whitespace-normalised); got {bad}"
    )


def assert_no_leak(snap: dict[str, Any], what: str, *, extra: dict[str, str] | None = None) -> None:
    sinks = {k: v for k, v in snap.items() if isinstance(v, str)}
    sinks.update(extra or {})
    bad = {k: [s for s in LEAKS if s in v] for k, v in sinks.items()}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"T-0117 {what}: thought text or a channel marker reached {bad}"


async def run_turn(
    tmp_path: Path,
    script: Sequence[Commit],
    *,
    policy: TurnPolicy = LOOP_OFF,
    steps: Sequence[Sequence[Commit]] = (),
    search: Any = None,
    continuation: Sequence[Commit] | None = None,
) -> tuple[ToolRig, dict[str, Any]]:
    async with tool_rig(
        tmp_path, script=script, policy=policy, search=search, continuation=continuation
    ) as rig:
        if steps:
            rig.llm.script_steps(*steps)
            await loop_turn(rig, len(steps) + 1)
        else:
            await rig.ask("what does it say")
            await rig.turn_done()
        snap = snapshot(rig)
    return rig, snap


def tool_call_records(rig: ToolRig) -> list[dict[str, Any]]:
    return rig.kind("tool.call")


# =====================================================================================================
# Criterion 2: voiced step 1, loop off
# =====================================================================================================


async def test_step1_loop_off_closed_thought_not_voiced_or_stored(tmp_path):
    """Criterion 2 (P4; CLAUDE.md): tools on, loop off, ``answer("Sure, <|channel>thought SECRET-THOUGHT<channel|>
    here it is. ")`` and no call voices and stores exactly ``Sure, here it is. `` everywhere, and the thought is in no
    sink. Reviewer mutations 2 (today's behaviour), 5 and 6."""
    rig, snap = await run_turn(tmp_path, [answer(RAW2)])
    assert_voiced_exactly(snap, CLEAN2, "criterion 2")
    assert_no_leak(snap, "criterion 2")
    assert_never_voiced(rig, what=" (T-0117 criterion 2)")
    assert_history_clean(rig)
    assert "".join(answer_commits(rig, "t1-gen")) == CLEAN2, (
        f"T-0117 criterion 2: the machine's answer commits join to C(raw) {CLEAN2!r}; got "
        f"{answer_commits(rig, 't1-gen')}"
    )
    # Criterion 8, last item: thinking stays logged verbatim in llm.commit (.claude/rules/conductor.md)
    assert (
        "".join(r["text"] for r in rig.kind("llm.commit") if r.get("segment") == "answer") == RAW2
    )


# =====================================================================================================
# Criterion 3: a prefix before a call, and the repair seed
# =====================================================================================================


@pytest.mark.parametrize("draft", ["valid", "repairable"])
async def test_step1_loop_off_prefix_before_call_and_repair_seed_clean(tmp_path, draft):
    """Criterion 3 (H-T117-5): the prefix before a call voices and stores ``Let me look. ``; every ``tool.call``
    record's ``spoken_prefix`` is C(raw) and holds no thought; with a repairable draft the repair request's continued
    assistant text holds none either. Reviewer mutation 9."""
    if draft == "valid":
        script, continuation = [answer(RAW3), tool(VALID_CLICK)], None
    else:
        script, continuation = [answer(RAW3), tool(REPAIRABLE_CLICK)], [tool(REPAIRED_CLICK)]
    rig, snap = await run_turn(tmp_path, script, continuation=continuation)
    assert_voiced_exactly(snap, CLEAN3, f"criterion 3 ({draft})")
    assert_no_leak(snap, f"criterion 3 ({draft})")
    records = tool_call_records(rig)
    assert records, "premise: the call was judged (tool.call records written)"
    prefixes = [r.get("spoken_prefix") for r in records]
    assert all(p == CLEAN3 for p in prefixes), (
        f"T-0117 criterion 3: tool.call spoken_prefix is C(raw) {CLEAN3!r} on every record; got {prefixes}"
    )
    if draft == "repairable":
        bodies = rig.llm.continuation_bodies()
        assert bodies, "premise: a repair request was sent"
        seeds = [json.dumps(b["messages"][-1], ensure_ascii=False) for b in bodies]
        assert not [s for s in seeds if SECRET in s], (
            f"T-0117 criterion 3: the repair's continued assistant text holds the thought: {seeds}"
        )
        assert all("Let me look." in s for s in seeds), (
            f"T-0117 criterion 3: the repair still continues the spoken prefix; got {seeds}"
        )


# =====================================================================================================
# Criteria 4 and 5: loop on
# =====================================================================================================


async def test_step1_loop_on_final_answer_clean(tmp_path):
    """Criterion 4: criterion 2 under ``TurnPolicy(tool_loop=True)`` (a final answer at step 1, no call)."""
    rig, snap = await run_turn(tmp_path, [answer(RAW2)], policy=loop_policy())
    assert_voiced_exactly(snap, CLEAN2, "criterion 4")
    assert_no_leak(snap, "criterion 4")
    assert "".join(answer_commits(rig, "t1-gen")) == CLEAN2


async def test_report_step_clean(tmp_path):
    """Criterion 5: step 1 is a prefix plus a search call; the step-2 report carries a closed thought. The report
    voices ``Found it: the café opens at nine. ``, history is ``"<prefix> <report>"``, and nothing of the thought is
    voiced, stored, published, or in the next request (turn 2's, which replays history)."""
    search = FakeSearch()
    async with tool_rig(
        tmp_path,
        policy=loop_policy(),
        search=search,
        script=[answer(PREFIX), tool(SEARCH_CALL)],
    ) as rig:
        rig.llm.script_steps([answer(RAW5)])
        await loop_turn(rig, 2)
        snap = snapshot(rig)
        report = answer_commits(rig, "t1-gen-s2")
        assert SECRET not in step_request(rig, 2).messages[-1].text(), "step 2's own request"
        rig.llm.script = [answer("Okay. ")]
        await rig.ask("and when does it close")
        await rig.until(lambda: len(rig.lines()) == 2, "turn 2 never ended")
        t2 = [p for rid, p, _ in rig.llm.generates if "-t2-gen" in rid]
    expected = PREFIX + CLEAN5
    assert snap["TtsSpeak.source"] == expected, (
        f"T-0117 criterion 5: voiced {expected!r}; got {snap['TtsSpeak.source']!r}"
    )
    assert snap["history"] == f"{PREFIX} {CLEAN5}", (
        f"T-0117 criterion 5: history {PREFIX + ' ' + CLEAN5!r}; got {snap['history']!r}"
    )
    assert "".join(report) == CLEAN5, f"T-0117 criterion 5: the report's commits {report}"
    assert_no_leak(snap, "criterion 5")
    assert t2, "premise: turn 2 sent a request"
    for prompt in t2:
        dumped = json.dumps(prompt.model_dump(mode="json"), ensure_ascii=False)
        assert SECRET not in dumped, "T-0117 criterion 5: turn 2's request replays the thought"


# =====================================================================================================
# Criterion 6: an open marker nothing closes (adversarial: the buggy-client case)
# =====================================================================================================


async def test_unclosed_thought_at_end_is_dropped(tmp_path):
    """Criterion 6 (P4; adversarial; reviewer mutation 1): the last commit opens a thought and the stream ends,
    ``thought_open=False``. ``Okay. `` is voiced and stored; nothing after the open marker is released, neither while
    streaming nor at the flush; no ``TtsSpeak`` before or after ``llm.done`` holds ``SECRET``."""
    rig, snap = await run_turn(tmp_path, [answer("Okay. "), answer(TAIL6)])
    for c in rig.speaks():
        assert "SECRET" not in c.source and "SECRET" not in c.text, c
    assert answer_commits(rig, "t1-gen") and "".join(answer_commits(rig, "t1-gen")) == CLEAN6, (
        f"T-0117 criterion 6: the released answer text is {CLEAN6!r}; got {answer_commits(rig, 't1-gen')}"
    )
    assert_voiced_exactly(snap, CLEAN6, "criterion 6")
    assert_no_leak(snap, "criterion 6")


async def test_unclosed_in_one_commit_dropped(tmp_path):
    """Criterion 6 as the task writes it: one commit ``Okay. <|channel>thought SECRET-THOUGHT``."""
    rig, snap = await run_turn(tmp_path, [answer(RAW6)])
    assert "".join(answer_commits(rig, "t1-gen")) == CLEAN6, answer_commits(rig, "t1-gen")
    assert_voiced_exactly(snap, CLEAN6, "criterion 6")
    assert_no_leak(snap, "criterion 6")


async def test_unclosed_then_closed_in_a_later_commit_voices_the_rest(tmp_path):
    """Criterion 6, second half (reviewer mutation 5, guard too wide): ``Okay. <|channel>x`` / ``y<channel|> fine. ``
    voices ``Okay.  fine. `` (the double space kept, H-T117-2)."""
    rig, snap = await run_turn(tmp_path, [answer(t) for t in RAW6B])
    assert "".join(answer_commits(rig, "t1-gen")) == CLEAN6B, answer_commits(rig, "t1-gen")
    assert snap["TtsSpeak.source"] == CLEAN6B and snap["history"] == CLEAN6B, (
        f"T-0117 criterion 6: voiced and stored {CLEAN6B!r}; got {snap['TtsSpeak.source']!r} / "
        f"{snap['history']!r}"
    )


# =====================================================================================================
# Criterion 7 at session level: markers split and spliced (adversarial)
# =====================================================================================================

SPLITS = {
    # the most likely failure: stripping per commit instead of over the concatenation (reviewer mutation 3)
    "marker_split": ("Sure, <|chan", f"nel>thought {SECRET}<chan", "nel|>here it is. "),
    # the open in one commit, the close two commits later
    "close_two_later": ("Sure, <|channel>thought ", f"{SECRET} ", "<channel|>here it is. "),
}


@pytest.mark.parametrize("case", list(SPLITS))
async def test_marker_split_across_commits(tmp_path, case):
    """Criterion 7 (adversarial; reviewer mutations 3 and 8): the markers are split across commits. Voiced and
    stored ``Sure, here it is. ``, and no answer commit the machine received (so nothing voiced before
    ``llm.done``) holds ``SECRET``."""
    rig, snap = await run_turn(tmp_path, [answer(t) for t in SPLITS[case]])
    released = answer_commits(rig, "t1-gen")
    assert not [t for t in released if "SECRET" in t], (
        f"T-0117 criterion 7 ({case}): a released answer commit holds the thought: {released}"
    )
    assert "".join(released) == CLEAN2, f"T-0117 criterion 7 ({case}): released {released}"
    assert_voiced_exactly(snap, CLEAN2, f"criterion 7 ({case})")
    assert_no_leak(snap, f"criterion 7 ({case})")


#: case -> (commits, C(raw) by hand)
SPLICES = {
    # ``<eos>`` deleted splices ``<|channel>``, which a later ``<channel|>`` closes
    "literal": ((f"Sure, <|chan<eos>nel>thought {SECRET}<channel|>ok. ",), "Sure, ok. "),
    # removing the block ``<|channel>x<channel|>`` splices ``<|chan`` + ``nel>`` into an open nothing closes
    "block_removal": ((f"Sure, <|chan<|channel>x<channel|>nel>{SECRET}",), "Sure, "),
    # the same splice across commits
    "block_removal_split": (("Sure, <|chan", "<|channel>x<channel|>", f"nel>{SECRET}"), "Sure, "),
    # ruling F1: deleting the stray close splices ``<|chan`` + ``nel>`` into an open nothing closes; it counts as
    # written in full, so nothing after it is voiced (a stray-close deletion left to the final
    # ``drop_special_literals`` voices the secret: reviewer A's M20)
    "stray_close_splice": ((f"Sure, <|chan<channel|>nel>{SECRET}",), "Sure, "),
    "stray_close_splice_split": (("Sure, <|chan<chann", f"el|>nel>{SECRET}"), "Sure, "),
    # ruling F3: removing the inner block splices an open that the next close closes; block removal repeats until
    # stable before the truncate and before strays are deleted, so the text after that close is voiced
    "spliced_open_then_closed": (("<|chan<|channel>x<channel|>nel>A<channel|>B",), "B"),
    "spliced_open_then_closed_words": ((f"Hi <|chan<|channel>{SECRET}<channel|>nel>t<channel|> ok. ",), "Hi  ok. "),
    # F5: round 1 removes a block and deletes two strays, splicing ``<|channel>x<channel|>``, which round 2 removes
    "two_rounds": ((f"A {OPEN}thought {SECRET}{CLOSE}<|chan<channel|>nel>x<chan<channel|>nel|> B. ",), "A  B. "),
}  # fmt: skip


@pytest.mark.parametrize("case", list(SPLICES))
async def test_literal_spliced_marker(tmp_path, case):
    """Criterion 1's last bullet (adversarial; reviewer mutation 4, one pass with no fixpoint): a marker spliced by a
    deletion is handled as one written in full, so ``SECRET`` is never voiced; rulings F1 and F3 on reviewer A's
    findings (exact voiced and stored text)."""
    commits, expected = SPLICES[case]
    rig, snap = await run_turn(tmp_path, [answer(t) for t in commits])
    released = answer_commits(rig, "t1-gen")
    assert "".join(released) == expected, (
        f"T-0117 criterion 1 ({case}): released {released}, C(raw) is {expected!r}"
    )
    for c in rig.speaks():
        assert "SECRET" not in c.source and "SECRET" not in c.text, c
    assert snap["TtsSpeak.source"] == expected, (
        f"T-0117 criterion 1 ({case}): voiced {snap['TtsSpeak.source']!r}, C(raw) is {expected!r}"
    )
    assert snap["history"] == expected, snap["history"]
    assert_no_leak(snap, f"criterion 1 ({case})")


#: case -> (raw, C(raw) by hand under H-T117-3: the first ``<channel|>`` after an open closes it)
NESTED = {
    "repeated": (
        "A <|channel>x SECRET<channel|> B <|channel>y SECRET<channel|> C. ",
        "A  B  C. ",
    ),
    "nested": (
        "<|channel>a SECRET <|channel>b SECRET<channel|>c<channel|> d. ",
        "c d. ",
    ),
}


@pytest.mark.parametrize("case", list(NESTED))
async def test_repeated_and_nested_markers(tmp_path, case):
    """H-T117-3 (reviewer mutation 12, nesting-aware depth counting): repeated blocks are each removed with the
    bytes between them kept; a nested open is part of the first block, and the second close is stray, so ``c`` is
    answer. The hand-written expectation agrees with ``route(...).spoken_prefix`` and with ``template_content``
    (the pinned template's ``strip_thinking``, before its trim)."""
    raw, expected = NESTED[case]
    assert router.route(raw).spoken_prefix == expected, "premise: the router's rule"
    assert router.template_content(raw) == expected.strip(), "premise: the template's rule"
    rig, snap = await run_turn(tmp_path, [answer(raw)])
    assert "".join(answer_commits(rig, "t1-gen")) == expected, answer_commits(rig, "t1-gen")
    assert snap["TtsSpeak.source"] == expected and snap["history"] == expected, (
        f"T-0117 H-T117-3 ({case}): voiced and stored {expected!r}; got {snap['TtsSpeak.source']!r} / "
        f"{snap['history']!r}"
    )
    for c in rig.speaks():
        assert "SECRET" not in c.source and "SECRET" not in c.text, c


# =====================================================================================================
# Criterion 12: the word "thought" is speech (guard too wide; green at the base)
# =====================================================================================================


@pytest.mark.parametrize("text", ["I thought so. ", "I thought so, and I still do. "])
async def test_the_word_thought_is_voiced(tmp_path, text):
    """Criterion 12 (reviewer mutation 7): with no marker, ``thought`` is an ordinary word, voiced and stored
    verbatim, and no note is written. Holds at the base; it fails a cleaner that matches the word."""
    rig, snap = await run_turn(tmp_path, [answer(text)])
    assert snap["TtsSpeak.source"] == text and snap["history"] == text, (
        f"T-0117 criterion 12: {text!r} voiced and stored verbatim; got {snap['TtsSpeak.source']!r} / "
        f"{snap['history']!r}"
    )
    assert _norm(snap["astra.spoken"]) == _norm(text)
    assert notes(rig, NOTE_WHAT) == []


# =====================================================================================================
# Criterion 8: the note
# =====================================================================================================

#: case -> (policy, script, steps, the expected notes written out by hand, the thought texts)
NOTE_CASES: dict[str, tuple[Any, list[Commit], tuple[list[Commit], ...], list[dict[str, Any]], tuple[str, ...]]] = {
    "closed": (
        LOOP_OFF,
        [answer(RAW2)],
        (),
        [{"what": NOTE_WHAT, "request_id": "t1-gen", "step": 1, "chars": len(BLOCK2), "unclosed": False}],
        (f"thought {SECRET}",),
    ),
    "unclosed": (
        LOOP_OFF,
        [answer(RAW6)],
        (),
        [{"what": NOTE_WHAT, "request_id": "t1-gen", "step": 1, "chars": len(TAIL6), "unclosed": True}],
        (f"thought {SECRET}",),
    ),
    "reclosed": (
        LOOP_OFF,
        [answer(t) for t in RAW6B],
        (),
        [{"what": NOTE_WHAT, "request_id": "t1-gen", "step": 1, "chars": len("<|channel>xy<channel|>"),
          "unclosed": False}],
        ("xy",),
    ),
    # one note per stream: step 1 (prefix + call) and the step-2 report each carry a closed thought
    "two_streams": (
        loop_policy(),
        [answer(RAW3), tool(SEARCH_CALL)],
        ([answer(RAW5)],),
        [
            {"what": NOTE_WHAT, "request_id": "t1-gen", "step": 1, "chars": len(BLOCK2), "unclosed": False},
            {"what": NOTE_WHAT, "request_id": "t1-gen-s2", "step": 2, "chars": len(BLOCK2), "unclosed": False},
        ],
        (f"thought {SECRET}",),
    ),
    # ruling F1: the stray close is deleted (not counted: it is no thought text), which splices an open nothing
    # closes; the cut removes ``<|channel>`` + the secret
    "stray_close_splice": (
        LOOP_OFF,
        [answer(f"Sure, <|chan<channel|>nel>{SECRET}")],
        (),
        [{"what": NOTE_WHAT, "request_id": "t1-gen", "step": 1, "chars": len(OPEN) + len(SECRET),
          "unclosed": True}],
        (SECRET,),
    ),
    # ruling F3: two blocks removed in round 1 (the second spliced by the first's removal), nothing cut
    "spliced_open_then_closed": (
        LOOP_OFF,
        [answer("Hi <|chan<|channel>SECRET<channel|>nel>t<channel|> ok. ")],
        (),
        [{"what": NOTE_WHAT, "request_id": "t1-gen", "step": 1,
          "chars": len("<|channel>SECRET<channel|>") + len("<|channel>t<channel|>"), "unclosed": False}],
        ("SECRET",),
    ),
    # F5: ``chars`` counts across rounds: round 1 removes BLOCK2 (42), round 2 the spliced ``<|channel>x<channel|>``
    # (21); 63 in all, by hand
    "two_rounds": (
        LOOP_OFF,
        [answer(f"A {BLOCK2}<|chan<channel|>nel>x<chan<channel|>nel|> B. ")],
        (),
        [{"what": NOTE_WHAT, "request_id": "t1-gen", "step": 1, "chars": 42 + 21, "unclosed": False}],
        (f"thought {SECRET}",),
    ),
    # nothing removed: no note (reviewer mutation 10's second half); holds at the base
    "clean": (LOOP_OFF, [answer("Sure, here it is. ")], (), [], ()),
}  # fmt: skip


def _grams(text: str, n: int = 4) -> set[str]:
    return {text[i : i + n] for i in range(len(text) - n + 1)}


@pytest.mark.parametrize("case", NOTE_CASES)
async def test_note_counts_chars_never_text(tmp_path, case):
    """Criterion 8 (reviewer mutation 10): one session-log ``note`` per stream whose thought stripping removed
    anything, exactly ``{what, request_id, step, chars, unclosed}``; ``chars`` counts the removed characters,
    markers included (hand-computed); no other value holds a 4-character substring of the thought; written before
    that stream's ``llm.done``; none on a clean turn; ``llm.commit`` keeps the raw text."""
    policy, script, steps, expected, thoughts = NOTE_CASES[case]
    search = FakeSearch() if steps else None
    rig, _ = await run_turn(tmp_path, script, policy=policy, steps=steps, search=search)
    found = notes(rig, NOTE_WHAT)
    assert found == expected, f"T-0117 criterion 8 ({case}): notes {found}, expected {expected}"
    for note in found:
        assert set(note) == NOTE_KEYS, sorted(note)
        others = json.dumps({k: v for k, v in note.items() if k != "what"}, ensure_ascii=False)
        for thought in thoughts:
            hits = sorted(g for g in _grams(thought) if g in others)
            assert not hits, (
                f"T-0117 criterion 8 ({case}): the note holds thought text {hits}: {note}"
            )
    # order: each note precedes its own stream's llm.done
    kinds = [(r["kind"], r["data"]) for r in rig.log_records]
    for note in found:
        at = next(i for i, (k, d) in enumerate(kinds) if k == "note" and d == note)
        done = next(
            i
            for i, (k, d) in enumerate(kinds)
            if k == "llm.done" and d.get("request_id") == note["request_id"]
        )
        assert at < done, (
            f"T-0117 criterion 8 ({case}): the note is written before llm.done ({at} vs {done})"
        )
    raw = "".join(c.text for c in script if c.segment == "answer")
    logged = "".join(
        r["text"]
        for r in rig.kind("llm.commit")
        if r.get("segment") == "answer" and r.get("request_id") == "t1-gen"
    )
    assert logged == raw, "T-0117 criterion 8: llm.commit keeps the raw text verbatim"
