"""T-0121: a thought block inside a ``thought_open=False`` answer commit is never voiced or stored (tools off).

Session-level rows for criteria 1-4 and 6-8 of docs/tasks/T-0121.md, over the T-0092c rig with no kit
(``tests/unit/_tool_turn_support.tool_rig(kit=None)``: the session built as on the default path), the fake LLM, the
``ScriptedClock`` and the recording audio fake. The sinks are read with T-0117's ``snapshot`` and checked with its
``assert_voiced_exactly``/``assert_no_leak`` (imported from ``test_session_thought_in_answer``, not copied).

Every expected text is T(raw) written out by hand from the task's definition: repeat, until a round changes nothing,
(1) ``router.answer_channel`` until stable (a closed block removed with its words, the text from an open nothing
closes cut), (2) every stray ``<channel|>`` deleted. Nothing is inserted and no other literal is deleted. Nothing is
read from ``_ThoughtCleaner`` or the router to get an expected value.

``test_the_word_thought_is_voiced`` (criterion 8) and the no-note cases of the note row hold at the base (791dc54)
and carry no marker: they are the guard-too-wide rows (``.claude/rules/tests.md``). Every other row needs the fix
and is red until T-0121 lands (activated at the start of ``/implement T-0121``).

Readings where the task is silent (listed in the test-author's report):

- ``TtsSpeak.text`` and ``astra.spoken`` are the TTS spelling, compared whitespace-normalised (as T-0117 does);
  ``TtsSpeak.source``, the audio plays' sources, ``played_text`` and history are compared byte-exact;
- the HUD is checked for leaks too (``.claude/rules/conductor.md``: thought text never reaches the HUD panes), with
  the other sinks T-0117's ``snapshot`` reads, though criterion 1 does not list it;
- "the utterances actually played" are the audio fake's plays with a non-empty source (clips play with ``""``);
- after a barge-in the machine stores ``f"{played_text} [interrupted]"`` (docs/02 §4, ruling A6, the
  ``HistoryTurn`` docstring), so criterion 2's "stores the ``played_text`` the audio fake reports" is read as that;
- criterion 7's "after the held tail is submitted" is observed by wrapping the session's ``submit`` and the session
  log's ``record`` (the public seam every record goes through) into one ordered trace.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools off, n = 1 and
deterministic per row. No measurement.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.speech_gate import normalise_for_tts
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.turn import TurnState
from tests.unit._tool_turn_support import (
    ToolRig,
    answer,
    assert_history_clean,
    assert_never_voiced,
    tool_rig,
)
from tests.unit.test_session_thought_in_answer import (
    NOTE_KEYS,
    NOTE_WHAT,
    _grams,
    _norm,
    answer_commits,
    assert_no_leak,
    assert_voiced_exactly,
    snapshot,
)
from tests.unit.test_session_tool_loop import notes

SECRET = "SECRET-THOUGHT"
OPEN, CLOSE = "<|channel>", "<channel|>"
GEN = "t1-gen"

# --- the raw texts and T(raw), written out by hand ---------------------------------------------------------

BLOCK = f"{OPEN}thought {SECRET}{CLOSE}"
assert len(BLOCK) == 42  # 10 + len("thought ") 8 + 14 + 10, counted by hand
#: criterion 1: the block removed, nothing inserted: two spaces (H-T117-2)
RAW1 = f"Here {BLOCK} it is. "
CLEAN1 = "Here  it is. "

#: criterion 4: the open mid-sentence, nothing closes it
TAIL4 = f"{OPEN}thought {SECRET}"
RAW4 = f"Okay {TAIL4}"
CLEAN4 = "Okay "
#: criterion 4, second half: a later commit closes it
RAW4B = ("Okay <|channel>x", "y<channel|> fine. ")
CLEAN4B = "Okay  fine. "

#: the T-0117 differential's moved turn 3: ``Wait <|channel> there. `` cut at the open
RAW_WAIT = ("Wait <|chan", "nel> there. ")
CLEAN_WAIT = "Wait "

#: criterion 6
RAW6 = "Done <channel|>here. "
CLEAN6 = "Done here. "


def open_answer(text: str) -> Commit:
    return Commit(request_id="?", canvas_idx=0, segment="answer", text=text, thought_open=True)


# --- running a tools-off turn ------------------------------------------------------------------------------


async def run_off(
    tmp_path: Path, scripts: Sequence[Sequence[Commit]], *, trace: list[Any] | None = None
) -> tuple[ToolRig, dict[str, Any]]:
    """One tools-off session, one turn per script; the snapshot is taken after the last turn.

    ``trace`` (if given) receives ``("submit", event)`` and ``("log", kind, data)`` in the order they happen."""
    async with tool_rig(tmp_path, kit=None) as rig:
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
        for i, script in enumerate(scripts):
            rig.llm.script = list(script)
            await rig.ask("what does it say")
            await rig.until(
                lambda n=i + 1: rig.state.state is TurnState.IDLE and len(rig.lines()) == n,
                f"T-0121: turn {i + 1} never ended",
            )
            await rig.advance(300)
        snap = snapshot(rig)
    return rig, snap


def released_ok(rig: ToolRig, expected: str, what: str, request_id: str = GEN) -> list[str]:
    """The answer commits the machine received join to T(raw); none holds the secret."""
    released = answer_commits(rig, request_id)
    assert not [t for t in released if "SECRET" in t], (
        f"T-0121 {what}: an answer commit the machine received holds the thought: {released}"
    )
    assert "".join(released) == expected, (
        f"T-0121 {what}: the machine's answer commits join to T(raw) {expected!r}; got {released}"
    )
    return released


def no_secret_spoken(rig: ToolRig, what: str) -> None:
    for c in rig.speaks():
        assert "SECRET" not in c.source and "SECRET" not in c.text, f"T-0121 {what}: {c}"


# =====================================================================================================
# Criterion 1: a closed block in one commit
# =====================================================================================================


async def test_closed_thought_not_voiced_or_stored(tmp_path):
    """Criterion 1: tools off, ``answer("Here <|channel>thought SECRET-THOUGHT<channel|> it is. ")``. Every
    ``TtsSpeak.source``, the plays' sources, ``played_text`` and history are exactly ``Here  it is. `` (the positive
    string, so dropping the answer fails); the thought and the markers are in no sink, ``turns.jsonl`` and the
    memory episodes included. Mutations: clean ``source`` at the gate, cut at the first open, collapse whitespace."""
    rig, snap = await run_off(tmp_path, [[answer(RAW1)]])
    assert_voiced_exactly(snap, CLEAN1, "T-0121 criterion 1")
    assert_no_leak(snap, "T-0121 criterion 1")
    assert_never_voiced(rig, what=" (T-0121 criterion 1)")
    assert_history_clean(rig)
    released_ok(rig, CLEAN1, "criterion 1")
    drained = [e.played_text for e in rig.machine.events(E.PlaybackDrained)]
    assert drained and drained[-1] == CLEAN1, (
        f"T-0121 criterion 1: played_text is exactly {CLEAN1!r}; got {drained}"
    )
    # criterion 7, last item: llm.commit keeps the raw text
    assert (
        "".join(r["text"] for r in rig.kind("llm.commit") if r.get("segment") == "answer") == RAW1
    )


# =====================================================================================================
# Criterion 2: history equals what was played
# =====================================================================================================

#: case -> (commits, T(raw) by hand); criteria 1, 3, 4 and 6
PLAYED = {
    "closed": ((RAW1,), CLEAN1),
    "split": (("Here <|chan", f"nel>thought {SECRET}<chan", "nel|> it is. "), CLEAN1),
    "unclosed": ((RAW4,), CLEAN4),
    "reclosed": (RAW4B, CLEAN4B),
    "stray_close": ((RAW6,), CLEAN6),
}


@pytest.mark.parametrize("case", list(PLAYED))
async def test_history_equals_what_was_played(tmp_path, case):
    """Criterion 2: for every utterance played, ``normalise_for_tts(source) == text`` (the gate's public renderer);
    history equals the join of the played utterances' sources, in order, and that join is T(raw) and thought-free."""
    commits, expected = PLAYED[case]
    rig, snap = await run_off(tmp_path, [[answer(t) for t in commits]])
    for c in rig.speaks():
        if c.source:
            assert normalise_for_tts(c.source) == c.text, (
                f"T-0121 criterion 2 ({case}): text is the gate's rendering of source: {c}"
            )
    played = [s for _, s in rig.audio.plays if s]
    assert snap["history"] == "".join(played), (
        f"T-0121 criterion 2 ({case}): history {snap['history']!r} is the join of what was played {played}"
    )
    assert "".join(played) == expected, (
        f"T-0121 criterion 2 ({case}): what was played joins to T(raw) {expected!r}; got {played}"
    )
    assert_no_leak(snap, f"T-0121 criterion 2 ({case})", extra={"played": json.dumps(played)})


BARGED = f"Here {BLOCK} it is. This answer is long enough to still be playing at the cut. "
BARGED_FIRST = "Here  it is. "
BARGED_CLEAN = BARGED_FIRST + "This answer is long enough to still be playing at the cut. "


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(BARGED, id="block"),
        # the same answer with no marker: green at the base, it proves the cut lands mid-way through the second
        # sentence on this rig's timing, so the ``block`` case's premise is not what reddens it
        pytest.param(BARGED_CLEAN, id="no_block"),
    ],
)
async def test_history_equals_what_was_played_barge_in(tmp_path, raw):
    """Criterion 2, the barge-in row: a takeover cuts the answer mid-way, after the removed block. History stores
    the ``played_text`` the audio fake reports (plus the ``[interrupted]`` marker), which holds no thought byte, is
    longer than the first sentence (the cut is after the block) and is a prefix of T(raw)."""
    async with tool_rig(tmp_path, kit=None, script=[answer(raw)]) as rig:
        await rig.ask("tell me more")
        await rig.until(
            lambda: (
                (rig.state.state is TurnState.IDLE and bool(rig.lines()))
                or (
                    rig.state.state is TurnState.SPEAKING
                    and any(s and "long enough" in s for _, s in rig.audio.plays)
                )
            ),
            "the turn neither ended nor reached the second sentence",
            limit_ms=10_000,
        )
        played = [s for _, s in rig.audio.plays if s]
        assert rig.state.state is TurnState.SPEAKING, (
            f"T-0121 criteria 1-2 (barge-in): the text after the removed block is voiced, so the second sentence "
            f"is still playing; the turn ended having played {played}, storing "
            f"{[h.assistant_spoken for h in rig.state.history]}"
        )
        await rig.advance(1200, step=100)
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.until(
            lambda: bool(rig.machine.events(E.PlaybackStopped)),
            "the takeover never stopped playback",
        )
        await rig.until(lambda: bool(rig.state.history), "the cut turn was never stored")
        stopped = rig.machine.events(E.PlaybackStopped)[-1].played_text
        stored = rig.state.history[-1].assistant_spoken
    for leak in (SECRET, "thought SECRET", OPEN, CLOSE):
        assert leak not in stopped and leak not in stored, (
            f"T-0121 criterion 2 (barge-in): {leak!r} in played_text {stopped!r} / history {stored!r}"
        )
    assert stored == f"{stopped} [interrupted]", (
        f"T-0121 criterion 2 (barge-in): history stores the played_text the audio fake reports; "
        f"played {stopped!r}, stored {stored!r}"
    )
    assert BARGED_CLEAN.startswith(stopped) and len(stopped) > len(BARGED_FIRST), (
        f"T-0121 criterion 2 (barge-in): played_text {stopped!r} is a prefix of T(raw) cut after the removed "
        f"block (premise: the cut is mid-way through the second sentence)"
    )


# =====================================================================================================
# Criterion 3: markers split across commits (adversarial: stripping per commit)
# =====================================================================================================

SPLITS = {
    # the most likely failure: stripping per commit instead of over the concatenation
    "marker_split": (answer("Here <|chan"), answer(f"nel>thought {SECRET}<chan"), answer("nel|> it is. ")),
    # the open in commit 1, the close two commits later
    "close_two_later": (answer("Here <|channel>thought "), answer(f"{SECRET} "), answer("<channel|> it is. ")),
    # I15: a thought_open=True commit in between neither flushes nor resets the hold
    "across_an_open_commit": (answer("Here <|chan"), open_answer("a draft"),
                              answer(f"nel>thought {SECRET}<channel|> it is. ")),
}  # fmt: skip


@pytest.mark.parametrize("case", list(SPLITS))
async def test_marker_split_across_commits(tmp_path, case):
    """Criterion 3 (adversarial; mutations: strip per commit, hold no tail, reset the hold on a ``thought_open=True``
    commit, clean only in ``flush``): criterion 1's result, and no ``LlmCommit`` the machine receives holds
    ``SECRET`` or a marker prefix that later completes. T(raw) holds no ``<`` at all, so any ``<`` in a released
    answer commit is such a prefix (each release is a prefix of T(raw), criterion 5)."""
    rig, snap = await run_off(tmp_path, [list(SPLITS[case])])
    released = released_ok(rig, CLEAN1, f"criterion 3 ({case})")
    assert not [t for t in released if "<" in t], (
        f"T-0121 criterion 3 ({case}): a released answer commit holds a marker prefix: {released}"
    )
    # N4 (implementation notes: a commit that cleans to nothing because its text was withheld is not submitted).
    # By hand from the hold rule: commit 1 releases ``Here `` (``<|chan`` / ``<|channel>thought `` withheld), the
    # middle commit releases nothing (all thought, or the thought_open=True commit, which bypasses the cleaner),
    # commit 3 releases `` it is. ``. So the machine receives exactly two voiced answer commits, no ``""``.
    assert released == ["Here ", " it is. "], (
        f"T-0121 criterion 3 ({case}): the machine's voiced answer commits are exactly ['Here ', ' it is. '] "
        f"(a fully withheld commit is not submitted as ''); got {released}"
    )
    assert_voiced_exactly(snap, CLEAN1, f"T-0121 criterion 3 ({case})")
    assert_no_leak(snap, f"T-0121 criterion 3 ({case})")
    if case == "across_an_open_commit":
        opens = [
            (e.text, e.thought_open)
            for e in rig.machine.events(E.LlmCommit)
            if e.request_id == GEN and e.thought_open
        ]
        assert opens == [("a draft", True)], (
            f"T-0121 criterion 10: the thought_open=True commit reaches the machine as sent; got {opens}"
        )


# =====================================================================================================
# Criterion 4: an open nothing closes (adversarial), and one a later commit closes (guard too wide)
# =====================================================================================================

UNCLOSED = {
    "one_commit": (RAW4,),
    "open_split": ("Okay <|chan", f"nel>thought {SECRET}"),
}


@pytest.mark.parametrize("case", list(UNCLOSED))
async def test_unclosed_mid_sentence_is_dropped(tmp_path, case):
    """Criterion 4 (adversarial; mutation: closed blocks only, keeping the unclosed tail): ``Okay <|channel>thought
    SECRET-THOUGHT``, ``thought_open=False``, the open mid-sentence (no terminator before it, so a gate that only
    voices whole sentences cannot hide the bug). Voiced and stored exactly ``Okay ``; nothing after the open is
    submitted to the machine before or at ``llm.done``."""
    rig, snap = await run_off(tmp_path, [[answer(t) for t in UNCLOSED[case]]])
    released_ok(rig, CLEAN4, f"criterion 4 ({case})")
    no_secret_spoken(rig, f"criterion 4 ({case})")
    assert_voiced_exactly(snap, CLEAN4, f"T-0121 criterion 4 ({case})")
    assert_no_leak(snap, f"T-0121 criterion 4 ({case})")


async def test_unclosed_then_closed_voices_the_rest(tmp_path):
    """Criterion 4, second half (guard too wide; mutation: cut at the first open even when closed): ``Okay
    <|channel>x`` / ``y<channel|> fine. `` voices and stores ``Okay  fine. ``."""
    rig, snap = await run_off(tmp_path, [[answer(t) for t in RAW4B]])
    released_ok(rig, CLEAN4B, "criterion 4 (reclosed)")
    assert_voiced_exactly(snap, CLEAN4B, "T-0121 criterion 4 (reclosed)")
    assert_no_leak(snap, "T-0121 criterion 4 (reclosed)")


async def test_t0117_tools_off_wait_row(tmp_path):
    """The named spec change to ``test_t0117_differential``: its tools-off turn 3, ``Wait <|chan`` / ``nel> there.
    `` (a split open nothing closes), left the c7e04f0 comparison because c7e04f0 stores the thought marker. Its
    expected behaviour is T(raw) ``Wait ``: voiced and stored, nothing after the spliced open."""
    rig, snap = await run_off(tmp_path, [[answer(t) for t in RAW_WAIT]])
    released_ok(rig, CLEAN_WAIT, "the moved T-0117 turn 3")
    assert_voiced_exactly(snap, CLEAN_WAIT, "T-0121 (the moved T-0117 turn 3)")
    assert_no_leak(snap, "T-0121 (the moved T-0117 turn 3)")
    assert "there" not in snap["history"] and "there" not in snap["TtsSpeak.source"], snap[
        "history"
    ]


#: case -> (commits, T(raw) by hand)
SPLICES = {
    # round 1 deletes the stray close, splicing ``<|chan`` + ``nel>`` into an open nothing closes; round 2 cuts there
    "stray_close_splice": ((f"Here <|chan<channel|>nel>{SECRET}",), "Here "),
    "stray_close_splice_split": (("Here <|chan<chann", f"el|>nel>{SECRET}"), "Here "),
    # removing ``<|channel>x<channel|>`` splices an open the next close closes (F3: answer_channel until stable)
    "block_removal_splice": (
        ("Hi <|chan<|channel>x<channel|>nel>SECRET<channel|> ok. ",),
        "Hi  ok. ",
    ),
}

#: N9: round 1 deletes *both* stray closes (``<|channel>AB``), round 2 cuts at the spliced open: T(raw) is ``""``.
#: Deleting one close per round would splice ``<|channel>A<channel|>B``, remove the block and voice ``B``.
ALL_STRAYS = {
    "one_commit": ("<|chan<channel|>nel>A<channel|>B",),
    "split": ("<|chan<chan", "nel|>nel>A<channel|>B"),
}


@pytest.mark.parametrize("case", list(ALL_STRAYS))
async def test_every_stray_close_in_a_round_is_deleted(tmp_path, case):
    """Adversarial (N9; mutation: delete one stray ``<channel|>`` per round): ``<|chan<channel|>nel>A<channel|>B``
    is all thought (T(raw) ``""``), so nothing reaches the machine as voiced text, nothing is voiced or played, and
    nothing is stored. The note counts 12 (``<|channel>AB``), unclosed."""
    rig, snap = await run_off(tmp_path, [[answer(t) for t in ALL_STRAYS[case]]])
    assert answer_commits(rig, GEN) == [], (
        f"T-0121 N9 ({case}): no voiced answer commit reaches the machine; got {answer_commits(rig, GEN)}"
    )
    voiced = {
        k: snap[k] for k in ("TtsSpeak.source", "TtsSpeak.text", "astra.spoken", "audio plays")
    }
    assert all(v == "" for v in voiced.values()), (
        f"T-0121 N9 ({case}): nothing is voiced; got {voiced}"
    )
    # nothing stored: no history entry, or one whose spoken text is empty (the task does not say which)
    assert snap["history"] in (None, ""), (
        f"T-0121 N9 ({case}): nothing is stored; got {snap['history']!r}"
    )
    assert notes(rig, NOTE_WHAT) == [_note(GEN, 12, True)], notes(rig, NOTE_WHAT)
    assert_no_leak(snap, f"T-0121 N9 ({case})")


@pytest.mark.parametrize("case", list(SPLICES))
async def test_spliced_marker(tmp_path, case):
    """Adversarial (T-0117 F1/F3; mutation: one pass with no fixpoint): a marker a deletion splices is handled like
    one written in full. ``Here <|chan<channel|>nel>SECRET-THOUGHT`` stores ``Here ``; ``Hi <|chan<|channel>x
    <channel|>nel>SECRET<channel|> ok. `` stores ``Hi  ok. ``."""
    commits, expected = SPLICES[case]
    rig, snap = await run_off(tmp_path, [[answer(t) for t in commits]])
    released_ok(rig, expected, f"splice ({case})")
    no_secret_spoken(rig, f"splice ({case})")
    assert snap["TtsSpeak.source"] == expected and snap["history"] == expected, (
        f"T-0121 splice ({case}): voiced and stored {expected!r}; got {snap['TtsSpeak.source']!r} / "
        f"{snap['history']!r}"
    )
    assert_no_leak(snap, f"T-0121 splice ({case})")


# =====================================================================================================
# Criterion 6: a stray close
# =====================================================================================================

STRAYS = {"one_commit": (RAW6,), "split": ("Done <chan", "nel|>here. ")}


@pytest.mark.parametrize("case", list(STRAYS))
async def test_stray_close_removed_without_note(tmp_path, case):
    """Criterion 6 (H-T121-3; mutation: count stray closes in the note): ``Done <channel|>here. `` voices and
    stores ``Done here. ``, no marker reaches history, and no note is written (nothing was thought text)."""
    rig, snap = await run_off(tmp_path, [[answer(t) for t in STRAYS[case]]])
    released_ok(rig, CLEAN6, f"criterion 6 ({case})")
    assert_voiced_exactly(snap, CLEAN6, f"T-0121 criterion 6 ({case})")
    assert_no_leak(snap, f"T-0121 criterion 6 ({case})")
    assert notes(rig, NOTE_WHAT) == [], notes(rig, NOTE_WHAT)


# =====================================================================================================
# Criterion 8: the word "thought" is speech (guard too wide; green at the base)
# =====================================================================================================


@pytest.mark.parametrize(
    "text", ["I thought so, and I still do. ", "thought: a thought, thoughtfully. "]
)
async def test_the_word_thought_is_voiced(tmp_path, text):
    """Criterion 8 (mutation: match the word "thought"): with no marker, ``thought`` is an ordinary word, voiced and
    stored verbatim; no note. Holds at the base."""
    rig, snap = await run_off(tmp_path, [[answer(text)]])
    assert snap["TtsSpeak.source"] == text and snap["history"] == text, (
        f"T-0121 criterion 8: {text!r} voiced and stored verbatim; got {snap['TtsSpeak.source']!r} / "
        f"{snap['history']!r}"
    )
    assert _norm(snap["astra.spoken"]) == _norm(text)
    assert "".join(answer_commits(rig, GEN)) == text
    assert notes(rig, NOTE_WHAT) == []


# =====================================================================================================
# Criterion 7: the note
# =====================================================================================================


def _note(request_id: str, chars: int, unclosed: bool) -> dict[str, Any]:
    return {
        "what": NOTE_WHAT,
        "request_id": request_id,
        "step": 1,
        "chars": chars,
        "unclosed": unclosed,
    }


#: case -> (one script per turn, the expected notes by hand, the thought texts). ``chars`` counts what
#: ``answer_channel`` removed, markers included; a stray close is not counted.
NOTE_CASES: dict[str, tuple[list[list[str]], list[dict[str, Any]], tuple[str, ...]]] = {
    "closed": ([[RAW1]], [_note(GEN, 42, False)], (f"thought {SECRET}",)),
    "unclosed": ([[RAW4]], [_note(GEN, len(TAIL4), True)], (f"thought {SECRET}",)),
    "reclosed": ([list(RAW4B)], [_note(GEN, len("<|channel>xy<channel|>"), False)], ("xy",)),
    # the moved T-0117 turn: ``<|channel> there. `` cut (18 characters)
    "wait": ([list(RAW_WAIT)], [_note(GEN, 18, True)], (" there",)),
    # the stray close (10) is not counted; the cut removes ``<|channel>`` + the secret: 10 + 14
    "stray_close_splice": ([[f"Here <|chan<channel|>nel>{SECRET}"]], [_note(GEN, 24, True)], (SECRET,)),
    # two blocks removed by answer_channel: ``<|channel>x<channel|>`` (21) and ``<|channel>SECRET<channel|>`` (26)
    "block_removal_splice": ([["Hi <|chan<|channel>x<channel|>nel>SECRET<channel|> ok. "]],
                             [_note(GEN, 47, False)], ("SECRET",)),
    # a held tail ``<|cha`` released at the end of the stream; the note comes after it
    "held_tail": ([["A <|channel>x SECRET<channel|> B <|cha"]], [_note(GEN, len("<|channel>x SECRET<channel|>"),
                                                                       False)], ("x SECRET",)),
    # one note per stream: two turns, each with a block
    "two_turns": ([[RAW1], [RAW1]], [_note(GEN, 42, False), _note("t2-gen", 42, False)], (f"thought {SECRET}",)),
    # nothing removed by answer_channel: no note (green at the base)
    "stray_only": ([[RAW6]], [], ()),
    "clean": ([["Sure, here it is. "]], [], ()),
}  # fmt: skip

GREEN_AT_BASE = frozenset({"stray_only", "clean"})


@pytest.mark.parametrize(
    "case",
    [pytest.param(c) for c in NOTE_CASES],
)
async def test_note_counts_chars_never_text(tmp_path, case):
    """Criterion 7 (H-T121-4; mutations: text in the note, a note when ``chars == 0``, counting stray closes): one
    session-log ``note`` per stream whose thought stripping removed anything, exactly ``{what, request_id, step: 1,
    chars, unclosed}`` with hand-computed ``chars``; no other value holds a 4-character substring of the thought;
    written after the held tail is submitted and before ``llm.done``; ``llm.commit`` keeps the raw text."""
    scripts, expected, thoughts = NOTE_CASES[case]
    trace: list[Any] = []
    rig, _ = await run_off(tmp_path, [[answer(t) for t in s] for s in scripts], trace=trace)
    found = notes(rig, NOTE_WHAT)
    assert found == expected, f"T-0121 criterion 7 ({case}): notes {found}, expected {expected}"
    for note in found:
        assert set(note) == NOTE_KEYS, sorted(note)
        others = json.dumps({k: v for k, v in note.items() if k != "what"}, ensure_ascii=False)
        for thought in thoughts:
            hits = sorted(g for g in _grams(thought) if g in others)
            assert not hits, (
                f"T-0121 criterion 7 ({case}): the note holds thought text {hits}: {note}"
            )
        rid = note["request_id"]
        at = next(
            i for i, t in enumerate(trace) if t[0] == "log" and t[1] == "note" and t[2] == note
        )
        done = next(
            i
            for i, t in enumerate(trace)
            if t[0] == "log" and t[1] == "llm.done" and t[2].get("request_id") == rid
        )
        commits = [
            i
            for i, t in enumerate(trace)
            if t[0] == "submit"
            and isinstance(t[1], E.LlmCommit)
            and t[1].request_id == rid
            and t[1].segment == "answer"
        ]
        assert commits and commits[-1] < at < done, (
            f"T-0121 criterion 7 ({case}): the note comes after the stream's last answer commit (the held tail) "
            f"and before llm.done; trace positions: commits {commits}, note {at}, llm.done {done}"
        )
    for k, script in enumerate(scripts, start=1):
        logged = "".join(
            r["text"]
            for r in rig.kind("llm.commit")
            if r.get("segment") == "answer" and r.get("request_id") == f"t{k}-gen"
        )
        assert logged == "".join(script), (
            "T-0121 criterion 7: llm.commit keeps the raw text verbatim"
        )
    if case == "held_tail":
        assert answer_commits(rig, GEN)[-1:] == ["<|cha"], (
            f"premise: the held tail ``<|cha`` is the stream's last answer commit; got {answer_commits(rig, GEN)}"
        )
