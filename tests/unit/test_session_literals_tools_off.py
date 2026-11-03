"""T-0124: special literals the speech gate trims or drops are never stored, nor published as played (tools off).

Session-level rows for criteria 1-7 and 9 of docs/tasks/T-0124.md, over the T-0092c rig with no kit
(``tests/unit/_tool_turn_support.tool_rig(kit=None)``, the default path), the fake LLM, the ``ScriptedClock`` and the
recording audio fake. The turn driver is T-0121's ``run_off`` (with its ordered ``submit``/``record`` trace) and the
sinks are read with T-0117's ``snapshot`` (HUD, ``turns.jsonl``, memory episodes included), imported, not copied.

Every expected text is U(raw) written out by hand from the task's rule: T(raw) (T-0121: closed thought blocks
removed, the text from an open nothing closes cut, stray ``<channel|>`` deleted), then L: until a round changes
nothing, cut at the earliest gate trim literal (``<end_of_turn>``, ``<eos>``, ``<|endoftext|>``, ``<|think|>``,
``<turn|>``, ``<|tool_response>``, ``<|tool_call>``, ``<|channel>``) and delete every other special literal. Nothing
is read from ``_LiteralCleaner`` (absent at the base) to get an expected value.

Pending rows (``pending(task="T-0124")``) are red at the base 8b3f6d0; unmarked rows hold there: criterion 9 (tools
on, ``_AnswerCleaner``), criterion 6's thought-first case (T-0121 already handles it; it is the stage-order guard),
the no-literal controls and the no-note cases.

Readings where the task is silent (listed in the test-author's report):

- ``astra.spoken`` carries only ``text`` (the TTS spelling, ``contracts/protocol.AstraSpoken``), so criterion 1's
  "``astra.spoken`` sources" is read as its texts, compared whitespace-normalised like ``TtsSpeak.text``;
  ``astra.spoken.played`` is the join of its utterances' ``played_text``, compared byte-exact;
- ``played_text`` is the last ``PlaybackDrained`` of the turn (as T-0121 reads it);
- "no ``now``" is a whole-word match (``\\bnow\\b``), so a JSON key like ``unknown`` cannot trip it;
- after a barge-in the machine stores ``f"{played_text} [interrupted]"`` (docs/02 §4, ruling A6), as T-0121 reads
  criterion 2's barge-in row;
- "no literal" is none of ``SPECIAL_TOKENS`` and ``speech_gate._TRIM_TOKENS``.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools off unless a row says
otherwise (then ``dry_run``, loop off), n = 1 and deterministic per row. No measurement.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import speech_gate
from local_astra.conductor.speech_gate import normalise_for_tts
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.turn import TurnState
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS
from tests.unit._tool_turn_support import (
    ToolRig,
    answer,
    assert_history_clean,
    assert_never_voiced,
    tool_rig,
)
from tests.unit.test_session_thought_in_answer import (
    NOTE_WHAT as THOUGHT_NOTE_WHAT,
)
from tests.unit.test_session_thought_in_answer import (
    _norm,
    answer_commits,
    snapshot,
)
from tests.unit.test_session_thought_tools_off import open_answer, run_off
from tests.unit.test_session_tool_loop import notes

SECRET = "SECRET-THOUGHT"
GEN = "t1-gen"
NOTE_WHAT = "special literals in an answer commit were not voiced or stored"
NOTE_KEYS = frozenset({"what", "request_id", "step", "chars", "cut"})
#: "No literal" (Tests to write): none of SPECIAL_TOKENS and the gate's trim table
LITERALS = tuple(dict.fromkeys((*SPECIAL_TOKENS, *speech_gate._TRIM_TOKENS)))
#: criterion 1's seven trim literals (``<|channel>`` is T-0121's, and is a thought open there)
TRIMS = (
    "<eos>",
    "<turn|>",
    "<|tool_call>",
    "<|tool_response>",
    "<|think|>",
    "<end_of_turn>",
    "<|endoftext|>",
)
NOW = re.compile(r"\bnow\b")

# --- the raw texts and U(raw), written out by hand ---------------------------------------------------------

#: criterion 2: no ``<|channel>`` in raw, so T(raw) only deletes the stray close; L cuts at ``<eos>``
RAW2 = f"Sure, <|chan<eos>nel>thought {SECRET}<channel|>ok. "
CLEAN2 = "Sure, <|chan"
#: len(T(raw)) - len(U(raw)): T(raw) = "Sure, <|chan<eos>nel>thought SECRET-THOUGHTok. " (6+6+5+4+8+14+4 = 47); 47 - 12
CHARS2 = 35
#: criterion 6: T(raw) removes the block, ``<eos>`` with it; nothing is left for L
RAW6 = "Here <|channel>a<eos>b<channel|> ok. "
CLEAN6 = "Here  ok. "
#: criterion 6, spliced: round 1 deletes ``<pad>``, splicing ``<|tool_call>``; round 2 cuts there
RAW6S = "Press <|tool<pad>_call> now. "
CLEAN6S = "Press "


def press(lit: str) -> str:
    return f"Press {lit} now. "


# --- reading the sinks -------------------------------------------------------------------------------------


def sinks(rig: ToolRig, snap: dict[str, Any]) -> dict[str, Any]:
    """Criterion 1's sinks: the exact ones (sources, plays, played_text, history, astra.spoken.played) and the TTS
    spellings (TtsSpeak.text, astra.spoken), plus turns.jsonl and the memory episodes for the leak checks."""
    drained = [e.played_text for e in rig.machine.events(E.PlaybackDrained)]
    played_msgs = [
        m for m in rig.published.messages if getattr(m, "type", None) == "astra.spoken.played"
    ]
    return {
        "exact": {
            "TtsSpeak.source": snap["TtsSpeak.source"],
            "audio plays": snap["audio plays"],
            "played_text": drained[-1] if drained else None,
            "history": snap["history"],
            "astra.spoken.played": "".join(
                u.played_text for m in played_msgs for u in m.utterances
            ),
        },
        "voice": {"TtsSpeak.text": snap["TtsSpeak.text"], "astra.spoken": snap["astra.spoken"]},
        "other": {
            "history (all)": snap["history (all)"],
            "turns.jsonl": snap["turns.jsonl"],
            "memory episodes": snap["memory episodes"],
        },
    }


def assert_stored_exactly(rig: ToolRig, snap: dict[str, Any], expected: str, what: str) -> None:
    got = sinks(rig, snap)
    bad = {k: v for k, v in got["exact"].items() if v != expected}
    assert not bad, f"T-0124 {what}: stored/played text must be exactly {expected!r}; got {bad}"
    bad = {k: v for k, v in got["voice"].items() if v != _norm(expected)}
    assert not bad, (
        f"T-0124 {what}: the TTS/HUD spelling must be {_norm(expected)!r} (whitespace-normalised); got {bad}"
    )


def assert_no_literal(
    rig: ToolRig, snap: dict[str, Any], what: str, *, words: tuple[str, ...] = ()
) -> None:
    """No literal (and none of ``words``, whole-word) in any criterion-1 sink, turns.jsonl or memory episodes."""
    got = sinks(rig, snap)
    flat = {k: v for group in got.values() for k, v in group.items() if isinstance(v, str)}
    bad: dict[str, list[str]] = {}
    for k, v in flat.items():
        hits = [lit for lit in LITERALS if lit in v]
        hits += [w for w in words if re.search(rf"\b{re.escape(w)}\b", v)]
        if hits:
            bad[k] = hits
    assert not bad, f"T-0124 {what}: a literal or post-stop text reached {bad}"


def released(rig: ToolRig, expected: str, what: str, request_id: str = GEN) -> list[str]:
    """The voiced answer commits the machine received join to U(raw), and none holds a literal."""
    texts = answer_commits(rig, request_id)
    assert not [t for t in texts if any(lit in t for lit in LITERALS)], (
        f"T-0124 {what}: an answer commit the machine received holds a literal: {texts}"
    )
    assert "".join(texts) == expected, (
        f"T-0124 {what}: the machine's answer commits join to U(raw) {expected!r}; got {texts}"
    )
    return texts


def voice(rig: ToolRig) -> str:
    return _norm(" ".join(c.text for c in rig.speaks()))


# =====================================================================================================
# Criterion 1: each literal, tools off
# =====================================================================================================


@pytest.mark.parametrize("lit", TRIMS)
async def test_each_trim_literal_is_neither_stored_nor_published(tmp_path, lit):
    """Criterion 1 (mutations: delete trim literals instead of cutting; build played_text or history from text;
    clean ``source`` at the gate boundary): one commit ``Press {lit} now. `` stores, plays and publishes exactly
    ``Press `` in every sink; no literal and no ``now`` reaches any of them, turns.jsonl or the memory episodes; the
    voice is ``Press``, as at the base."""
    rig, snap = await run_off(tmp_path, [[answer(press(lit))]])
    assert voice(rig) == "Press", f"premise (the voice is unchanged): {voice(rig)!r}"
    assert_stored_exactly(rig, snap, "Press ", f"criterion 1 ({lit})")
    assert_no_literal(rig, snap, f"criterion 1 ({lit})", words=("now",))
    assert_never_voiced(rig, what=f" (T-0124 criterion 1, {lit})")
    assert_history_clean(rig)
    assert released(rig, "Press ", f"criterion 1 ({lit})") == ["Press "]


#: raw -> (U(raw), voice)
DROPS = {
    "pad": ("Hi <pad>there. ", "Hi there. ", "Hi there."),
    "quote": ('Say <|"|>yes<|"|>. ', "Say yes. ", "Say yes."),
}


@pytest.mark.parametrize("case", list(DROPS))
async def test_drop_literals_are_deleted_from_history(tmp_path, case):
    """Criterion 1, drop rows (mutation: cut at a drop literal): ``Hi <pad>there. `` stores ``Hi there. `` and voices
    ``Hi there.``; ``Say <|"|>yes<|"|>. `` stores ``Say yes. ``. Nothing is inserted where a literal was."""
    raw, clean, spoken = DROPS[case]
    rig, snap = await run_off(tmp_path, [[answer(raw)]])
    assert voice(rig) == spoken, f"premise (the voice is unchanged): {voice(rig)!r}"
    assert_stored_exactly(rig, snap, clean, f"criterion 1 ({case})")
    assert_no_literal(rig, snap, f"criterion 1 ({case})")
    released(rig, clean, f"criterion 1 ({case})")


# =====================================================================================================
# Criterion 2: the ``<eos>``-spliced thought (adversarial: history and HUD, not only the voice)
# =====================================================================================================


async def test_eos_spliced_thought_never_reaches_history_or_hud(tmp_path):
    """Criterion 2 (H-T124-3; the task's worst case): ``Sure, <|chan<eos>nel>thought SECRET-THOUGHT<channel|>ok. ``.
    The gate already cuts the *voice* at ``<eos>``; the bug is that history, the plays, ``astra.spoken.played``, the
    HUD, turns.jsonl and the memory episodes hold the thought. They hold neither ``SECRET-THOUGHT`` nor ``thought
    SECRET``; history is exactly ``Sure, <|chan``; the voice is base's; the T-0124 note counts 35, cut."""
    rig, snap = await run_off(tmp_path, [[answer(RAW2)]])
    assert voice(rig) == CLEAN2, f"premise (the voice is unchanged): {voice(rig)!r}"
    got = sinks(rig, snap)
    flat = {k: v for g in got.values() for k, v in g.items() if isinstance(v, str)}
    flat["HUD"] = snap["HUD"]
    bad = {k: [s for s in (SECRET, "thought SECRET") if s in v] for k, v in flat.items()}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"T-0124 criterion 2: the spliced thought reached {bad}"
    assert rig.state.history[-1].assistant_spoken == CLEAN2, rig.state.history[-1].assistant_spoken
    assert_stored_exactly(rig, snap, CLEAN2, "criterion 2")
    assert_no_literal(rig, snap, "criterion 2")
    released(rig, CLEAN2, "criterion 2")
    assert notes(rig, NOTE_WHAT) == [_note(GEN, CHARS2, True)], notes(rig, NOTE_WHAT)


# =====================================================================================================
# Criterion 3: history equals what was played
# =====================================================================================================

#: case -> (commits, U(raw) by hand); criteria 1, 2, 4, 5 and 6
PLAYED: dict[str, tuple[tuple[Commit, ...], str]] = {
    **{f"trim {lit}": ((answer(press(lit)),), "Press ") for lit in TRIMS},
    "drop pad": ((answer("Hi <pad>there. "),), "Hi there. "),
    "drop quote": ((answer('Say <|"|>yes<|"|>. '),), "Say yes. "),
    "eos splice": ((answer(RAW2),), CLEAN2),
    "split tool_call": ((answer("Press <|tool"), answer("_call> now. ")), "Press "),
    "split end_of_turn": ((answer("Hi <end_of"), answer("_turn> bye. ")), "Hi "),
    "split eos splice": (
        (
            answer("Sure, <|chan"),
            answer("<eos>nel>thought SEC"),
            answer("RET-THOUGHT<channel|>ok. "),
        ),
        CLEAN2,
    ),
    "split across open": (
        (answer("Press <|tool"), open_answer("draft"), answer("_call> now. ")),
        "Press ",
    ),
    "stopped": ((answer("First. <eos>Second. "), answer("Third. ")), "First. "),
    "thought first": ((answer(RAW6),), CLEAN6),
    "spliced": ((answer(RAW6S),), CLEAN6S),
}
#: hold at the base already (T-0121 removes the block, ``<eos>`` with it): the stage-order guard rows
PLAYED_GREEN = frozenset({"thought first"})


@pytest.mark.parametrize("case", [pytest.param(c) for c in PLAYED])
async def test_history_equals_what_was_played(tmp_path, case):
    """Criterion 3: for every utterance played, ``normalise_for_tts(source) == text``; no ``source`` holds a
    literal; history equals the join of the played sources, in order, and that join is U(raw)."""
    commits, expected = PLAYED[case]
    rig, snap = await run_off(tmp_path, [list(commits)])
    for c in rig.speaks():
        if c.source:
            assert normalise_for_tts(c.source) == c.text, (
                f"T-0124 criterion 3 ({case}): text is the gate's rendering of source: {c}"
            )
            hits = [lit for lit in LITERALS if lit in c.source]
            assert not hits, f"T-0124 criterion 3 ({case}): a source holds {hits}: {c}"
    played = [s for _, s in rig.audio.plays if s]
    assert snap["history"] == "".join(played), (
        f"T-0124 criterion 3 ({case}): history {snap['history']!r} is the join of what was played {played}"
    )
    assert "".join(played) == expected, (
        f"T-0124 criterion 3 ({case}): what was played joins to U(raw) {expected!r}; got {played}"
    )


BARGED = "Hi <pad>there, this answer is long enough to still be playing at the cut. "
BARGED_CLEAN = "Hi there, this answer is long enough to still be playing at the cut. "


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(BARGED, id="literal"),
        # the same answer with no literal: green at the base, it proves the cut lands mid-answer on this rig's
        # timing, after the place the literal was, so the ``literal`` case's premise is not what reddens it
        pytest.param(BARGED_CLEAN, id="no_literal"),
    ],
)
async def test_history_equals_what_was_played_barge_in(tmp_path, raw):
    """Criterion 3, the barge-in row: a takeover cuts the answer mid-way, after the deleted ``<pad>``. History stores
    the ``played_text`` the audio fake reports (plus ``[interrupted]``), which holds no literal, is a proper prefix
    of U(raw) and is longer than ``Hi there,`` (the cut is after the literal)."""
    async with tool_rig(tmp_path, kit=None, script=[answer(raw)]) as rig:
        await rig.ask("tell me more")
        await rig.until(
            lambda: (
                (rig.state.state is TurnState.IDLE and bool(rig.lines()))
                or (rig.state.state is TurnState.SPEAKING and any(s for _, s in rig.audio.plays))
            ),
            "the turn neither ended nor started playing",
            limit_ms=10_000,
        )
        assert rig.state.state is TurnState.SPEAKING, "premise: the answer is playing"
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
    for text in (stopped, stored):
        hits = [lit for lit in LITERALS if lit in text]
        assert not hits, (
            f"T-0124 criterion 3 (barge-in): {hits} in played_text {stopped!r} / history {stored!r}"
        )
    assert stored == f"{stopped} [interrupted]", (
        f"T-0124 criterion 3 (barge-in): history stores the played_text the audio fake reports; "
        f"played {stopped!r}, stored {stored!r}"
    )
    assert BARGED_CLEAN.startswith(stopped) and len("Hi there,") < len(stopped) < len(
        BARGED_CLEAN
    ), (
        f"T-0124 criterion 3 (barge-in): played_text {stopped!r} is a proper prefix of U(raw) cut after the "
        f"deleted literal (premise: the cut is mid-answer)"
    )


# =====================================================================================================
# Criterion 4: split across commits (adversarial: per-commit cleaning)
# =====================================================================================================

#: case -> (commits, U(raw), the exact voiced commits the machine receives, or None where the task leaves the split
#: open). Exact lists follow from the hold rule: ``<|tool``/``<end_of``/``<|endof`` can still grow into a literal and
#: are held; a later commit that completes one cuts there, so it releases nothing and (non-empty raw that cleans to
#: nothing) is not submitted.
SPLITS: dict[str, tuple[tuple[Commit, ...], str, list[str] | None]] = {
    "tool_call": ((answer("Press <|tool"), answer("_call> now. ")), "Press ", ["Press "]),
    "end_of_turn": ((answer("Hi <end_of"), answer("_turn> bye. ")), "Hi ", ["Hi "]),
    # after the cut nothing can change, so whether ``<|chan`` comes with commit 2 or at the end is not pinned
    "eos_splice": (
        (
            answer("Sure, <|chan"),
            answer("<eos>nel>thought SEC"),
            answer("RET-THOUGHT<channel|>ok. "),
        ),
        CLEAN2,
        None,
    ),
}


@pytest.mark.parametrize("case", list(SPLITS))
async def test_literal_split_across_commits(tmp_path, case):
    """Criterion 4 (adversarial; mutations: clean per commit, hold no tail): a literal split across commits gives
    the whole commit's result, and no ``LlmCommit`` the machine receives holds a literal or a prefix that later
    completes one (every release is a prefix of U(raw), which holds no literal)."""
    commits, expected, exact = SPLITS[case]
    rig, snap = await run_off(tmp_path, [list(commits)])
    texts = released(rig, expected, f"criterion 4 ({case})")
    acc = ""
    for t in texts:
        acc += t
        assert expected.startswith(acc), (
            f"T-0124 criterion 4 ({case}): the cumulative release {acc!r} is not a prefix of U(raw) {expected!r}"
        )
    if exact is not None:
        assert texts == exact, (
            f"T-0124 criterion 4 ({case}): voiced commits {texts}, expected {exact}"
        )
    assert_stored_exactly(rig, snap, expected, f"criterion 4 ({case})")
    assert_no_literal(rig, snap, f"criterion 4 ({case})", words=("now", "bye"))
    if case == "eos_splice":
        bad = [k for k, v in {"HUD": snap["HUD"], "history": snap["history"]}.items() if "SEC" in v]
        assert not bad, f"T-0124 criterion 4 ({case}): thought text reached {bad}"


async def test_literal_split_across_a_thought_commit(tmp_path):
    """Criterion 4, I15 (mutations: feed ``thought_open=True`` commits to the literal stage, flush or reset the
    hold on them): ``Press <|tool`` / a ``thought_open=True`` commit ``draft`` / ``_call> now. `` stores ``Press ``;
    the open commit reaches the machine as sent; the held ``<|tool`` is neither released nor dropped by it."""
    rig, snap = await run_off(
        tmp_path, [[answer("Press <|tool"), open_answer("draft"), answer("_call> now. ")]]
    )
    opens = [
        (e.text, e.thought_open)
        for e in rig.machine.events(E.LlmCommit)
        if e.request_id == GEN and e.thought_open
    ]
    assert opens == [("draft", True)], (
        f"T-0124 criterion 4 (I15): the open commit as sent; got {opens}"
    )
    assert released(rig, "Press ", "criterion 4 (I15)") == ["Press "]
    assert_stored_exactly(rig, snap, "Press ", "criterion 4 (I15)")
    assert_no_literal(rig, snap, "criterion 4 (I15)", words=("now",))


#: case -> (commits, U(raw)); ``f`` and ``x`` are in no ``SPECIAL_TOKENS`` entry, only in ``<end_of_turn>`` and
#: ``<|endoftext|>``, so a hold alphabet built from ``SPECIAL_TOKENS`` alone releases these tails at once
EOT_PREFIXES = {
    "end_of_turn": (("Hi <end_of", "_turn> bye. "), "Hi "),
    "endoftext": (("Hi <|endof", "text|> bye. "), "Hi "),
}


@pytest.mark.parametrize("case", list(EOT_PREFIXES))
async def test_end_of_turn_prefix_is_held(tmp_path, case):
    """Criterion 4 (adversarial; mutation: hold alphabet from ``SPECIAL_TOKENS`` only): ``Hi <end_of`` is held until
    ``_turn> bye. `` completes ``<end_of_turn>``; the machine receives exactly ``["Hi "]``; ``bye`` is never
    stored."""
    commits, expected = EOT_PREFIXES[case]
    rig, snap = await run_off(tmp_path, [[answer(t) for t in commits]])
    assert released(rig, expected, f"criterion 4 ({case})") == [expected]
    assert_stored_exactly(rig, snap, expected, f"criterion 4 ({case})")
    assert_no_literal(rig, snap, f"criterion 4 ({case})", words=("bye",))


# =====================================================================================================
# Criterion 5: stopped means stopped
# =====================================================================================================


async def test_nothing_after_the_cut_is_submitted(tmp_path):
    """Criterion 5 (mutation: keep releasing after a cut): ``First. <eos>Second. `` / ``Third. `` stores
    ``First. `` and voices ``First.``; the machine receives exactly ``["First. "]`` before ``llm.done`` and nothing
    after (no ``Second``, no ``Third``)."""
    rig, snap = await run_off(tmp_path, [[answer("First. <eos>Second. "), answer("Third. ")]])
    assert voice(rig) == "First.", f"premise (the voice is unchanged): {voice(rig)!r}"
    assert released(rig, "First. ", "criterion 5") == ["First. "]
    every = [e.text for e in rig.machine.events(E.LlmCommit) if e.request_id == GEN]
    assert not [t for t in every if "Second" in t or "Third" in t], every
    assert_stored_exactly(rig, snap, "First. ", "criterion 5")
    assert_no_literal(rig, snap, "criterion 5", words=("Second", "Third"))


# =====================================================================================================
# Criterion 6: thought first, then literals
# =====================================================================================================


async def test_thought_block_goes_first(tmp_path):
    """Criterion 6 (adversarial; mutation: L before T, which cuts at ``<|channel>`` and gives ``Here ``): ``Here
    <|channel>a<eos>b<channel|> ok. `` stores and voices ``Here  ok. `` (T-0121 criterion 1); T-0121's note counts
    the block (27 characters); no T-0124 note. Holds at the base (T-0121 already removes the block): unmarked."""
    rig, snap = await run_off(tmp_path, [[answer(RAW6)]])
    assert_stored_exactly(rig, snap, CLEAN6, "criterion 6")
    released(rig, CLEAN6, "criterion 6")
    block = "<|channel>a<eos>b<channel|>"
    assert len(block) == 27
    assert notes(rig, THOUGHT_NOTE_WHAT) == [
        {"what": THOUGHT_NOTE_WHAT, "request_id": GEN, "step": 1, "chars": 27, "unclosed": False}
    ], notes(rig, THOUGHT_NOTE_WHAT)
    assert notes(rig, NOTE_WHAT) == [], notes(rig, NOTE_WHAT)


SPLICED = {"one_commit": (RAW6S,), "split": ("Press <|tool<pa", "d>_call> now. ")}


@pytest.mark.parametrize("case", list(SPLICED))
async def test_spliced_literal(tmp_path, case):
    """Criterion 6 (mutation: one pass with no fixpoint): ``Press <|tool<pad>_call> now. `` deletes ``<pad>``,
    which splices ``<|tool_call>``, which then cuts: stores ``Press ``."""
    rig, snap = await run_off(tmp_path, [[answer(t) for t in SPLICED[case]]])
    released(rig, CLEAN6S, f"criterion 6 ({case})")
    assert_stored_exactly(rig, snap, CLEAN6S, f"criterion 6 ({case})")
    assert_no_literal(rig, snap, f"criterion 6 ({case})", words=("now",))


# =====================================================================================================
# Criterion 7: the note
# =====================================================================================================


def _note(request_id: str, chars: int, cut: bool) -> dict[str, Any]:
    return {"what": NOTE_WHAT, "request_id": request_id, "step": 1, "chars": chars, "cut": cut}


def _thought_note(request_id: str, chars: int, unclosed: bool) -> dict[str, Any]:
    return {
        "what": THOUGHT_NOTE_WHAT,
        "request_id": request_id,
        "step": 1,
        "chars": chars,
        "unclosed": unclosed,
    }


#: case -> (one script per turn, the expected T-0124 notes, the expected T-0121 notes), by hand.
#: ``chars`` = len(T(raw)) - len(U(raw)): never what T-0121 removed.
NOTE_CASES: dict[str, tuple[list[list[str]], list[dict[str, Any]], list[dict[str, Any]]]] = {
    "pad": ([["Hi <pad>there. "]], [_note(GEN, 5, False)], []),
    **{f"trim {lit}": ([[press(lit)]], [_note(GEN, len(f"{lit} now. "), True)], []) for lit in TRIMS},
    "eos splice": ([[RAW2]], [_note(GEN, CHARS2, True)], []),
    # T-0121 removes ``<|channel>x<channel|>`` (21) first; L then deletes ``<pad>`` (5): two notes, T-0121's first
    "both": ([["Here <|channel>x<channel|> ok. Hi <pad>there. "]], [_note(GEN, 5, False)],
             [_thought_note(GEN, 21, False)]),
    # ``Press <|tool<pad>_call> now. ``: len("<|tool<pad>_call> now. ") = 6 + 5 + 6 + 6 = 23
    "spliced": ([[RAW6S]], [_note(GEN, 23, True)], []),
    # a held tail ``<|to`` released at the end of the stream; the note comes after it
    "held tail": ([["A <pad>B <|to"]], [_note(GEN, 5, False)], []),
    # one note per stream: two turns, each with a literal
    "two turns": ([["Hi <pad>there. "], [press("<eos>")]],
                  [_note(GEN, 5, False), _note("t2-gen", len("<eos> now. "), True)], []),
    # nothing dropped by L: no T-0124 note (green at the base)
    "clean": ([["Sure, here it is. "]], [], []),
    "thought only": ([[RAW6]], [], [_thought_note(GEN, 27, False)]),
    "stray only": ([["Done <channel|>here. "]], [], []),
}  # fmt: skip
NOTE_GREEN = frozenset({"clean", "thought only", "stray only"})


@pytest.mark.parametrize("case", [pytest.param(c) for c in NOTE_CASES])
async def test_note_counts_chars_never_text(tmp_path, case):
    """Criterion 7 (H-T124-2; mutations: merge the count into T-0121's note, a note when ``chars == 0``, text in the
    note, count T-0121's removals in ``chars``): exactly one session-log note per stream where the literal stage
    dropped anything, exactly ``{what, request_id, step: 1, chars, cut}`` with hand-computed values; T-0121's note
    unchanged; written after the held tail is submitted, after T-0121's note, before ``llm.done``; ``llm.commit``
    keeps the raw text."""
    scripts, expected, thought_expected = NOTE_CASES[case]
    trace: list[Any] = []
    rig, _ = await run_off(tmp_path, [[answer(t) for t in s] for s in scripts], trace=trace)
    found = notes(rig, NOTE_WHAT)
    assert found == expected, f"T-0124 criterion 7 ({case}): notes {found}, expected {expected}"
    assert notes(rig, THOUGHT_NOTE_WHAT) == thought_expected, (
        f"T-0124 criterion 7 ({case}): T-0121's note is unchanged; got {notes(rig, THOUGHT_NOTE_WHAT)}"
    )
    for note in found:
        assert set(note) == NOTE_KEYS, sorted(note)
        rid = note["request_id"]
        mine = [i for i, t in enumerate(trace) if t[0] == "log" and t[1] == "note" and t[2] == note]
        assert len(mine) == 1, mine
        done = [
            i
            for i, t in enumerate(trace)
            if t[0] == "log" and t[1] == "llm.done" and t[2].get("request_id") == rid
        ]
        commits = [
            i
            for i, t in enumerate(trace)
            if t[0] == "submit"
            and isinstance(t[1], E.LlmCommit)
            and t[1].request_id == rid
            and t[1].segment == "answer"
        ]
        theirs = [
            i
            for i, t in enumerate(trace)
            if t[0] == "log"
            and t[1] == "note"
            and t[2].get("what") == THOUGHT_NOTE_WHAT
            and t[2].get("request_id") == rid
        ]
        assert commits and done and commits[-1] < mine[0] < done[0], (
            f"T-0124 criterion 7 ({case}): the note comes after the stream's last answer commit (the held tail) "
            f"and before llm.done; commits {commits}, note {mine}, llm.done {done}"
        )
        assert all(i < mine[0] for i in theirs), (
            f"T-0124 criterion 7 ({case}): T-0121's note comes first; T-0121 {theirs}, T-0124 {mine}"
        )
    for k, script in enumerate(scripts, start=1):
        logged = "".join(
            r["text"]
            for r in rig.kind("llm.commit")
            if r.get("segment") == "answer" and r.get("request_id") == f"t{k}-gen"
        )
        assert logged == "".join(script), (
            "T-0124 criterion 7: llm.commit keeps the raw text verbatim"
        )
    if case == "held tail":
        assert answer_commits(rig, GEN)[-1:] == ["<|to"], (
            f"premise: the held tail ``<|to`` is the stream's last answer commit; got {answer_commits(rig, GEN)}"
        )


# =====================================================================================================
# Criterion 9: tools on, guard rows (green at the base, no marker)
# =====================================================================================================

#: case -> commits; criteria 1, 2 and 4's rows, tools on (default kit), loop off
TOOLS_ON: dict[str, tuple[Commit, ...]] = {
    **{f"trim {lit}": (answer(press(lit)),) for lit in TRIMS},
    "drop pad": (answer("Hi <pad>there. "),),
    "drop quote": (answer('Say <|"|>yes<|"|>. '),),
    "eos splice": (answer(RAW2),),
    "split tool_call": (answer("Press <|tool"), answer("_call> now. ")),
    "split end_of_turn": (answer("Hi <end_of"), answer("_turn> bye. ")),
    "split eos splice": (
        answer("Sure, <|chan"),
        answer("<eos>nel>thought SEC"),
        answer("RET-THOUGHT<channel|>ok. "),
    ),
    "split across open": (answer("Press <|tool"), open_answer("draft"), answer("_call> now. ")),
}


async def _tools_on(tmp_path: Path, commits: tuple[Commit, ...]) -> tuple[ToolRig, dict[str, Any]]:
    async with tool_rig(tmp_path, script=list(commits)) as rig:
        await rig.ask("what does it say")
        await rig.turn_done()
        await rig.advance(300)
        snap = snapshot(rig)
    return rig, snap


#: T-0126 re-scope R2: tools on, the gate trim literals that are not tokenizer literals are cut (T-0126 criteria 1-2);
#: the history T-0126 stores for these rows, by hand. Red at 4a25a85 (which stores them verbatim): pending T-0126.
KEPT = {
    "trim <end_of_turn>": "Press ",
    "trim <|endoftext|>": "Press ",
    "split end_of_turn": "Hi ",
}


@pytest.mark.parametrize(
    "case",
    [pytest.param(c) for c in TOOLS_ON],
)
async def test_tools_on_rows_store_no_literal(tmp_path, case):
    """Criterion 9 (negative; mutation: route tools-on through the new stage): with the default kit, loop off,
    ``_AnswerCleaner`` deletes every tokenizer literal (I4), so these rows store no literal and no ``SECRET``;
    byte-identity with 8b3f6d0 is ``test_t0124_differential``'s.

    T-0126 re-scope R2 (docs/tasks/T-0126.md, "Owned files"; a spec change to criterion 9 as re-read in T-0124's
    Rulings): "no literal" is checked over ``SPECIAL_TOKENS ∪ speech_gate._TRIM_TOKENS``, and the three ``KEPT`` rows
    (``<end_of_turn>`` / ``<|endoftext|>``, which T-0126 cuts with tools on) expect T-0126's history. Those three are
    pending T-0126; the others hold at the base, unmarked."""
    _, snap = await _tools_on(tmp_path, TOOLS_ON[case])
    for key in (
        "TtsSpeak.source",
        "audio plays",
        "history",
        "history (all)",
        "turns.jsonl",
        "memory episodes",
    ):
        v = snap[key] or ""
        hits = [lit for lit in LITERALS if lit in v] + (["SECRET"] if "SECRET" in v else [])
        assert not hits, f"T-0124 criterion 9 ({case}): {hits} in {key}: {v[:300]!r}"
    assert "SECRET" not in snap["HUD"], f"T-0124 criterion 9 ({case}): SECRET in the HUD"
    if case == "trim <|tool_call>":
        # the task's example (I4: tools on deletes, tools off cuts)
        assert (snap["history"], snap["TtsSpeak.text"]) == ("Press  now. ", "Press now."), (
            f"T-0124 criterion 9: tools on, {press('<|tool_call>')!r} stores 'Press  now. ' and voices "
            f"'Press now.'; got {snap['history']!r} / {snap['TtsSpeak.text']!r}"
        )
    # T-0126 re-scope R2 (criteria 1-2 of T-0126): ``<end_of_turn>`` and ``<|endoftext|>`` are not SPECIAL_TOKENS,
    # so I4 does not delete them; T-0126 cuts there with tools on, as the gate does, so history is ``KEPT[case]``.
    kept = KEPT
    if case in kept:
        assert snap["history"] == kept[case], (
            f"T-0126 R2 ({case}): tools on cuts at the gate's end-of-turn literals, so history is "
            f"{kept[case]!r}; got {snap['history']!r}"
        )
