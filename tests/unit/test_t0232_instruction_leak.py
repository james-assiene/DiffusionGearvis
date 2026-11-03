"""T-0232: speech never voices a sentence of Gearvis's own instructions.

Found at James's desk, 2026-11-02 12:26, turn t7 (session log ``logs/sessions/
20261102-122638-2470211.jsonl``, seq 8812-8816): after 20 browser steps the answer of step s21 was the
commit ``" Use no markdown, no lists, and numbers as words when short."`` and it was voiced
(``tts.utterance`` t7-u1). The same commit recurs in turn t8 (seq 11745). Copied here as a literal;
this module never reads ``logs/``.

Every expected value comes from the task's acceptance criteria, the ``prompt_builder`` constants (the
source the task names), hand-written rows, or the frozen copy of the gate at main eb58ea00
(``_frozen_speech_gate_t0232_base.py``) for criterion 3's "voices exactly what main voices". Nothing
is computed by re-running the new gate's own logic.

**The rule under test (criterion 1 as amended by ruling R-1).** Matching is on word sequences:
lowercase, keep only ``[a-z0-9_]+`` words, drop the fillers ``{use, and, also, please, always, then,
so, you, are, i}`` on both sides. An answer sentence is an instruction leak when (forward) its
remaining sequence, at least 9 words (rulings R-7/R-7b; R-1 said 6), is a contiguous run of one instruction
sentence's sequence, or
(reverse, self-talk quoting the instructions) it contains one instruction sentence's whole sequence,
at least 6 words, contiguously. A leak is refused with reason ``"instructions"``: ``text == ""``,
``source`` byte-exact, counted, its source and reason index-matched in
``refused_sources``/``silenced_reasons`` (criterion 2), and the rest of the turn is *not* latched.
After a T-0224 repeat latch the reason is ``"repeat"`` (R-4). The forward 9-word floor is pinned on
both sides by 8-word runs (voiced) and 9-word runs (refused) of the same instruction sentences; the
reverse 6-word floor by "Let me answer the user briefly." (voiced: NATIVE's 4-word first sentence is
below it) against NATIVE 3's 6 words embedded in a longer sentence (refused).

**Name this module reads (criterion 4, R-5).** ``speech_gate.INSTRUCTION_SENTENCES``: an iterable of
``str``, one instruction sentence per item, compared after T-0224's normalisation, so its exact
casing/punctuation is free. Only ``prompt_builder``'s constants are in it (R-2: the persona and the
screen reminder in ``session.py`` are out of scope; their rows were dropped).

**Readings stated rather than guessed:**

- ``USER_INSTRUCTIONS_HEADER`` is part of the user's own block, so it is not in the list and its
  sentences are voiced (R-3).
- ``UI_INSTRUCTIONS`` (the A2UI block) is not "persona, speaking and tool guidance": not pinned
  either way, except that the P4 r1 corpus row of criterion 3 must voice what main voices.
- S_SOURCE's ``google_search`` is one word in its source and two in its spoken text; the row
  expects a refusal whichever form the gate compares.

Regime: pure CPU, n = 1 per row unless a Hypothesis row says otherwise, no timing claimed.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from local_astra.conductor import prompt_builder as pb
from local_astra.conductor import speech_gate as speech_gate_module
from local_astra.conductor.speech_gate import SpeechGate
from local_astra.contracts.llm import Commit
from tests.unit import _frozen_speech_gate_t0232_base as frozen
from tests.unit._hud_rig import answer, frame, running
from tests.unit._session_log_rig import SPEECH_REFUSED, make_log, of_kind, read_records
from tests.unit.test_speech_gate import ABBREVIATION_ROWS, FENCE_INPUTS
from tests.unit.test_t0224_speech_repeat_and_maths import MATHS_ROWS, P4_COMMITS

REPO = Path(__file__).resolve().parents[2]

# --- helpers -------------------------------------------------------------------------------------


def commit(text: str, idx: int = 0) -> Commit:
    return Commit(request_id="t1-gen", canvas_idx=idx, segment="answer", text=text)


def speak(gate: Any, *chunks: str) -> list[Any]:
    """Feed the chunks as successive canvases, flush, return every sentence (refused included),
    asserting the turn machine's conservation law at every step (T-0004 A1, T-0052 A20)."""
    out: list[Any] = []
    consumed = ""
    for i, chunk in enumerate(chunks):
        out.extend(gate.feed(commit(chunk, i)))
        consumed += chunk
        rebuilt = "".join(s.source for s in out) + gate.pending
        assert rebuilt == consumed, (
            "conservation: sources plus pending must rebuild every byte fed; the turn machine "
            f"raises ValueError otherwise and the turn goes silent. lost {len(consumed) - len(rebuilt)}"
        )
    out.extend(gate.close())
    assert "".join(s.source for s in out) == consumed and gate.pending == ""
    return out


def silenced_reasons(gate: Any) -> list[str]:
    """T-0224 R-2: ``SpeechGate.silenced_reasons``, index-matched to ``refused_sources``. Read,
    never called."""
    reasons = getattr(gate, "silenced_reasons", None)
    assert isinstance(reasons, list), f"the gate exposes no list silenced_reasons; got {reasons!r}"
    return list(reasons)


def norm(sentence: str) -> str:
    """The task's normalised form, written independently: casefold, whitespace collapsed, trailing
    punctuation (and closing brackets/quotes) dropped."""
    collapsed = " ".join(sentence.casefold().split())
    return collapsed.rstrip(".!?,;:\"')]} ")


#: R-1's filler words, dropped on both sides before matching.
FILLERS = frozenset({"use", "and", "also", "please", "always", "then", "so", "you", "are", "i"})


def words(text: str) -> list[str]:
    """R-1's word sequence, written from the ruling for the rows' *setup* checks only (the gate's
    own matcher is never called): lowercase, ``[a-z0-9_]+`` words, fillers dropped."""
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in FILLERS]


def is_run(needle: list[str], hay: list[str]) -> bool:
    """``needle`` is a contiguous run of ``hay`` (brute force)."""
    n = len(needle)
    return any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


def _constant_sentences() -> list[tuple[str, str]]:
    """Every sentence of the in-scope prompt_builder constants, split by hand on ``. `` etc. (a
    setup oracle, coarse on purpose: the rows only need "is a run of some instruction sentence")."""
    texts = {
        "CLOSE_THE_THOUGHT": pb.CLOSE_THE_THOUGHT,
        "SPOKEN_STYLE": pb.SPOKEN_STYLE,
        "NATIVE_TOOL_INSTRUCTIONS": pb.NATIVE_TOOL_INSTRUCTIONS,
        **{f"LIVE_GUIDANCE[{k}]": v for k, v in pb.LIVE_GUIDANCE.items()},
        "WAIT_TOOL_GUIDANCE": pb.WAIT_TOOL_GUIDANCE,
    }
    return [
        (name, s)
        for name, text in texts.items()
        for s in re.split(r"(?<=[.!?])\s+", text.strip())
        if s
    ]


ALL_ROWS = _constant_sentences()


def instruction_list() -> list[str]:
    """Criterion 4: ``speech_gate.INSTRUCTION_SENTENCES`` (this module's name for the list; the
    task names none). Missing is an assertion, not an ``AttributeError``."""
    found = getattr(speech_gate_module, "INSTRUCTION_SENTENCES", None)
    assert found is not None and not isinstance(found, str), (
        "criterion 4: speech_gate exposes no INSTRUCTION_SENTENCES list derived from prompt_builder"
    )
    items = list(found)
    assert items and all(isinstance(s, str) for s in items), (
        f"criterion 4: INSTRUCTION_SENTENCES must be non-empty strings; got {items[:3]!r}"
    )
    return items


def assert_refused_alone(raw: str) -> None:
    """``raw`` as the whole answer: refused, reason ``instructions``, source kept, counted."""
    gate = SpeechGate()
    out = speak(gate, raw)
    assert [s.text for s in out] == [""], (
        f"criterion 1: an instruction sentence is never voiced; {raw!r} gave texts {[s.text for s in out]}"
    )
    assert [s.source for s in out] == [raw], "setup: one sentence, source byte-exact (A20)"
    assert gate.refused_count == 1, f"criterion 2: the refusal is counted; got {gate.refused_count}"
    assert gate.refused_sources == [raw], (
        f"criterion 2: the refused source is retrievable byte-exact; got {gate.refused_sources}"
    )
    assert silenced_reasons(gate) == ["instructions"], (
        f"criterion 1: the reason is 'instructions'; got {silenced_reasons(gate)}"
    )


def assert_voiced_as_main(raw: str) -> None:
    """``raw`` as the whole answer gives exactly what the frozen main gate gives (and main voices
    something), with nothing refused."""
    base_gate = frozen.SpeechGate()
    base = speak(base_gate, raw)
    assert any(s.text for s in base) and base_gate.refused_count == 0, "setup: main voices it"
    gate = SpeechGate()
    out = speak(gate, raw)
    assert [(s.source, s.text) for s in out] == [(s.source, s.text) for s in base], (
        f"too wide: {raw!r} is not an instruction sentence under criterion 1 and must be voiced as "
        f"on main; got {[s.text for s in out]}"
    )
    assert gate.refused_count == 0 and silenced_reasons(gate) == []


# --- criterion 1: the leak and other instruction sentences are refused ------------------------------

#: The committed answer of turn t7 step s21, byte-exact (seq 8812), voiced on main as t7-u1.
LEAK = " Use no markdown, no lists, and numbers as words when short."

#: The speaking sentence criterion 4 names.
SPEAKING = (
    "You are speaking aloud: short sentences, no markdown, no lists, numbers as words when short."
)


def test_the_leak_is_voiced_on_main() -> None:
    """Setup: the frozen main gate voices the logged commit, as the log says (t7-u1). Green today."""
    out = speak(frozen.SpeechGate(), LEAK)
    assert [s.text for s in out] == ["Use no markdown, no lists, and numbers as words when short."]


def test_the_exact_leaked_sentence_is_refused() -> None:
    """The task's finding, under R-1: the leaked t7 answer is never voiced, reason ``instructions``.
    It paraphrases ``SPEAKING`` ("Use" and "and" added, both fillers); its remaining words "no
    markdown no lists numbers as words when short" (9) are a contiguous run of SPEAKING's."""
    leak, spoken = words(LEAK), words(SPEAKING)
    assert len(leak) == 9 and is_run(leak, spoken), (
        f"setup (hand oracle of R-1): {leak} in {spoken}"
    )
    assert_refused_alone(LEAK)


#: Instruction sentences copied verbatim from the prompt_builder constants they come from, one per
#: constant group the criterion names (speaking, closing the thought, native tools, v1, v2/v3, v3,
#: v4's two sentences, the wait guidance).
INSTRUCTION_ROWS: list[tuple[str, str]] = [
    ("SPOKEN_STYLE 1", SPEAKING),
    ("SPOKEN_STYLE 2", "If you also show something, put it in the ui channel, never in speech."),
    ("CLOSE_THE_THOUGHT 2", "Never leave the answer inside the thinking channel."),
    ("NATIVE_TOOL_INSTRUCTIONS 3", "Use at most one call per answer."),
    (
        "LIVE_TOOL_GUIDANCE v1 1",
        "After an action, the next screenshot shows the screen as it is now.",
    ),
    (
        "LIVE_TOOL_GUIDANCE_V2 1",
        "The latest screenshot in this turn already shows the screen as it is after any action you "
        "have taken.",
    ),
    (
        "LIVE_TOOL_GUIDANCE 3",
        "When asked about a whole page (the cheapest item, every item, a list), scroll to see the "
        "rest of the page before answering.",
    ),
    (
        "LIVE_TOOL_GUIDANCE_V3 4",
        "To leave the current website for another, press Ctrl+L, which opens the start page, then "
        "type the other site's web address (such as github.com) into the start page's address box "
        "and press Enter, not into the current site's search box.",
    ),
    (
        "S_SHARED_GUIDANCE (v4)",
        "A shared screen shows what the user means, but you act only in your own browser, which is "
        "always available: to do a web task from a shared screen, press Ctrl+L to open your "
        "browser's start page and continue there.",
    ),
    (
        "S_SOURCE_GUIDANCE (v4, its source says google_search, its spoken text google search)",
        "Before you answer with a fix or a fact you found through google_search, open the page that "
        "states it, read it there, and name that page in your answer.",
    ),
    (
        "WAIT_TOOL_GUIDANCE",
        "If the latest screenshot shows the page still loading or the results you expect not there "
        "yet, call wait and look at the new screenshot instead of telling the user to wait or "
        "refresh.",
    ),
]


def test_the_rows_are_sentences_of_the_prompt_builder_constants() -> None:
    """Setup, green today: every row above is verbatim inside the system text prompt_builder builds
    from its constants, so the rows test the instructions and not a typo."""
    corpus = "\n".join(
        [
            pb.CLOSE_THE_THOUGHT,
            pb.SPOKEN_STYLE,
            pb.NATIVE_TOOL_INSTRUCTIONS,
            *pb.LIVE_GUIDANCE.values(),
            pb.WAIT_TOOL_GUIDANCE,
        ]
    )
    for name, row in INSTRUCTION_ROWS:
        assert row in corpus, f"{name}: {row!r} is not in prompt_builder's constants"


@pytest.mark.parametrize(("name", "row"), INSTRUCTION_ROWS, ids=[n for n, _ in INSTRUCTION_ROWS])
def test_an_instruction_sentence_is_refused(name: str, row: str) -> None:
    """Criterion 1: an answer sentence equal to an instruction sentence is refused (``text ''``),
    reason ``instructions``, counted and retrievable (criterion 2)."""
    assert_refused_alone(row)


@pytest.mark.parametrize(
    "variant",
    [
        # case, doubled whitespace, a different final punctuation (T-0224's normalisation)
        "YOU ARE SPEAKING ALOUD:  short sentences, no markdown, no lists, numbers as words when short!",
        "never leave the answer\ninside the thinking channel",
        "  Use at most one call per answer...",
    ],
)
def test_a_normalised_variant_of_an_instruction_sentence_is_refused(variant: str) -> None:
    """Criterion 1: the comparison is on the normalised form, so case, whitespace and trailing
    punctuation do not let an instruction sentence through."""
    gate = SpeechGate()
    out = speak(gate, variant)
    assert [s.text for s in out] == [""], (
        f"criterion 1: {variant!r} voiced as {[s.text for s in out]}"
    )
    assert silenced_reasons(gate) == ["instructions"]


@pytest.mark.parametrize(
    "fragment",
    [
        # word-bounded tail of SPOKEN_STYLE's first sentence
        "No markdown, no lists, numbers as words when short.",
        # middle run of NATIVE_TOOL_INSTRUCTIONS' second sentence (9 words)
        "First say in one short sentence what you will do.",
        # word-bounded middle of LIVE_TOOL_GUIDANCE_V2's second sentence
        "Before saying an action worked, check that the latest screenshot shows its effect.",
    ],
)
def test_a_forward_run_of_an_instruction_sentence_is_refused(fragment: str) -> None:
    """R-1/R-7b forward: an answer sentence of 9+ remaining words that is a contiguous run of an
    instruction sentence's words is refused."""
    assert norm(fragment) in norm(
        " ".join([pb.SPOKEN_STYLE, pb.NATIVE_TOOL_INSTRUCTIONS, pb.LIVE_TOOL_GUIDANCE_V2])
    ), "setup: the fragment is inside an instruction"
    assert_refused_alone(fragment)


def test_an_instruction_sentence_split_across_canvases_is_refused() -> None:
    """Criterion 1 on the multi-canvas path: the segmenter joins the canvases first."""
    gate = SpeechGate()
    out = speak(
        gate,
        "You are speaking aloud: short sen",
        "tences, no markdown, no lists, ",
        "numbers as words when short.",
    )
    assert [s.text for s in out] == [""], f"criterion 1: voiced {[s.text for s in out]}"
    assert silenced_reasons(gate) == ["instructions"]


# --- criterion 1: too wide ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "short",
    ["No lists.", "No markdown.", "Short sentences.", "Answer normally.", "Call no tool."],
)
def test_a_short_instruction_fragment_is_voiced(short: str) -> None:
    """R-1: a run of fewer than 6 remaining words is not a leak. Each row (2-3 words) is a run of an
    instruction sentence and is still voiced exactly as on main."""
    assert len(words(short)) < 6 and any(is_run(words(short), words(s)) for _, s in ALL_ROWS), (
        "setup"
    )
    assert_voiced_as_main(short)


#: R-7b's forward floor of 9 remaining words, both sides, on the same instruction sentences (v1-v3's
#: "whole page" sentence, v2's verification sentence, SPOKEN_STYLE's first): runs of 5-8 words are
#: voiced, runs of 9 refused. None of these rows holds a whole instruction sentence (no reverse match).
SHORT_RUNS = (
    "Scroll to see the rest.",
    "Check that the latest screenshot.",
    "No markdown, no lists, numbers.",
    "Scroll to see the rest of.",
    "Check that the latest screenshot shows.",
    "Scroll to see the rest of the.",
    "Check that the latest screenshot shows its.",
    "No markdown, no lists, numbers as words.",
    # 8 words: refused under R-7's 8, voiced under R-7b's 9 (R-7's accepted risk, now voiced)
    "Scroll to see the rest of the page.",
    "Check that the latest screenshot shows its effect.",
    "No markdown, no lists, numbers as words when.",
)
NINE_WORD_RUNS = (
    "Scroll to see the rest of the page before.",
    "Check that the latest screenshot shows its effect for.",
    "Short sentences, no markdown, no lists, numbers as words.",
)


def _is_reverse_match(text: str) -> bool:
    return any(len(words(s)) >= 6 and is_run(words(s), words(text)) for _, s in ALL_ROWS)


@pytest.mark.parametrize("run", SHORT_RUNS)
def test_a_run_of_fewer_than_nine_words_is_voiced(run: str) -> None:
    """R-7b, too wide: 5-8 remaining words, a contiguous run of an instruction sentence, voiced as on
    main. Fails until the forward floor is 9 for the 6-, 7- and 8-word rows."""
    w = words(run)
    assert 5 <= len(w) <= 8 and any(is_run(w, words(s)) for _, s in ALL_ROWS), f"setup: {w}"
    assert not _is_reverse_match(run), "setup: no whole instruction sentence inside"
    assert_voiced_as_main(run)


@pytest.mark.parametrize("run", NINE_WORD_RUNS)
def test_a_nine_word_run_of_an_instruction_is_refused(run: str) -> None:
    """R-7b, absent: 9 remaining words, a contiguous run of an instruction sentence, refused (the
    shortest logged forward leak, 2026-11-02, has 9)."""
    w = words(run)
    assert len(w) == 9 and any(is_run(w, words(s)) for _, s in ALL_ROWS), f"setup: {w}"
    assert not _is_reverse_match(run), "setup: forward only"
    assert_refused_alone(run)


#: R-7's named natural false positives (review at 90f18e38), voiced.
R7_FALSE_POSITIVES = (
    "If it does not, try again.",
    "The latest screenshot shows the page still loading.",
)


@pytest.mark.parametrize("row", R7_FALSE_POSITIVES)
def test_r7_natural_false_positives_are_voiced(row: str) -> None:
    """R-7/R-7b: the two sentences the review named are voiced as on main. The second is an exact
    8-word run of WAIT_TOOL_GUIDANCE, which is why the floor is 9 (R-7b)."""
    assert len(words(row)) <= 8, "setup"
    assert not _is_reverse_match(row), "setup"
    assert_voiced_as_main(row)


@pytest.mark.parametrize(
    "near",
    [
        # every word but one is a run of NATIVE 3; "recall" is not "call", "answe" is not "answer"
        "Use at most one recall per answer.",
        "Use at most one call per answe.",
        # in order inside v1-v3's "whole page" sentence but not contiguous ("rest of the" skipped):
        # R-1 says a contiguous run, not a subsequence
        "Scroll to see the page before answering.",
        "Check that the screenshot shows its effect.",
        # what the model says while working: shares words with v1-v3, is not inside one
        "I will scroll to see the rest of the page.",
        "The latest screenshot shows the crate is now shiny metal.",
        "Scroll to see the rest of the pages before answering.",
        # the speaking sentence's words, in the user's own subject
        "You are speaking aloud to me in short sentences today.",
    ],
)
def test_a_near_miss_is_voiced_as_on_main(near: str) -> None:
    """Too-wide direction (rules/tests.md): sentences that share words with an instruction but whose
    word sequence is neither a run of one (forward) nor contains a whole one (reverse) are voiced
    exactly as on main."""
    w = words(near)
    for _, s in ALL_ROWS:
        assert not is_run(w, words(s)), f"setup: {near!r} is a run of {s!r}"
        assert not (len(words(s)) >= 6 and is_run(words(s), w)), f"setup: {near!r} holds {s!r}"
    assert_voiced_as_main(near)


# --- R-1 reverse: self-talk that quotes a whole instruction sentence ------------------------------

#: Two answer commits found by the T-0232 replay of the logged sessions, byte-exact (copied once):
#: ``logs/t0207-completion-check-ab-cb016fd/sessions/20261030-093123-510840.jsonl`` seq 333 (request
#: t1-gen-s4) and ``logs/t0212-threejs-arms-5b87b72/sessions/20261031-110733-199725.jsonl`` seq 1013
#: (request t1-gen-s21). Both were voiced in full on main.
QUOTE_T0207 = (
    '\nWait, the prompt says "This screenshot is the browser screen now, after your last action." but '
    "I don't see the updated screenshot in the text block. Usually, the system provides a new "
    "screenshot.\n\nActually, I just called `click_at`. I should check if it worked.\nWait, I don't have "
    "the new screenshot yet. I have to assume it worked or wait for the next turn.\nActually, the "
    'instructions say "The latest screenshot in this turn already shows the screen as it is after any '
    "action you have taken. Before saying an action worked, check that the latest screenshot shows its "
    "effect\".\n\nSince I don't see the new screenshot in the thought block, I will just state I've "
    "added it.\n\nWait, looking at the prompt again, there is no new screenshot after the `click_at`.\n\n"
    "I have added the cast-iron skillet to the cart."
)
QUOTE_T0212 = (
    '\nWait, the instructions say "When an action is needed, first say in one short sentence what you '
    "will do, then call exactly one of the declared tools. When no action is needed, answer normally "
    "and call no tool.\"\n\nI haven't finished. I need to set Y and Z.\nI will try to type 2 into X again."
)


@pytest.mark.parametrize(
    ("commit_text", "leaked_prefixes"),
    [
        # sentence 1 holds v2's first sentence whole (reverse); sentence 2 is a run of v2's second
        (
            QUOTE_T0207,
            ('Actually, the instructions say "The latest', "Before saying an action worked"),
        ),
        # sentence 1 holds NATIVE's second sentence whole (reverse); sentence 2 is NATIVE's fourth
        (QUOTE_T0212, ('\nWait, the instructions say "When an action', "When no action is needed")),
    ],
    ids=["t0207_s4", "t0212_s21"],
)
def test_self_talk_quoting_a_whole_instruction_sentence_is_refused(
    commit_text: str, leaked_prefixes: tuple[str, ...]
) -> None:
    """R-1 reverse: a sentence that *contains* a whole instruction sentence (6+ words) is refused,
    the quote's second sentence is refused forward, and every other sentence of the commit is voiced
    exactly as on main (no latch). Expected values: main's segmentation and texts, with the two
    named sentences blanked by hand."""
    base = speak(frozen.SpeechGate(), commit_text)
    leaked = [
        i
        for i, s in enumerate(base)
        if s.source.lstrip("\n").startswith(tuple(p.lstrip("\n") for p in leaked_prefixes))
    ]
    assert len(leaked) == 2 and all(base[i].text for i in leaked), "setup: main voices both"
    expected = [("" if i in leaked else s.text) for i, s in enumerate(base)]
    gate = SpeechGate()
    out = speak(gate, commit_text)
    assert [s.source for s in out] == [s.source for s in base], "segmentation is unchanged"
    assert [s.text for s in out] == expected, (
        f"R-1 reverse: only the two quoted sentences are refused; got {[s.text for s in out]}"
    )
    assert silenced_reasons(gate) == ["instructions", "instructions"]


@pytest.mark.parametrize(
    "row",
    [
        # NATIVE 3's whole sequence (6 words) inside a longer sentence
        "Remember to use at most one call per answer here.",
        # SPEAKING's whole sequence inside a quote
        'The prompt told me: "you are speaking aloud: short sentences, no markdown, no lists, '
        'numbers as words when short" so I will.',
    ],
)
def test_a_sentence_embedding_a_whole_instruction_sentence_is_refused(row: str) -> None:
    """R-1 reverse, hand-built rows: extra words before and after do not let a whole instruction
    sentence through."""
    assert any(len(words(s)) >= 6 and is_run(words(s), words(row)) for _, s in ALL_ROWS), "setup"
    assert not any(is_run(words(row), words(s)) for _, s in ALL_ROWS), "setup: not forward"
    assert_refused_alone(row)


# --- criterion 1: the user's own instructions block is not part of the list ------------------------


def test_the_user_instructions_header_is_not_in_the_list() -> None:
    """Criterion 1: "the user's own instructions block excluded". Its fixed header is part of that
    block (this module's reading), so neither of its sentences is in the list."""
    keys = {norm(s) for s in instruction_list()}
    for sentence in (
        "The user's own instructions follow.",
        "Apply them unless they conflict with anything above.",
    ):
        assert sentence in pb.USER_INSTRUCTIONS_HEADER, "setup"
        assert norm(sentence) not in keys, (
            f"criterion 1: {sentence!r} is the user's block, not ours"
        )


@pytest.mark.parametrize(
    "row",
    ["The user's own instructions follow.", "Apply them unless they conflict with anything above."],
)
def test_a_user_instructions_header_sentence_is_voiced(row: str) -> None:
    """Criterion 1, behaviour: the header sentences are voiced as on main."""
    assert_voiced_as_main(row)


def test_the_gate_takes_no_user_instructions() -> None:
    """Criterion 1: the list is fixed at import; a gate is built with no arguments, so nothing the
    user typed can join it (the session constructs ``SpeechGate()`` per turn)."""
    import inspect

    params = [p for p in inspect.signature(SpeechGate).parameters.values() if p.default is p.empty]
    assert params == [], (
        f"SpeechGate() must stay constructible with no arguments; required {params}"
    )
    assert instruction_list()  # and the list exists


# --- criterion 2: no latch, reasons index-matched, history --------------------------------------

AFTER = "The crate is now shiny metal."
LATER = "I clicked the material tab and set metallic to one."


def test_the_refusal_does_not_latch_the_rest_of_the_turn() -> None:
    """Criterion 2: "The refusal does not latch the rest of the turn." The sentences after an
    instruction sentence are voiced; only the instruction sentence is refused."""
    gate = SpeechGate()
    out = speak(gate, f"{SPEAKING} {AFTER} ", f"{LATER}")
    assert [s.text for s in out] == ["", AFTER, LATER], (
        f"criterion 2: only the instruction sentence is refused; got {[s.text for s in out]}"
    )
    assert gate.refused_sources == [f"{SPEAKING} "] and silenced_reasons(gate) == ["instructions"]


def test_voiced_refused_voiced_refused_keeps_reasons_index_matched() -> None:
    """Criterion 2: interleaved, each refusal has its own reason in order; nothing latches."""
    wait = INSTRUCTION_ROWS[-1][1]
    gate = SpeechGate()
    out = speak(gate, f"{AFTER} {SPEAKING} {LATER} {wait} Done.")
    assert [s.text for s in out] == [AFTER, "", LATER, "", "Done."]
    assert gate.refused_sources == [f"{SPEAKING} ", f"{wait} "]
    assert silenced_reasons(gate) == ["instructions", "instructions"]


def test_an_instruction_sentence_said_twice_is_refused_as_instructions_both_times() -> None:
    """Criteria 1-2 with T-0224: a refused sentence is not voiced, so its second occurrence is not a
    repeat of a voiced sentence; both refusals say ``instructions`` and nothing after them latches."""
    gate = SpeechGate()
    out = speak(gate, f"{SPEAKING} {SPEAKING} {AFTER}")
    assert [s.text for s in out] == ["", "", AFTER]
    assert silenced_reasons(gate) == ["instructions", "instructions"]


def test_the_repeat_stop_still_works_after_an_instruction_refusal() -> None:
    """Criterion 2 / T-0224 unchanged: a voiced sentence repeated after an instruction refusal is
    still a repeat, and latches as T-0224 says."""
    gate = SpeechGate()
    out = speak(gate, f"{LATER} {SPEAKING} {LATER} {AFTER}")
    assert [s.text for s in out] == [LATER, "", "", ""]
    assert silenced_reasons(gate) == ["instructions", "repeat", "repeat"]


def test_an_instruction_sentence_after_a_repeat_latch_reports_repeat() -> None:
    """R-4: once T-0224's repeat stop has latched, a later instruction sentence is refused with
    reason ``repeat`` (the latch wins). Main already refuses it as a repeat-latched sentence, so
    this row pins that T-0232 does not change the reason."""
    gate = SpeechGate()
    out = speak(gate, f"{LATER} {LATER} {SPEAKING} {AFTER}")
    assert [s.text for s in out] == [LATER, "", "", ""]
    assert silenced_reasons(gate) == ["repeat", "repeat", "repeat"]


# --- the review's killing rows (T-0232 review at 90f18e38) -------------------------------------


def test_branch_order_an_instruction_sentence_is_not_taken_for_a_repeat() -> None:
    """Review row 1, branch order: the voiced first sentence "Numbers as words when short." (5 words,
    voiced; 27 normalised characters) ends the speaking sentence, so by T-0224 R-3/R-7 the speaking
    sentence is also a *repeat*. The instruction check must win before the repeat check: reason
    ``instructions``, no latch, the third sentence voiced. A gate that checks the repeat first
    refuses it as ``repeat`` and latches the third."""
    gate = SpeechGate()
    out = speak(gate, f"Numbers as words when short. {SPEAKING} It has the reviews.")
    assert [s.text for s in out] == ["Numbers as words when short.", "", "It has the reviews."], (
        f"review row 1: got {[s.text for s in out]}"
    )
    assert silenced_reasons(gate) == ["instructions"]


def test_a_refused_instruction_sentence_is_not_a_voiced_key() -> None:
    """Review row 2, rebuilt for R-7b: "To see the rest of the page before answering." (9 words, a
    run of v1-v3's "whole page" sentence) is refused as instructions; the next sentence embeds it at
    its end (word-bounded, so a T-0224 repeat of a *voiced* key), but a refused sentence is not
    voiced, so it is voiced ("ll scroll ..." is not a run of the instruction), and so is "Done now."."""
    first = "To see the rest of the page before answering."
    assert len(words(first)) == 9 and any(is_run(words(first), words(s)) for _, s in ALL_ROWS)
    second = "Then I'll scroll to see the rest of the page before answering."
    assert not any(is_run(words(second), words(s)) for _, s in ALL_ROWS), "setup: not forward"
    assert not _is_reverse_match(second), "setup: not reverse"
    gate = SpeechGate()
    out = speak(gate, f"{first} {second} Done now.")
    assert [s.text for s in out] == ["", second, "Done now."], (
        f"review row 2: got {[s.text for s in out]}"
    )
    assert silenced_reasons(gate) == ["instructions"]


def test_the_reverse_floor_voices_a_sentence_holding_a_short_instruction_sentence() -> None:
    """Review row 3, the reverse floor: "Let me answer the user briefly." holds NATIVE's whole first
    sentence "Answer the user briefly." (4 words, below the reverse floor of 6) and is 6 words, below
    the forward floor; it is voiced as on main."""
    row = "Let me answer the user briefly."
    assert words("Answer the user briefly.") == ["answer", "the", "user", "briefly"], "setup"
    assert "Answer the user briefly." in pb.NATIVE_TOOL_INSTRUCTIONS, "setup"
    assert_voiced_as_main(row)


async def test_the_refusal_is_recorded_and_history_keeps_the_committed_source(
    tmp_path: Path,
) -> None:
    """Criterion 2 end to end through a real ``Session`` over the fakes: the instruction sentence is
    not handed to TTS, the existing ``speech.refused`` record carries ``reasons == ["instructions"]``,
    and history keeps the committed source of what was voiced (T-0224 R-5), without the refused
    sentence."""
    log = make_log(tmp_path / "logs")
    async with running(
        tmp_path / "data",
        script=[answer(f"{SPEAKING} {AFTER} ", canvas_idx=0), answer(LATER, canvas_idx=1)],
        frames=[frame(0, "a")],
        ms_per_word=100,
        session_log=log,
    ) as rig:
        await rig.ask("make the crate shiny metal")
        await rig.speak_out()
        spoken = list(rig.tts.texts())
        history = list(rig.state.history)

    assert spoken == [AFTER, LATER], f"criterion 1: TTS was handed {spoken}"
    refusals = of_kind(read_records(Path(log.path)), SPEECH_REFUSED)
    assert refusals, "criterion 2: no speech.refused record for the instruction sentence"
    data = refusals[-1]["data"]
    assert data.get("reasons") == ["instructions"], (
        f"criterion 2: the record's reasons are index-matched; got {data}"
    )
    assert history, "setup: the turn left no history entry"
    said = history[-1].assistant_spoken
    assert AFTER in said and LATER in said, (
        f"criterion 2: history keeps the voiced sources; {said!r}"
    )
    assert "speaking aloud" not in said, (
        f"criterion 2: the refused sentence is not in history; {said!r}"
    )


# --- criterion 3: no instruction sentence, no change ---------------------------------------------

WORD = st.text(alphabet="abcdefghijklmnop", min_size=1, max_size=8)
LONG = st.builds(
    lambda words, end: " ".join(words).capitalize() + end,
    st.lists(WORD, min_size=1, max_size=8),
    st.sampled_from([".", "!", "?"]),
)
#: Short sentences, several of them word-bounded fragments of instruction sentences (< 20 chars).
SHORT_POOL = (
    "Yes.",
    "OK.",
    "No lists.",
    "Short sentences.",
    "Answer normally.",
    "Call no tool.",
    "Hmm?",
)
#: Natural sentences a live turn says, none an instruction sentence or inside one.
NATURAL_POOL = (
    AFTER,
    LATER,
    "I will scroll to see the rest of the page.",
    "The latest screenshot shows the crate is now shiny metal.",
    "You have a mistake in the formula for acceleration.",
    "I opened the start page and typed the address.",
)
SENTENCE = st.one_of(LONG, st.sampled_from(SHORT_POOL), st.sampled_from(NATURAL_POOL))
SEP = st.sampled_from([" ", "  ", "\n", "\n\n"])
CUTS = st.lists(st.integers(0, 400), max_size=5)


def chunked(cuts: list[int], text: str) -> list[str]:
    bounds = [0, *sorted({min(c, len(text)) for c in cuts}), len(text)]
    return [text[a:b] for a, b in pairwise(bounds)]


def assert_same_as_main(chunks: list[str]) -> None:
    base_gate = frozen.SpeechGate()
    base = speak(base_gate, *chunks)
    new_gate = SpeechGate()
    new = speak(new_gate, *chunks)
    assert [(s.source, s.text) for s in new] == [(s.source, s.text) for s in base]
    assert new_gate.refused_count == base_gate.refused_count
    assert silenced_reasons(new_gate) == list(base_gate.silenced_reasons)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(SENTENCE, SEP), min_size=1, max_size=8), CUTS)
@example([("No lists.", " "), (AFTER, " "), ("Answer normally.", "\n")], [3, 20])
def test_a_turn_without_an_instruction_sentence_voices_exactly_what_main_voices(
    pairs: Any, cuts: list[int]
) -> None:
    """Criterion 3, the T-0224 frozen-gate oracle pattern: turns of random prose, short instruction
    fragments and natural live sentences (repeats allowed, so the repeat stop is exercised on both
    sides), cut at arbitrary canvas edges, give the same sources, texts, refusal count and reasons
    as main eb58ea00. n = 200, Hypothesis default seed. Green today; must stay green."""
    text = "".join(s + sep for s, sep in pairs)
    assert_same_as_main(chunked(cuts, text))


#: The existing speech-gate corpora (criterion 3: "a property over the existing speech-gate
#: corpora"): T-0224's P4 r1 commits and maths rows, T-0004's abbreviation rows and fence inputs.
CORPUS: list[tuple[str, list[str]]] = [
    ("p4_r1", list(P4_COMMITS)),
    *[(f"maths{i}", [raw]) for i, (raw, _) in enumerate(MATHS_ROWS)],
    *[(f"abbrev_{name}", [raw]) for name, raw, _ in ABBREVIATION_ROWS],
    *[(f"fence{i}", [raw]) for i, raw in enumerate(FENCE_INPUTS)],
]


@pytest.mark.parametrize(("name", "chunks"), CORPUS, ids=[n for n, _ in CORPUS])
def test_the_existing_corpora_voice_exactly_what_main_voices(name: str, chunks: list[str]) -> None:
    """Criterion 3 over the fixed corpora: none holds an instruction sentence, so the gate's output
    equals main's row for row. Green today; must stay green."""
    assert_same_as_main(chunks)


# --- criterion 4: the list is derived from prompt_builder ----------------------------------------


def test_the_list_includes_the_speaking_sentence() -> None:
    """Criterion 4's pin: the list includes "You are speaking aloud: short sentences, no markdown, no
    lists, numbers as words when short." (compared after T-0224's normalisation)."""
    keys = {norm(s) for s in instruction_list()}
    assert norm(SPEAKING) in keys, (
        f"criterion 4: the speaking sentence is not in the list ({len(keys)} entries)"
    )


def test_the_list_covers_every_guidance_version_and_the_wait_guidance() -> None:
    """Criterion 1's scope as criterion 4 derives it: one hand-picked sentence of each row above is
    an entry of the list (v1-v4, native tools, wait, closing the thought, speaking)."""
    keys = {norm(s) for s in instruction_list()}
    missing = [
        name
        for name, row in INSTRUCTION_ROWS
        if norm(row) not in keys and norm(normalise(row)) not in keys
    ]
    assert not missing, f"criterion 1/4: these instruction sentences are not in the list: {missing}"


def normalise(row: str) -> str:
    """The spoken rendering of a row (S_SOURCE's ``google_search`` -> ``google search``): either
    form of an entry is accepted by the coverage row above."""
    return str(speech_gate_module.normalise_for_tts(row))


def test_the_list_is_derived_from_prompt_builder_at_import_not_copied() -> None:
    """Criterion 4 "derived from prompt_builder constants at import (no copy)": in a fresh
    interpreter, ``prompt_builder.SPOKEN_STYLE`` is replaced *before* ``speech_gate`` is imported;
    the list then holds the replacement's sentence and not the original speaking sentence, and the
    gate refuses the replacement. A copied literal keeps the original and fails."""
    probe = (
        "import local_astra.conductor.prompt_builder as pb\n"
        "pb.SPOKEN_STYLE = 'Zebra crossings glow violet under the harbour moon. Owls count quietly.'\n"
        "from local_astra.conductor import speech_gate as sg\n"
        "from local_astra.contracts.llm import Commit\n"
        "items = list(getattr(sg, 'INSTRUCTION_SENTENCES', ()))\n"
        "low = [' '.join(s.casefold().split()).rstrip('.!? ') for s in items]\n"
        "print('ZEBRA', 'zebra crossings glow violet under the harbour moon' in low)\n"
        "print('SPEAKING', any('you are speaking aloud' in s for s in low))\n"
        "g = sg.SpeechGate()\n"
        "out = list(g.feed(Commit(request_id='r', canvas_idx=0, segment='answer',\n"
        "    text='Zebra crossings glow violet under the harbour moon.'))) + list(g.close())\n"
        "print('VOICED', repr([s.text for s in out]))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    done = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    lines = dict(line.split(" ", 1) for line in done.stdout.strip().splitlines())
    assert lines.get("ZEBRA") == "True", (
        f"criterion 4: the list does not follow prompt_builder.SPOKEN_STYLE; got {done.stdout!r}"
    )
    assert lines.get("SPEAKING") == "False", (
        f"criterion 4: the original speaking sentence survives its constant's replacement (a copy); "
        f"got {done.stdout!r}"
    )
    assert lines.get("VOICED") == "['']", (
        f"criterion 1/4: the replaced sentence is voiced: {done.stdout!r}"
    )


def test_the_speech_gate_imports_prompt_builder_and_carries_no_copy() -> None:
    """Criterion 4 "(no copy)", the source half: speech_gate.py imports prompt_builder, and no
    instruction sentence appears in a string literal of its *code*. Scoped by ``ast`` to imports and
    non-docstring string constants, so a docstring or comment quoting a sentence does not trip it
    (rules/tests.md, source-grep guards)."""
    tree = ast.parse((REPO / "local_astra/conductor/speech_gate.py").read_text())
    imports = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = {a.name for a in node.names}
            imports |= node.module == "local_astra.conductor.prompt_builder" or (
                node.module == "local_astra.conductor" and "prompt_builder" in names
            )
        elif isinstance(node, ast.Import):
            imports |= any(a.name == "local_astra.conductor.prompt_builder" for a in node.names)
    assert imports, "criterion 4: speech_gate.py does not import prompt_builder"
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    literals = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]
    for name, row in INSTRUCTION_ROWS:
        copied = [lit for lit in literals if norm(row) in norm(lit)]
        assert not copied, f"criterion 4: {name} is copied into speech_gate.py as a code literal"
