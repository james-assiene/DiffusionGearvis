"""Speech gate (docs/03 §3, docs/01 D3). Guard tests are active; behaviour tests are T-0004."""

from __future__ import annotations

import re

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from local_astra.conductor.speech_gate import GateInputError, SpeechGate, normalise_for_tts
from local_astra.contracts.llm import Commit


def commit(text: str, segment: str = "answer", *, idx: int = 0, open_: bool = False) -> Commit:
    return Commit(request_id="r", canvas_idx=idx, segment=segment, text=text, thought_open=open_)  # type: ignore[arg-type]


# --- active: the guard is the whole point of the gate ---------------------------------------------


@pytest.mark.parametrize("segment", ["thought", "ui"])
def test_non_answer_segments_are_rejected_before_any_processing(segment):
    with pytest.raises(GateInputError):
        SpeechGate().feed(commit("never voiced", segment))


def test_unclosed_thought_is_rejected_even_when_labelled_answer():
    """docs/11 §2.1: multimodal prompts can leave the answer inside the thought channel."""
    with pytest.raises(GateInputError):
        SpeechGate().feed(commit("The answer is four", open_=True))


# --- activated for T-0004 (/implement step 3: markers removed) ------------------------------------
#
# Whitespace note (a defect found while activating these three): as originally written they
# demanded ``source == "It is a stack trace."`` *and* ``pending == "The error is on line"`` for the
# input ``"It is a stack trace. The error is on line"`` — the separating space belongs to neither,
# so ``"".join(sources) + pending`` is one byte short of the input. T-0003 criterion 10 (merged)
# makes the turn machine raise ``ValueError`` on exactly that, and T-0004 criterion 5 says the
# concatenated ``source`` *equals* the committed answer. The separator must therefore ride along
# with one of the two sides, and the task file does not say which; these tests are agnostic
# (``.strip()`` on the ``source`` side, exact on the ``text`` side that reaches TTS) and
# ``test_source_plus_pending_rebuilds_the_input_byte_for_byte`` below pins the conservation law.


def test_first_sentence_is_emitted_as_soon_as_it_is_complete():
    text = "It is a stack trace. The error is on line"
    g = SpeechGate()
    out = g.feed(commit(text))
    assert [x.source.strip() for x in out] == ["It is a stack trace."]
    assert out[0].text == "It is a stack trace."  # plain prose is unchanged
    assert g.pending.strip() == "The error is on line"
    assert "".join(x.source for x in out) + g.pending == text  # criterion 5


def test_held_clause_is_completed_by_the_next_commit_and_flushed_on_close():
    g = SpeechGate()
    g.feed(commit("The error is on line", idx=0))
    out = g.feed(commit(" four, where x is undefined. Try", idx=1))
    assert [x.source.strip() for x in out] == ["The error is on line four, where x is undefined."]
    assert [x.source.strip() for x in g.close()] == ["Try"]


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        ("See e.g. the second row. Then stop.", ["See e.g. the second row.", "Then stop."]),
        ("Dr. Smith wrote 3.14 here. Fine.", ["Dr. Smith wrote 3.14 here.", "Fine."]),
        ("Open https://example.com/a.b now. Ok.", ["Open https://example.com/a.b now.", "Ok."]),
        ("Is it right? Yes! Good.", ["Is it right?", "Yes!", "Good."]),
        ("Wait... I think so. Done.", ["Wait... I think so.", "Done."]),
    ],
)
def test_sentence_boundaries_handle_abbreviations_decimals_urls_and_punctuation(text, sentences):
    g = SpeechGate()
    out = list(g.feed(commit(text))) + list(g.close())
    assert [x.source.strip() for x in out] == sentences  # see the whitespace note above
    assert [x.text for x in out] == sentences, "prose, decimals and URLs are not rewritten"
    assert "".join(x.source for x in out) == text, "criterion 5: nothing lost, nothing invented"


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("Open ~/docs/report.pdf now.", "Open report dot pdf in your docs folder now."),
        ("Call fooBar_baz() next.", "Call foo bar baz next."),
        ("```python\nprint(1)\n```\nThat prints one.", "a code block. That prints one."),
        ("Four.<end_of_turn> garbage <|pad|>", "Four."),
        ("Line 4 has 3 errors on 2026-09-12.", "Line 4 has 3 errors on 2026-09-12."),  # identity
    ],
)
def test_normalisation_is_narrow_paths_identifiers_code_and_special_tokens_only(raw, spoken):
    assert normalise_for_tts(raw) == spoken


def test_text_after_eos_is_never_spoken():
    g = SpeechGate()
    out = list(g.feed(commit("Four.<end_of_turn> garbage <|pad|>"))) + list(g.close())
    assert [x.text for x in out] == ["Four."]


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.text(alphabet="abcdefghij .,!?0123456789", min_size=1, max_size=40),
        min_size=1,
        max_size=6,
    )
)
def test_nothing_lost_nothing_invented(chunks):
    g = SpeechGate()
    out = []
    for i, c in enumerate(chunks):
        out.extend(g.feed(commit(c, idx=i)))
    out.extend(g.close())
    joined = re.sub(r"\s+", " ", " ".join(x.text for x in out)).strip()
    # T-0224 re-pin: R-4. A generated sentence of 20+ characters can repeat (or be embedded in a
    # later one, R-3), and the repeat stop then refuses it and every later sentence: account for
    # them. The refusals must be repeat refusals (this alphabet has no "{", so the T-0052 guard
    # never fires) and form a suffix of the turn; the voiced prefix obeys the original law.
    reasons = list(getattr(g, "silenced_reasons", []))
    assert reasons == ["repeat"] * g.refused_count, reasons
    refused = [x for x in out if x.source.strip() and not x.text]
    first = out.index(refused[0]) if refused else len(out)
    assert all(not x.text for x in out[first:]), "R-3: everything after the first repeat is refused"
    kept = "".join(x.source for x in out[:first])
    expected = re.sub(r"\s+", " ", normalise_for_tts(kept)).strip()
    assert joined == expected
    assert "".join(x.source for x in out).replace(" ", "") == "".join(chunks).replace(" ", "")


# --- T-0004 extensions: the conservation law, the hold rule, the EOS trim -------------------------
#
# Every test below cites the acceptance criterion of docs/tasks/T-0004.md it comes from. Nothing
# here is derived from an implementation (there is none): the expected values are the *input*,
# computed by hand or by concatenation of the fixture chunks, never by re-running the gate's own
# logic inside the test.


def fed(gate: SpeechGate, *chunks: str) -> tuple[list, str]:
    """Feed chunks in order; return (sentences emitted so far, gate.pending)."""
    out: list = []
    for i, c in enumerate(chunks):
        out.extend(gate.feed(commit(c, idx=i)))
    return out, gate.pending


def sources(sentences) -> str:
    return "".join(s.source for s in sentences)


#: Text the gate must carry byte-for-byte. No ``<``/``>``/``|``: special tokens are criterion 6's
#: business and get their own tests below.
TEXT = st.text(alphabet="abcdefgij .,!?;:'\"()[]-/~_`\n0123456789", min_size=0, max_size=40)


@settings(max_examples=200, deadline=None)
@given(st.lists(TEXT, min_size=1, max_size=5))
def test_source_plus_pending_rebuilds_the_input_byte_for_byte(chunks):
    """Criterion 5, and T-0003 criterion 10's conservation check, seen from the gate's side.

    The turn machine computes ``"".join(sources) + remainder`` after *every* call and raises
    ``ValueError`` when it differs from what it handed in — a gate that trims, collapses or
    "tidies" whitespace in ``source`` crashes the turn at runtime. So: exact equality, including
    leading/trailing spaces, newlines and repeated blanks, at every intermediate step.
    """
    g = SpeechGate()
    seen: list = []
    consumed = ""
    for i, chunk in enumerate(chunks):
        seen.extend(g.feed(commit(chunk, idx=i)))
        consumed += chunk
        assert sources(seen) + g.pending == consumed
    seen.extend(g.close())
    assert sources(seen) == "".join(chunks), "close() must flush everything it still holds"
    assert g.pending == ""


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\n\n",
        "See e.g. the second row.",
        "Dr. Smith wrote 3.14 here.",
        "Wait... I think so",
        'He said "wait" and then left.',
        'He said "wait',
        "Consider the list (a, b",
        "Open ~/docs/report.pdf now.",
        "Call fooBar_baz() next.",
        "```python\nprint(1)\n```\nThat prints one.",
        "Two  spaces  and a trailing one ",
        "no terminal punctuation at all",
    ],
)
def test_awkward_inputs_are_conserved_byte_for_byte(text):
    """Criterion 5 on the concrete shapes that tempt a normaliser: abbreviations, decimals,
    ellipses, quotes, unclosed brackets, paths, code fences, odd whitespace, the empty string."""
    g = SpeechGate()
    out, held = fed(g, text)
    assert sources(out) + held == text
    out.extend(g.close())
    assert sources(out) == text
    assert g.pending == ""


def test_an_empty_commit_invents_nothing_and_loses_nothing():
    """Criterion 5. An empty canvas piece is a real wire condition; it must be a no-op."""
    g = SpeechGate()
    assert list(g.feed(commit(""))) == []
    assert g.pending == ""
    g.feed(commit("The error is on line", idx=0))
    held = g.pending
    assert list(g.feed(commit("", idx=1))) == [], "an empty commit completes no sentence"
    assert g.pending == held, "an empty commit must not disturb the held clause"
    assert sources(list(g.close())) == "The error is on line"


@pytest.mark.parametrize(
    "clause",
    [
        "The error is on line",
        "Consider the list (a, b",
        'He said "wait',
        "and then, after the second row,",
        "I think it is",
    ],
)
def test_an_unfinished_clause_is_held_until_close(clause):
    """Criterion 3, and CLAUDE.md "never voice a canvas that ends inside an unclosed thought":
    text with no completed sentence is held whole, then released — unchanged — by ``close()``."""
    g = SpeechGate()
    out, held = fed(g, clause)
    assert out == [], f"{clause!r} completes no sentence; nothing may be voiced yet"
    assert held == clause
    flushed = list(g.close())
    assert sources(flushed) == clause
    assert g.pending == ""


@settings(max_examples=150, deadline=None)
@given(st.text(alphabet="abcdefgij ,;:-'", min_size=1, max_size=40))
def test_text_without_a_sentence_end_is_never_voiced_before_final(clause):
    """Criterion 3 as a property: no ``.``/``?``/``!`` anywhere → nothing is complete."""
    g = SpeechGate()
    assert list(g.feed(commit(clause))) == []
    assert g.pending == clause
    assert sources(list(g.close())) == clause


def test_a_rejected_commit_leaves_no_trace_and_can_never_surface_later():
    """Criterion 1 + CLAUDE.md. Adversarial: a gate that buffers the text *before* validating the
    segment would leak an unclosed thought on the next ``close()`` — the exact failure the guard
    exists to prevent."""
    g = SpeechGate()
    g.feed(commit("The error is on line", idx=0))
    held = g.pending
    for bad in (
        commit("SECRET thinking", "thought", idx=1),
        commit("SECRET ui", "ui", idx=1),
        commit("SECRET unclosed.", idx=1, open_=True),
    ):
        with pytest.raises(GateInputError):
            g.feed(bad)
        assert g.pending == held, "a rejected commit must not enter the buffer"
    out = list(g.feed(commit(" four.", idx=2))) + list(g.close())
    spoken = sources(out) + "".join(s.text for s in out)
    assert "SECRET" not in spoken
    assert sources(out) == "The error is on line four."


@pytest.mark.parametrize(
    "text",
    [
        "Four.<end_of_turn> secret garbage <|pad|>",
        "Four. <|pad|> Five.",
        "Ready.<|think|> secret still thinking",
        "<end_of_turn> secret",
    ],
)
def test_special_tokens_and_post_eos_text_never_reach_the_tts_text(text):
    """Criterion 6: nothing that is a special token, and nothing after an EOS token, is spoken.
    Conservation (criterion 5) still holds on the ``source`` side for the very same input."""
    g = SpeechGate()
    out = list(g.feed(commit(text)))
    assert sources(out) + g.pending == text
    out.extend(g.close())
    spoken = "".join(s.text for s in out)
    assert "secret" not in spoken, "text after an EOS/special token is trimmed, never voiced"
    for token in ("<end_of_turn>", "<|pad|>", "<|think|>", "<|", "|>"):
        assert token not in spoken, f"{token!r} must never be handed to the TTS"


def test_close_twice_speaks_nothing_twice():
    """Criterion 5 ("nothing invented"): ``llm.done`` arriving twice must not re-voice the tail."""
    g = SpeechGate()
    g.feed(commit("A held clause"))
    assert sources(list(g.close())) == "A held clause"
    assert list(g.close()) == []
    assert g.pending == ""


# --- the gate inside the machine that polices it (T-0003 criterion 10) ---------------------------


def gate_segmenter(gate: SpeechGate):
    """Adapt ``SpeechGate`` to the injected segmenter contract of T-0003 criterion 10.

    ``segment(text, *, final) -> (sentences as (source, text) pairs, remainder)``. The machine
    hands back its whole unspoken tail — the gate's own ``pending`` plus the new commit — so only
    the suffix is fed. The adapter *never* repairs the gate's output: whatever it returns goes
    straight to the machine's conservation check.
    """

    def segment(text: str, *, final: bool):
        held = gate.pending
        assert text.startswith(held), (
            f"gate.pending {held!r} is not a prefix of the machine's unspoken tail {text!r}: "
            "the gate's held text must be exactly the remainder it last returned"
        )
        out = list(gate.feed(commit(text[len(held) :]))) if text[len(held) :] else []
        if final:
            out.extend(gate.close())
        return [(s.source, s.text) for s in out], gate.pending

    return segment


def test_the_gate_satisfies_the_turn_machines_segmenter_contract():
    """T-0003 criterion 10 + docs/02 §4 "never tts.speak with text the LLM has not committed".

    Driven through the real ``TurnMachine``, which raises ``ValueError`` on any call where
    ``"".join(sources) + remainder != text``. Two commits split mid-sentence, then ``llm.done``.
    """
    from local_astra.conductor.turn_machine import TurnMachine
    from local_astra.contracts import events as E

    machine = TurnMachine(segment=gate_segmenter(SpeechGate()))
    state = machine.initial()
    spoken: list = []
    events = [
        E.UserSpeechOnset(t_ms=1000, onset_ms=1000),
        E.SttWord(t_ms=1500, text="what", start_ms=1100, end_ms=1300),
        E.SttEndOfTurn(t_ms=2650, t_eot_ms=2600),
        E.SttFlushed(t_ms=2900, t_eot_ms=2600),
        E.Tick(t_ms=3100),
    ]
    for ev in events:
        state, _ = machine.step(state, ev)
    rid = state.active_generate
    assert rid is not None, "T-0003: a generate is live once the machine is THINKING"

    chunks = ["It is a stack trace. The error is on", " line four. Try again"]
    for i, chunk in enumerate(chunks):
        state, cmds = machine.step(
            state,
            E.LlmCommit(t_ms=3200 + i, request_id=rid, canvas_idx=i, segment="answer", text=chunk),
        )
        spoken.extend(c for c in cmds if c.type == "tts.speak")
    state, cmds = machine.step(state, E.LlmDone(t_ms=3400, request_id=rid))
    spoken.extend(c for c in cmds if c.type == "tts.speak")

    committed = "".join(chunks)
    assert state.committed_answer == committed
    assert "".join(c.source for c in spoken) == committed, "nothing lost, nothing invented"
    assert state.spoken_text == committed
    for c in spoken:
        assert c.source in committed, "every spoken source is committed text, verbatim"


# --- A6: a trim token LATCHES for the rest of the turn -------------------------------------------
#
# docs/tasks/T-0004.md ruling A6: `<end_of_turn>`, `<eos>`, `<|endoftext|>` and `<|think|>` all
# TRIM, and once one has been seen every *later* sentence of the same turn is silent — its
# `source` is still emitted (criterion 5, conservation) but its `text` is empty (CLAUDE.md: never
# voice the thought channel, never voice text the model did not commit as answer). The rows below
# each put a *completed* sentence after the token, which is what the earlier suite never did: a
# gate that resumed speaking after the token passed 46/46.


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        # EOS mid-canvas, then a complete sentence and a held one.
        (
            ("Four.<end_of_turn> junk. Five.",),
            [("Four.<end_of_turn> junk. ", "Four."), ("Five.", "")],
        ),
        # `<|think|>` is the thought-channel delimiter, not a tokenizer artifact.
        (
            ("Ready.<|think|> the secret is 42. Let me reconsider.",),
            [("Ready.<|think|> the secret is 42. ", "Ready."), ("Let me reconsider.", "")],
        ),
        # The token arrives in one commit, the later sentence in the next one.
        (
            ("Ready.<|think|> hidden reasoning. ", "Now the answer is five. And six."),
            [
                ("Ready.<|think|> hidden reasoning. ", "Ready."),
                ("Now the answer is five. ", ""),
                ("And six.", ""),
            ],
        ),
        # And with an EOS: the token is in the first commit, the sentences in the second.
        (
            ("Four.<eos> junk. ", "Five. Six."),
            [("Four.<eos> junk. ", "Four."), ("Five. ", ""), ("Six.", "")],
        ),
    ],
)
def test_a_trim_token_silences_every_later_sentence_of_the_turn(chunks, expected):
    g = SpeechGate()
    out, _ = fed(g, *chunks)
    out.extend(g.close())
    assert [(s.source, s.text) for s in out] == expected
    assert sources(out) == "".join(chunks), "criterion 5: every byte still travels in a source"
    assert g.pending == ""


def test_the_latch_holds_even_when_the_trimmed_sentence_was_emitted_by_close():
    """A6 adversarial: the token lands in the clause that only `close()` releases, and a *later*
    commit-free flush must still be silent. Here the whole turn after `<|think|>` is thinking."""
    g = SpeechGate()
    out, _ = fed(g, "Ready.<|think|> I should check", " the other branch. Actually no.")
    assert [(s.source, s.text) for s in out] == [
        ("Ready.<|think|> I should check the other branch. ", "Ready."),
    ]
    flushed = list(g.close())
    assert [(s.source, s.text) for s in flushed] == [("Actually no.", "")]


def test_nothing_after_a_trim_token_is_ever_handed_to_the_tts():
    """A6 as a blanket statement over the whole turn: the spoken side contains no byte that
    follows the first trim token, whichever sentence it ended up in."""
    g = SpeechGate()
    out, _ = fed(g, "Four.<end_of_turn> the SECRET plan. Step one. ", "Step two. Step three.")
    out.extend(g.close())
    spoken = " ".join(s.text for s in out)
    assert "SECRET" not in spoken
    for leaked in ("Step one", "Step two", "Step three"):
        assert leaked not in spoken, f"{leaked!r} follows an EOS token and must stay silent"
    assert spoken.split() == ["Four."], "only the pre-EOS prefix is voiced"


# --- A7: parentheses and code fences hold a terminator; double quotes do not ----------------------


def test_a_terminator_inside_an_open_parenthesis_is_not_a_sentence_end():
    """A7. `"Consider (a. b). Done."` fed in pieces: the `.` inside the parenthesis completes
    nothing, and when `)` finally arrives the whole parenthetical is ONE sentence."""
    g = SpeechGate()
    out, held = fed(g, "Consider (a. b")
    assert out == [], "a terminator inside an unclosed ( is not a boundary"
    assert held == "Consider (a. b"
    out2, held2 = fed(g, "). Done.")
    assert [(s.source, s.text) for s in out2] == [("Consider (a. b). ", "Consider (a. b).")]
    assert held2 == "Done."
    assert sources(out2) + held2 == "Consider (a. b). Done."


def test_a_terminator_inside_an_open_code_fence_is_not_a_sentence_end():
    """A7. Prose before the fence is spoken at once; the `.` inside the fence holds until the
    closing ```` ``` ```` arrives, and only then does the fenced sentence leave the gate."""
    g = SpeechGate()
    out, held = fed(g, "Ready. ```py\nx = 1. y = 2\n")
    assert [(s.source, s.text) for s in out] == [("Ready. ", "Ready.")]
    assert held == "```py\nx = 1. y = 2\n", "the unclosed fence is held whole"
    out2, held2 = fed(g, "```\nDone. ")
    assert [s.source for s in out2] == ["```py\nx = 1. y = 2\n```\nDone. "]
    assert [s.text for s in out2] == ["a code block. Done."]
    assert held2 == ""


@pytest.mark.parametrize(
    ("text", "first_source", "first_text", "rest"),
    [
        # A7, the new direction: an unbalanced `"` must NOT defer first speech to llm.done.
        ('He said "stop. Now fine.', 'He said "stop. ', 'He said "stop.', "Now fine."),
        # An inch mark is the same shape and just as common in model prose.
        (
            'The 5" drive failed. Replace it.',
            'The 5" drive failed. ',
            'The 5" drive failed.',
            "Replace it.",
        ),
    ],
)
def test_an_unbalanced_double_quote_does_not_hold_the_sentence(
    text, first_source, first_text, rest
):
    """A7: the double-quote clause of the balance rule is removed. Holding on a stray `"` pushes
    the first spoken sentence out to `llm.done` and blows the time-to-first-word budget
    (criterion 2, docs/01 §8), and no apostrophe-free heuristic can tell an opening quote from a
    stray one."""
    g = SpeechGate()
    out, held = fed(g, text)
    assert [(s.source, s.text) for s in out] == [(first_source, first_text)]
    assert held == rest


@pytest.mark.parametrize(
    ("text", "first_source", "first_text", "rest"),
    [
        ('He said "stop." Then he left.', 'He said "stop." ', 'He said "stop."', "Then he left."),
        ("Consider (a.) Then stop.", "Consider (a.) ", "Consider (a.)", "Then stop."),
        ("She left [sic.] Then it ended.", "She left [sic.] ", "She left [sic.]", "Then it ended."),
    ],
)
def test_a_closer_after_the_terminator_stays_attached_to_its_sentence(
    text, first_source, first_text, rest
):
    """A7: `)`, `"`, `]` following the `.` belong to the sentence that ends there — a gate that
    cuts at the `.` would either speak a dangling bracket next or hold the sentence forever."""
    g = SpeechGate()
    out, held = fed(g, text)
    assert [(s.source, s.text) for s in out] == [(first_source, first_text)]
    assert held == rest


#: A4/A7: the abbreviation list IS the rule, so every entry gets a row and the expected split is
#: written out by hand. Matching is case-insensitive, so the rows use natural prose casing.
ABBREVIATION_ROWS = [
    ("e.g.", "See e.g. the second row. Then stop.", ["See e.g. the second row. ", "Then stop."]),
    ("i.e.", "It is i.e. the first one. Then stop.", ["It is i.e. the first one. ", "Then stop."]),
    ("cf.", "See cf. the appendix. Then stop.", ["See cf. the appendix. ", "Then stop."]),
    ("dr.", "Ask Dr. Who first. Then stop.", ["Ask Dr. Who first. ", "Then stop."]),
    ("mr.", "Ask Mr. Smith first. Then stop.", ["Ask Mr. Smith first. ", "Then stop."]),
    ("mrs.", "Ask Mrs. Smith first. Then stop.", ["Ask Mrs. Smith first. ", "Then stop."]),
    ("ms.", "Ask Ms. Smith first. Then stop.", ["Ask Ms. Smith first. ", "Then stop."]),
    ("prof.", "Ask Prof. Smith first. Then stop.", ["Ask Prof. Smith first. ", "Then stop."]),
    ("sr.", "Ask Bob Sr. about it. Then stop.", ["Ask Bob Sr. about it. ", "Then stop."]),
    ("jr.", "Ask Bob Jr. about it. Then stop.", ["Ask Bob Jr. about it. ", "Then stop."]),
    ("st.", "Go to St. Giles first. Then stop.", ["Go to St. Giles first. ", "Then stop."]),
    ("vs.", "It is red vs. blue here. Then stop.", ["It is red vs. blue here. ", "Then stop."]),
    (
        "approx.",
        "It is approx. ten metres. Then stop.",
        ["It is approx. ten metres. ", "Then stop."],
    ),
    ("fig.", "See fig. two below. Then stop.", ["See fig. two below. ", "Then stop."]),
]


@pytest.mark.parametrize(
    ("text", "expected"),
    [(t, e) for _, t, e in ABBREVIATION_ROWS],
    ids=[a for a, _, _ in ABBREVIATION_ROWS],
)
def test_every_abbreviation_in_the_rule_keeps_its_sentence_together(text, expected):
    """A4 + A7: one row per entry, so emptying or thinning the list fails all of them, not two."""
    g = SpeechGate()
    out, _ = fed(g, text)
    out.extend(g.close())
    assert [s.source for s in out] == expected
    assert sources(out) == text


def test_the_abbreviation_rows_cover_the_whole_rule():
    """A4: the list in the implementation is the specification, not a sample — an entry added
    without a row (or a row without an entry) is a spec change and must be planned, not slipped
    in. This is the guard that keeps the parametrisation above exhaustive."""
    from local_astra.conductor.speech_gate import _ABBREVIATIONS

    assert {a for a, _, _ in ABBREVIATION_ROWS} == set(_ABBREVIATIONS)


# --- A8: the separator rides on the PRECEDING sentence, exactly -----------------------------------


def test_the_separator_after_a_terminator_belongs_to_the_sentence_before_it():
    """A8 (pinning A1). Exact equality, not `.strip()`: the space after `trace.` is part of the
    first sentence's `source`, so `pending` always starts at a word. A gate that let the
    separator ride on the *following* chunk still satisfies conservation but hands the turn
    machine a remainder that starts mid-whitespace, and the spoken/held boundary drifts."""
    text = "It is a stack trace. The error is on line"
    g = SpeechGate()
    out, held = fed(g, text)
    assert len(out) == 1
    assert out[0].source == "It is a stack trace. "
    assert out[0].text == "It is a stack trace."
    assert held == "The error is on line"
    assert out[0].source + held == text


@pytest.mark.parametrize(
    ("text", "expected_sources"),
    [
        ("One.  Two.  ", ["One.  ", "Two.  "]),  # a double space rides along whole
        ("One.\nTwo. ", ["One.\n", "Two. "]),  # so does a newline
        ("Is it? Yes! Fine. ", ["Is it? ", "Yes! ", "Fine. "]),
    ],
)
def test_whitespace_runs_after_a_terminator_ride_on_the_preceding_sentence(text, expected_sources):
    """A8: every kind of separator, not just a single space, lands on the left-hand sentence and
    the remainder starts at a word."""
    g = SpeechGate()
    out, held = fed(g, text)
    assert [s.source for s in out] == expected_sources
    assert held == ""
    assert sources(out) == text


# --- A9: `emitted` accumulates; `close()` is the only flush ---------------------------------------


def test_emitted_accumulates_every_sentence_of_the_turn_in_order():
    """A9: one gate per turn, so `emitted` is the turn's spoken transcript. It must contain every
    sentence ever returned, in order, including the one `close()` flushes. (A9 explicitly does not
    require a reset: a barge-in discards the gate object, it does not reuse it.)"""
    g = SpeechGate()
    assert g.emitted == []
    returned: list = []
    returned.extend(g.feed(commit("It is a stack trace. The error is on", idx=0)))
    assert g.emitted == returned
    returned.extend(g.feed(commit(" line four. Try", idx=1)))
    assert g.emitted == returned
    returned.extend(g.close())
    assert g.emitted == returned
    assert [s.source for s in g.emitted] == [
        "It is a stack trace. ",
        "The error is on line four. ",
        "Try",
    ]


def test_close_flushes_everything_and_a_second_close_adds_nothing_to_emitted():
    """A9 + A3: `close()` speaks, so it is called once per turn; calling it again must not
    re-voice the tail (that would be `nothing invented` violated on the wire)."""
    g = SpeechGate()
    g.feed(commit("A held clause"))
    flushed = list(g.close())
    assert [s.source for s in flushed] == ["A held clause"]
    assert g.emitted == flushed, "the flushed clause is part of the turn's transcript"
    assert list(g.close()) == []
    assert g.emitted == flushed, "a second close() adds nothing"
    assert g.pending == ""


# --- T-0022: an unclosed code fence is summarised, not spelled out --------------------------------
#
# docs/tasks/T-0022.md. R1: a ``` with no closing ``` is a fence that runs to the end of the input
# and gets the *same* marker as a closed one ("a code block."), so the user hears the same thing
# whether the model finished the block or ran into `max_tokens`. R2: `source` still carries the
# backticks and the code byte for byte; only `text` changes. R3: `close()` is the only place the
# case can arise and A7's hold rule is untouched. R4: the segment guard is not the fix.
#
# Expected strings below are written by hand from R1 — substitute the fence, drop nothing else,
# collapse whitespace, strip — never by running the gate. The `pending(task="T-0022")` markers
# were removed when /implement T-0022 started: criteria 1-3 and 9 are red on the pre-fix gate,
# criteria 4-8 are the negative half of the task — they already hold on the gate T-0004 merged and
# must still hold after the fence rule changes.

#: The reviewer's probe from the task's Goal section (criterion 1 / criterion 2).
C1_RAW = "Here it is: ```python\nfor x in y:\n    print(x)"
C1_SPOKEN = "Here it is: a code block."


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        # Criterion 1: an unterminated fence runs to the end of the string and gets the marker.
        pytest.param(
            C1_RAW,
            C1_SPOKEN,
            id="unclosed-fence-with-code",
        ),
        # Criterion 3: the degenerate case — an opening fence with nothing after it at all.
        pytest.param(
            "See below: ```",
            "See below: a code block.",
            id="bare-opening-fence",
        ),
        # Criterion 4, the adversarial row: prose BETWEEN and AFTER two closed fences must
        # survive. A greedy or open-arm-first pattern (`r"```.*(?:```)?"`, `r"```.*|```.*?```"`)
        # swallows "Then", the second fence and "ends it." and still passes every other row here.
        # It is green before the fix too: this is exactly the behaviour that must not change.
        pytest.param(
            "```py\nx = 1\n```\nThen ```sh\nls\n``` ends it.",
            "a code block. Then a code block. ends it.",
            id="two-closed-fences-with-prose",
        ),
        # The same shape with the *second* fence left open: one marker each, prose in between.
        pytest.param(
            "```py\nx = 1\n```\nThen ```sh\nls",
            "a code block. Then a code block.",
            id="closed-fence-then-unclosed-fence",
        ),
        # Order matters (Implementation notes): the fence rule runs before the path and identifier
        # rules, so code inside an unclosed fence is gone before they can spell it out.
        pytest.param(
            "See ```py\n~/docs/report.pdf fooBar_baz()",
            "See a code block.",
            id="unclosed-fence-hides-paths-and-identifiers",
        ),
    ],
)
def test_an_unclosed_fence_is_summarised_like_a_closed_one(raw, spoken):
    """Criteria 1, 3 and 4 on the pure function (Tests-to-write 1).

    Called directly, not through `close()`: `normalise_for_tts` is public and documented as
    "fenced code -> a code block" (R1), so a `close()`-only patch must fail here.
    """
    assert normalise_for_tts(raw) == spoken


def test_a_canvas_that_ends_inside_a_code_fence_is_summarised_not_spelled_out():
    """Criterion 2: the reviewer's probe, end to end. R2 pins `source` byte for byte."""
    g = SpeechGate()
    out, held = fed(g, C1_RAW)
    assert out == [], "nothing outside the fence terminates: no sentence is complete yet"
    assert held == C1_RAW
    flushed = list(g.close())
    assert [s.source for s in flushed] == [C1_RAW], "R2: source is conserved, unchanged"
    assert [s.text for s in flushed] == [C1_SPOKEN]
    assert g.pending == "", "A3: nothing is held forever"


def test_the_open_fence_tail_leaves_close_as_one_summarised_sentence():
    """R3: at `final=True` the balance check still keeps the `.` inside the open fence from
    splitting it, so the tail arrives as ONE source and is spoken as one marker — not as
    "a code block." followed by a second sentence made of the rest of the code."""
    g = SpeechGate()
    out, _held = fed(g, "Here. ```py\nx = 1. y = 2")
    assert [s.text for s in out] == ["Here."]
    flushed = list(g.close())
    assert [(s.source, s.text) for s in flushed] == [("```py\nx = 1. y = 2", "a code block.")]
    assert g.pending == ""


def test_an_open_fence_still_holds_the_terminator_until_close():
    """Criterion 5 (A7 regression, active): a terminator inside an open fence is still not a
    sentence end, so the prose before the fence is spoken at once and nothing with a backtick or
    a code byte in it leaves the gate before `close()`. A "fix" that relaxed
    `chunk.count("```") % 2` to make the unclosed case work would voice the code sentence by
    sentence mid-turn — the opposite of the goal (Implementation notes)."""
    text = "Here. ```py\nx = 1. y = 2"
    g = SpeechGate()
    out, held = fed(g, text)
    assert [(s.source, s.text) for s in out] == [("Here. ", "Here.")]
    assert held == "```py\nx = 1. y = 2", "the open fence is held whole"
    for s in out:
        assert "`" not in s.text
        assert "x = 1" not in s.text and "y = 2" not in s.text
    assert sources(out) + held == text


@pytest.mark.parametrize(
    ("segment", "open_"),
    [("thought", False), ("ui", False), ("answer", True)],
    ids=["thought-segment", "ui-segment", "thought-open"],
)
def test_an_unclosed_fence_in_a_non_answer_segment_is_still_rejected(segment, open_):
    """Criterion 7 / R4 (active): the unclosed-fence fix must NOT be implemented by loosening the
    `feed` guard so the normaliser can see the payload. `ui` and `thought` text never reaches the
    normaliser at all — an unclosed fence inside an A2UI payload is the ui_gate's business
    (T-0010) — and a rejected commit still leaves no trace in the buffer."""
    g = SpeechGate()
    g.feed(commit("Here it is: ", idx=0))
    held = g.pending
    with pytest.raises(GateInputError):
        g.feed(commit("```py\nx", segment, idx=1, open_=open_))
    assert g.pending == held, "a rejected commit must not enter the buffer"
    out = list(g.close())
    assert sources(out) == "Here it is: ", "the rejected fence never surfaces on close()"
    assert "`" not in "".join(s.text for s in out)


def test_a_trimmed_token_inside_an_open_fence_never_leaks():
    """Criterion 8 (active): trimming happens *before* the fence rule (Implementation notes), so
    an open fence after `<|think|>` cannot reappear through the marker path, and nothing after the
    token — code, backticks or the token itself — is handed to the TTS."""
    text = "Ready.<|think|> ```py\nsecret"
    g = SpeechGate()
    out, _ = fed(g, text)
    out.extend(g.close())
    assert [s.text for s in out] == ["Ready."]
    for s in out:
        for leaked in ("secret", "<|", "|>", "`"):
            assert leaked not in s.text, f"{leaked!r} must never reach the TTS"
    assert sources(out) == text, "R2: the trimmed bytes still travel inside source"


def test_a_trim_token_before_an_open_fence_still_latches_for_the_rest_of_the_turn():
    """Criterion 8, second half (A6, active): every *later* sentence of the turn stays silent even
    though the fence closes afterwards and a new sentence completes."""
    g = SpeechGate()
    chunks = ("Ready.<|think|> ```py\nsecret\n```\nStill hidden. ", "Next one. ")
    out, _ = fed(g, *chunks)
    out.extend(g.close())
    assert [s.text for s in out] == ["Ready.", ""]
    spoken = " ".join(s.text for s in out)
    for leaked in ("secret", "Still hidden", "Next one", "`"):
        assert leaked not in spoken
    assert sources(out) == "".join(chunks)


#: Every input used by criteria 1-5 and 8, for the conservation law of criterion 6.
FENCE_INPUTS = [
    C1_RAW,
    "See below: ```",
    "```py\nx = 1\n```\nThen ```sh\nls\n``` ends it.",
    "```py\nx = 1\n```\nThen ```sh\nls",
    "See ```py\n~/docs/report.pdf fooBar_baz()",
    "Here. ```py\nx = 1. y = 2",
    "Ready.<|think|> ```py\nsecret",
    "```",
    "``````",
    "a ``` b ``` c ``` d",
]


@pytest.mark.parametrize("text", FENCE_INPUTS, ids=[str(i) for i in range(len(FENCE_INPUTS))])
def test_a_fence_input_is_conserved_byte_for_byte_however_it_is_chunked(text):
    """Criterion 6 / R2 (active). The turn machine (T-0003 criterion 10) recomputes
    `"".join(sources) + remainder` after every call and raises `ValueError` when it differs, so a
    fence fix that stripped, collapsed or "tidied" `source` crashes the turn at runtime. Checked
    at every split point of the input, including splits that cut a ``` in half."""
    for cut in range(len(text) + 1):
        g = SpeechGate()
        seen = list(g.feed(commit(text[:cut], idx=0)))
        assert sources(seen) + g.pending == text[:cut], f"after chunk 1, cut={cut}"
        seen.extend(g.feed(commit(text[cut:], idx=1)))
        assert sources(seen) + g.pending == text, f"after chunk 2, cut={cut}"
        seen.extend(g.close())
        assert sources(seen) == text, f"close() must flush everything, cut={cut}"
        assert g.pending == ""


def drive_fence_through_the_machine(gate, chunks):
    """Feed `chunks` to a real `TurnMachine` wired to `gate`; return (state, tts.speak commands).

    Same adapter as `test_the_gate_satisfies_the_turn_machines_segmenter_contract`; the machine
    raises `ValueError` on any call where `"".join(sources) + remainder != text`.
    """
    from local_astra.conductor.turn_machine import TurnMachine
    from local_astra.contracts import events as E

    machine = TurnMachine(segment=gate_segmenter(gate))
    state = machine.initial()
    for ev in [
        E.UserSpeechOnset(t_ms=1000, onset_ms=1000),
        E.SttWord(t_ms=1500, text="what", start_ms=1100, end_ms=1300),
        E.SttEndOfTurn(t_ms=2650, t_eot_ms=2600),
        E.SttFlushed(t_ms=2900, t_eot_ms=2600),
        E.Tick(t_ms=3100),
    ]:
        state, _ = machine.step(state, ev)
    rid = state.active_generate
    assert rid is not None, "T-0003: a generate is live once the machine is THINKING"
    spoken: list = []
    for i, chunk in enumerate(chunks):
        state, cmds = machine.step(
            state,
            E.LlmCommit(t_ms=3200 + i, request_id=rid, canvas_idx=i, segment="answer", text=chunk),
        )
        spoken.extend(c for c in cmds if c.type == "tts.speak")
    state, cmds = machine.step(state, E.LlmDone(t_ms=3400, request_id=rid))
    spoken.extend(c for c in cmds if c.type == "tts.speak")
    return state, spoken


#: The probe split across two canvases, so the fence opens in one commit and never closes.
MACHINE_CHUNKS = ["Here it is: ```python\n", "for x in y:\n    print(x)"]


def test_an_unclosed_fence_passes_the_turn_machines_conservation_check():
    """Criterion 6 (active): the §4 invariant on `TtsSpeak.source` still holds for a canvas that
    ends mid-fence — every spoken source is committed text, verbatim."""
    state, spoken = drive_fence_through_the_machine(SpeechGate(), MACHINE_CHUNKS)
    committed = "".join(MACHINE_CHUNKS)
    assert state.committed_answer == committed
    assert "".join(c.source for c in spoken) == committed, "nothing lost, nothing invented"
    assert state.spoken_text == committed
    for c in spoken:
        assert c.source in committed, "every spoken source is committed text, verbatim"


def test_the_machine_never_hands_pocket_tts_a_backtick_from_an_unclosed_fence():
    """Criterion 2 seen from the wire: the `tts.speak` commands the machine emits for a canvas
    that ends mid-fence carry the summary, not the code."""
    _, spoken = drive_fence_through_the_machine(SpeechGate(), MACHINE_CHUNKS)
    assert [c.text for c in spoken] == [C1_SPOKEN]


@settings(max_examples=200, deadline=None)
@given(st.lists(st.text(alphabet="abc `\n. ", min_size=0, max_size=24), min_size=1, max_size=4))
@example(["Here ```py x"])
@example(["a ``` b", "``` c"])
@example(["```a``````"])
def test_no_emitted_text_ever_contains_a_fence_marker(chunks):
    """Criterion 9, the catch-all for R1: over an alphabet that includes the backtick, no
    `Sentence.text` from any feed/close sequence ever contains three consecutive backticks —
    whether the fence closed, stayed open, or was split across commits. `source` is still exact.
    """
    g = SpeechGate()
    out: list = []
    for i, chunk in enumerate(chunks):
        out.extend(g.feed(commit(chunk, idx=i)))
    out.extend(g.close())
    assert sources(out) == "".join(chunks)
    assert all("```" not in s.text for s in out), [s.text for s in out]
