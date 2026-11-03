"""T-0088a criterion 3: the speech gate refuses every Gemma 4 special token (human decision Q2: strip).

The 24 literals are written out in ``tests/unit/_detok_support.py`` (checked against the fixture by
``test_special_tokens.py``), so these rows depend on no file and on no implementation table.

Rows:
- **trimmed**, one row per literal: ``<turn|>``, ``<eos>``, ``<|tool_response>``, ``<|tool_call>`` (end of what may be
  spoken) and ``<|channel>`` (a thought) trim like A2: nothing from the literal on is voiced, in this sentence or a
  later one, and the trimmed bytes still travel in ``source``;
- **dropped**, one row per literal of the 18 the task lists: deleted in place, the rest of the turn still spoken;
- the T-0087 click-01 host decode (stray ``<channel|>`` before and after the answer, then the call);
- a literal split across two commits; a splice that only exists after a deletion;
- **never voiced** + **conservation** (A1, A20): a Hypothesis property over the 24 literals, prose and fragments;
- **tool segment**: ``Commit(segment="tool")`` raises ``GateInputError`` (a pin: main's ``!= "answer"`` check);
- *negative*: a Hypothesis differential against a frozen copy of main's gate on text with no special literal;
- **adversarial**: main's ``<\\|[^|>]*\\|>`` regex alone fails ``I will.<|tool_call>call:x{}<tool_call|>``; a gate that
  is too wide fails ``a < b``, ``x |> y`` and ``<|`` in prose.

Ambiguity stated, not guessed: criterion 3 names 5 trimmed and 18 dropped literals, 23 of the 24. The 24th,
``<|think|>``, is trimmed by main's A3 rule and ``test_speech_gate.py`` (unedited, criterion 4) pins that, so it is
pinned here as trimmed too. Ruling D2 settles the drop: a dropped literal is deleted with nothing, never a space;
``test_d2_a_dropped_literal_is_deleted_with_nothing`` pins that exactly, and the per-literal drop rows compare text
with whitespace removed so that they test the drop and nothing else.

Rows added after review (rulings at eeee0ab): exact splice-trim rows with the latch; D2 at the gate; D6 widened
(``</turn|>``, ``</channel|>``, ``</tool|>``, ``</image|>``, ``</audio|>`` are dropped, never trimmed); the exact D6
relation in the differential; a mixed prose-and-literal row.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from local_astra.conductor.speech_gate import GateInputError, SpeechGate
from local_astra.contracts.llm import Commit
from local_astra.toolcall.special_tokens import drop_special_literals
from tests.unit import _frozen_speech_gate_main as frozen
from tests.unit._detok_support import SPECIAL_LITERALS

#: Criterion 3 "Trimmed", plus ``<|think|>`` (main's A3, see the module docstring).
TRIMMED = ("<turn|>", "<eos>", "<|tool_response>", "<|tool_call>", "<|channel>")
TRIMMED_ALL = (*TRIMMED, "<|think|>")
#: Criterion 3 "Dropped", the task's list verbatim (18).
DROPPED = (
    "<channel|>",
    "<tool_call|>",
    '<|"|>',
    "<|turn>",
    "<|tool>",
    "<tool|>",
    "<tool_response|>",
    "<|image>",
    "<|image|>",
    "<image|>",
    "<|audio>",
    "<|audio|>",
    "<audio|>",
    "<|video|>",
    "<pad>",
    "<bos>",
    "<unk>",
    "<mask>",
)
#: Main's other trims (EOS spellings of other tokenizers); text after them is never voiced either.
MAIN_TRIMS = ("<end_of_turn>", "<|endoftext|>")


def commit(text: str, idx: int = 0, segment: str = "answer") -> Commit:
    return Commit(request_id="r", canvas_idx=idx, segment=segment, text=text)  # type: ignore[arg-type]


def run(chunks: Sequence[str], gate: Any = None) -> tuple[Any, list[Any]]:
    g = gate if gate is not None else SpeechGate()
    out: list[Any] = []
    for i, c in enumerate(chunks):
        out.extend(g.feed(commit(c, i)))
    out.extend(g.close())
    return g, out


def spoken(out: Sequence[Any]) -> str:
    return " ".join(s.text for s in out if s.text)


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def literals_in(text: str) -> list[str]:
    return [lit for lit in SPECIAL_LITERALS if lit in text]


def test_the_trimmed_and_dropped_lists_cover_the_24_literals() -> None:
    """Self-check of this module's tables: every literal is in exactly one class."""
    assert sorted((*TRIMMED_ALL, *DROPPED)) == sorted(SPECIAL_LITERALS)
    assert len(set(TRIMMED_ALL) & set(DROPPED)) == 0


# --------------------------------------------------------------------------- trimmed


@pytest.mark.parametrize("lit", TRIMMED_ALL)
def test_a_trim_literal_ends_what_may_be_spoken(lit: str) -> None:
    text = f"I will click it.{lit} secret words here. More secret text."
    _, out = run([text])
    assert "".join(s.source for s in out) == text, "A1: the trimmed bytes still travel in source"
    assert spoken(out) == "I will click it.", f"{lit!r} must trim like A2: {[s.text for s in out]}"
    assert all(s.text == "" for s in out[1:]), "A20: every later sentence travels with text=''"


@pytest.mark.parametrize("lit", TRIMMED_ALL)
def test_a_trim_literal_mid_sentence_trims_the_rest_of_that_sentence(lit: str) -> None:
    text = f"I will click{lit} the secret button. Then the secret."
    _, out = run([text])
    assert spoken(out) == "I will click"
    assert "secret" not in spoken(out)
    assert "".join(s.source for s in out) == text


@pytest.mark.parametrize("lit", TRIMMED)
def test_a_trim_literal_in_a_later_commit_trims_the_rest_of_the_turn(lit: str) -> None:
    chunks = ["I will click it. ", f"Now.{lit}secret. More secret.", " Still secret."]
    _, out = run(chunks)
    assert spoken(out) == "I will click it. Now."
    assert "".join(s.source for s in out) == "".join(chunks)


# --------------------------------------------------------------------------- dropped


@pytest.mark.parametrize("lit", DROPPED)
def test_a_dropped_literal_is_deleted_in_place_and_the_turn_goes_on(lit: str) -> None:
    text = f"I will{lit} click it. Then{lit} done.{lit}"
    _, out = run([text])
    assert "".join(s.source for s in out) == text
    assert not any(literals_in(s.text) for s in out), [s.text for s in out]
    assert squash(spoken(out)) == "Iwillclickit.Thendone.", (
        f"{lit!r} must be deleted in place, not trim the turn: {[s.text for s in out]}"
    )


@pytest.mark.parametrize("lit", DROPPED)
def test_a_dropped_literal_alone_in_a_sentence_voices_nothing_of_it(lit: str) -> None:
    text = f"First. {lit} Second."
    _, out = run([text])
    assert squash(spoken(out)) == "First.Second."
    assert "".join(s.source for s in out) == text


# --------------------------------------------------------------------------- recorded and split shapes


def test_the_t0087_click_01_host_decode_voices_only_the_answer() -> None:
    """docs/cd/toolcall-live-probe.md, click-01 arm (d), host decode after the thought."""
    text = (
        "<channel|>I will click the Sign in button.<channel|><|tool_call>call:click_at{x:488,y:518}"
        "<tool_call|><|tool_response>"
    )
    _, out = run([text])
    assert spoken(out) == "I will click the Sign in button."
    assert "".join(s.source for s in out) == text


def test_a_literal_split_across_two_commits_is_still_trimmed() -> None:
    chunks = ["I will click it.<|tool", "_call>call:click_at{x:1,y:2}<tool_call|>"]
    _, out = run(chunks)
    assert spoken(out) == "I will click it."
    assert "".join(s.source for s in out) == "".join(chunks)


@pytest.mark.parametrize(
    "text",
    [
        "Hi <|chan<channel|>nel> there. Done.",
        "Hi <tu<pad>rn|> there. Done.",
        'Hi <|tool_<|"|>call> there. Done.',
        "Hi <<eos>|tool_ca<pad>ll> there.",
    ],
)
def test_a_literal_spliced_by_a_deletion_is_never_voiced(text: str) -> None:
    _, out = run([text])
    assert not any(literals_in(s.text) for s in out), [s.text for s in out]
    assert "".join(s.source for s in out) == text


# --------------------------------------------------------------------------- never voiced, conservation

_PROSE = ["I", "will", "click", "it", "a", "b", "x", "y", " ", " ", ".", "!", ",", "\n"]
_FRAGMENTS = [
    "<|",
    "|>",
    "<|to",
    "ol_call>",
    "<",
    ">",
    "|",
    "chan",
    "nel>",
    "<|chan",
    "_call>",
    "tool",
]
_ALPHABET = [*SPECIAL_LITERALS, *MAIN_TRIMS, *_PROSE, *_FRAGMENTS]
_TRIM_SOURCES = (*TRIMMED_ALL, *MAIN_TRIMS)


@settings(max_examples=400, deadline=None)
@given(
    st.lists(st.lists(st.sampled_from(_ALPHABET), min_size=0, max_size=14), min_size=1, max_size=5)
)
def test_no_special_literal_is_ever_voiced_and_every_byte_is_conserved(
    parts: list[list[str]],
) -> None:
    """Hypothesis, 400 examples, up to 5 commits of up to 14 tokens from the 24 literals, main's two other EOS
    spellings, prose and fragments (``<|``, ``|>``, ``<|to``, ``ol_call>``…), so splices and split literals occur.

    Checked after every ``feed`` and after ``close()``: no ``text`` holds a literal; A1 (sources + pending rebuild
    every byte fed so far); A20 (once a source carries a trim literal, every later sentence has ``text == ""``)."""
    chunks = ["".join(p) for p in parts]
    g = SpeechGate()
    out: list[Any] = []
    fed = ""
    for i, c in enumerate(chunks):
        out.extend(g.feed(commit(c, i)))
        fed += c
        assert "".join(s.source for s in out) + g.pending == fed, "A1 after feed"
    out.extend(g.close())
    assert "".join(s.source for s in out) == fed and g.pending == "", "A1 after close"
    for s in out:
        assert not literals_in(s.text), (s.source, s.text)
    first_trim = next(
        (k for k, s in enumerate(out) if any(t in s.source for t in _TRIM_SOURCES)), None
    )
    if first_trim is not None:
        assert all(s.text == "" for s in out[first_trim + 1 :]), (
            "A20: nothing after a trim is voiced"
        )


# --------------------------------------------------------------------------- tool segment


def test_a_tool_commit_is_refused_and_leaves_no_trace() -> None:
    """Criterion 3 "Tool segment": a pin of main's ``segment != "answer"`` check. ``model_construct`` because the
    ``tool`` segment arrives in the contract with T-0088, not here (``contracts/**`` is unchanged by T-0088a)."""
    g = SpeechGate()
    g.feed(commit("A held clause"))
    tool = Commit.model_construct(
        request_id="r",
        canvas_idx=1,
        segment="tool",  # type: ignore[arg-type]
        text="<|tool_call>call:click_at{x:1,y:2}<tool_call|>",
        thought_open=False,
        fence_close=None,
        fence_continued=False,
    )
    with pytest.raises(GateInputError):
        g.feed(tool)
    assert g.pending == "A held clause"
    assert [s.text for s in g.close()] == ["A held clause"]


# --------------------------------------------------------------------------- negative: unchanged on special-free text

_FREE = [
    "I", "will", "say", "e.g.", "Dr.", "3.14", "~/docs/a.txt", "fooBar", "snake_case", "foo()", "```", "(", ")",
    "{", "}", '"component"', '"x":', "tool", "call", "_", "channel", "turn", "<", ">", "|", "<|", "|>",
    "<|pad|>", "<end_of_turn>", "<|endoftext|>", ".", "!", "?", ",", " ", " ", " ", "\n",
    "/eos", "/pad", "/bos", "/unk", "/mask",  # ruling D6: main's path rewrite turns </eos> into <eos>
]  # fmt: skip


def _pairs(out: Sequence[Any]) -> list[tuple[str, str]]:
    return [(s.source, s.text) for s in out]


#: T-0232 R-8: the repeat stop drops *every* trailing ASCII punctuation mark and "…" (T-0224 criterion 1's
#: "trailing punctuation dropped", as the gate measures it), not only ``.!?,;:"')]}``; a narrower strip kept
#: "…case !|>" apart from "…case" and let a repeat through the filter below.
_TRAILING_PUNCT = re.compile(r"[\s!-/:-@\[-`{-~…]+$")


def _repeat_key(s: str) -> str:
    return _TRAILING_PUNCT.sub("", " ".join(s.casefold().split()))


def _no_repeat_on_main(chunks: list[str]) -> bool:
    """T-0224 re-pin: R-4. True when no sentence main voices for ``chunks`` repeats an earlier voiced
    one under the T-0224 repeat stop (R-3: an equal normalised form or a word-bounded containment
    of an earlier voiced key of >= 20 characters). Deliberately wider than the rule, so the
    20-character boundary and the source-vs-text choice never decide: any earlier key of 10+
    characters found anywhere inside a later one, on the source or on the text, excludes the draw."""
    old = frozen.SpeechGate()
    out = [s for i, c in enumerate(chunks) for s in old.feed(commit(c, i))] + list(old.close())
    seen: list[str] = []
    for s in out:
        if not s.text:
            continue
        keys = [_repeat_key(s.source), _repeat_key(s.text)]
        if any(len(k) >= 10 and k in key for k in seen for key in keys):
            return False
        seen.extend(keys)
    return True


_R8_PATH = ["channel", "say", "3.14", "|>", "~/docs/a.txt", " ", "|", "channel", "snake_case"]


@settings(max_examples=400, deadline=None)
@given(st.lists(st.lists(st.sampled_from(_FREE), max_size=16), min_size=1, max_size=5))
# T-0232 R-8: the last sentence repeats "...|channelsnake case !|>?" (a repeat once "!|>?" is dropped); the old
# key kept "!|>" and the filter let it through, so the row went red.
@example(
    [
        ["!", " ", " ", "/eos"],
        [],
        [" ", "!", "|>", "?", " ", " ", *_R8_PATH],
        [" ", "!", "Dr.", "?", " ", " ", *_R8_PATH],
        [" ", "!", "|>", "?", " ", " ", *_R8_PATH],
    ]
)
def test_text_without_a_special_literal_gives_mains_output(parts: list[list[str]]) -> None:
    """Hypothesis differential, 400 examples, against ``_frozen_speech_gate_main`` (main 575e1d4), call by call.

    Sources and ``pending`` equal main's on every call. Each sentence's ``text`` equals main's, except — ruling D6,
    "never voiced" outranks parity — a sentence where *main* would voice a special literal (its path rewrite turns
    ``</eos>`` into ``<eos>``): there the gate's text must hold no literal."""
    chunks = ["".join(p) for p in parts]
    assume(not literals_in("".join(chunks)))
    assume(_no_repeat_on_main(chunks))  # T-0224 re-pin: R-4
    new, old = SpeechGate(), frozen.SpeechGate()
    calls = [
        (new.feed(commit(c, i)), old.feed(commit(c, i)), new.pending, old.pending)
        for i, c in enumerate(chunks)
    ]
    calls.append((new.close(), old.close(), new.pending, old.pending))
    for got, want, got_pending, want_pending in calls:
        assert [s.source for s in got] == [s.source for s in want]
        assert got_pending == want_pending
        for g, w in zip(got, want, strict=True):
            if literals_in(w.text):
                assert not literals_in(g.text), (g.source, g.text)
                # D6, exact: the gate is main's text with the literals deleted and whitespace re-collapsed
                assert g.text == " ".join(drop_special_literals(w.text).split()), (
                    g.source,
                    g.text,
                    w.text,
                )
            else:
                assert g.text == w.text, (g.source, g.text, w.text)
    assert new.refused_count == old.refused_count


#: Ruling D6: special-free prose that main's path rewrite turns into a special literal (``</eos>`` -> ``<eos>``).
D6_NAMES = ("eos", "pad", "bos", "unk", "mask")


@pytest.mark.parametrize("name", D6_NAMES)
@pytest.mark.parametrize(
    "shape", ["Close the tag </{n}> now. Then go on.", "Type </{n}>. Then go on."]
)
def test_a_literal_made_by_mains_path_rewrite_is_never_voiced(name: str, shape: str) -> None:
    """Ruling D6, one row per input: the committed text has no special literal, main voices one, the gate must not;
    the literal is dropped (not trimmed: the next sentence is still spoken) and every byte is conserved."""
    text = shape.format(n=name)
    lit = f"<{name}>"
    assert not literals_in(text), "self-check: the input is special-free"
    _, old = run([text], frozen.SpeechGate())
    assert any(lit in s.text for s in old), (
        f"self-check: main voices {lit!r} for {text!r}: {[s.text for s in old]}"
    )
    _, out = run([text])
    assert "".join(s.source for s in out) == text
    assert not any(literals_in(s.text) for s in out), [s.text for s in out]
    expected = squash(shape.format(n="").replace("</>", ""))
    assert squash(spoken(out)) == expected, [s.text for s in out]


@pytest.mark.parametrize(
    "text",
    [
        "Four.<end_of_turn> garbage <|pad|>",
        "Four. <|pad|> Five.",
        "Done.<|endoftext|> hidden text.",
        "Use a <|pad|> token.",
    ],
)
def test_mains_trims_and_drops_survive(text: str) -> None:
    _, new = run([text])
    _, old = run([text], frozen.SpeechGate())
    assert _pairs(new) == _pairs(old)


# --------------------------------------------------------------------------- adversarial


def test_mains_narrow_regex_is_not_enough() -> None:
    """Adversarial (main's ``<\\|[^|>]*\\|>``): it matches neither ``<|tool_call>`` nor ``<tool_call|>``, so main
    voices the call. The gate must speak only the sentence before it."""
    text = "I will.<|tool_call>call:x{}<tool_call|>"
    _, out = run([text])
    assert [s.text for s in out] == ["I will."]
    assert "".join(s.source for s in out) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Is a < b? Yes, and c > d.", ["Is a < b?", "Yes, and c > d."]),
        ("if a<b and c>d then stop.", ["if a<b and c>d then stop."]),
        ("x |> y pipes x into y.", ["x |> y pipes x into y."]),
        ("Type <| to go back.", ["Type <| to go back."]),
        ("A <channel and channel> pair.", ["A <channel and channel> pair."]),
        ("Write <turn> or <channel> here.", ["Write <turn> or <channel> here."]),
    ],
)
def test_a_gate_that_is_too_wide_silences_prose(text: str, expected: list[str]) -> None:
    """Adversarial (too wide): only exact special literals are refused; angle brackets and pipes in prose are
    spoken verbatim (and equal main's output)."""
    _, out = run([text])
    assert [s.text for s in out] == expected
    _, old = run([text], frozen.SpeechGate())
    assert [s.text for s in old] == expected, "self-check: main speaks these verbatim"


# --------------------------------------------------------------------------- rows after review (eeee0ab)


@pytest.mark.parametrize(
    ("text", "first_text"),
    [
        (
            "Hi <|chan<channel|>nel> there. Done.",
            "Hi",
        ),  # dropping <channel|> splices <|channel>: a trim
        ("Hi <tu<pad>rn|> there. Done.", "Hi"),  # dropping <pad> splices <turn|>: a trim
        ('Hi <|tool_<|"|>call> there. Done.', "Hi"),  # dropping <|"|> splices <|tool_call>: a trim
        (
            "Hi <<eos>|tool_ca<pad>ll> there. Done.",
            "Hi <",
        ),  # the raw <eos> is the earliest trim: cut there
    ],
)
def test_a_trim_literal_spliced_by_a_drop_trims_and_latches(text: str, first_text: str) -> None:
    """Ruling (eeee0ab) "splice trims": exact sentences. A trim literal that only exists after a drop trims like a
    raw one, and the latch silences every later sentence. Kills ``latch_raw_only`` (latch on the raw source only)
    and ``splice_drop_not_trim`` (cut at the raw trim index, then drop everything)."""
    _, out = run([text])
    first_source = text[: text.index("Done.")]
    assert [(s.source, s.text) for s in out] == [(first_source, first_text), ("Done.", "")]


@pytest.mark.parametrize("lit", ["<pad>", "<channel|>", '<|"|>', "<mask>"])
def test_d2_a_dropped_literal_is_deleted_with_nothing(lit: str) -> None:
    """Ruling D2 at the gate, unsquashed: ``Hel<pad>lo world.`` speaks ``Hello world.``, not ``Hel lo world.``."""
    text = f"Hel{lit}lo world."
    _, out = run([text])
    assert [(s.source, s.text) for s in out] == [(text, "Hello world.")]


@pytest.mark.parametrize("name", ["turn|", "channel|", "tool|", "image|", "audio|"])
def test_d6_a_trim_or_drop_literal_made_by_the_path_rewrite_is_dropped_never_trimmed(
    name: str,
) -> None:
    """D6 widened: ``</turn|>`` becomes ``<turn|>`` only after main's path rewrite; it is dropped even though
    ``<turn|>`` is a trim literal, so the next sentence is still spoken."""
    text = f"Close the tag </{name}> now. Then go on."
    _, old = run([text], frozen.SpeechGate())
    assert any(f"<{name}>" in s.text for s in old), (
        f"self-check: main voices <{name}>: {[s.text for s in old]}"
    )
    _, out = run([text])
    assert [(s.source, s.text) for s in out] == [
        (f"Close the tag </{name}> now. ", "Close the tag now."),
        ("Then go on.", "Then go on."),
    ]


def test_prose_brackets_survive_beside_a_real_literal() -> None:
    """Mixed row: ``<turn>`` is prose (not a literal) and stays; the real ``<pad>`` is deleted with nothing."""
    text = "Write <turn> here.<pad> Ok."
    _, out = run([text])
    assert [(s.source, s.text) for s in out] == [(text, "Write <turn> here. Ok.")]
