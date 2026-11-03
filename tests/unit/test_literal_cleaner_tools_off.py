"""T-0124 criterion 8 at the cleaner: ``session._LiteralCleaner``, chained after T-0121's ``_ThoughtCleaner``,
releases U(raw) = L(T(raw)), chunk by chunk (tools off).

``_LiteralCleaner`` does not exist at the base (8b3f6d0); it is read with ``getattr(session, "_LiteralCleaner")``
inside each row, behind a named assertion, so a row is red on that assertion and never on an import or attribute
error. The chain is the one the task's owned files describe for the session: every voiced commit goes through
``_ThoughtCleaner.feed`` and then ``_LiteralCleaner.feed``; at the end of the stream the thought stage's ``flush`` is
fed into the literal stage, then the literal stage's ``flush``.

The oracle ``l_rule`` is the task's definition of L written out from its text, never a call into the cleaner: repeat
until a round changes nothing: cut at the earliest occurrence of any ``speech_gate._TRIM_TOKENS`` literal, then
delete every ``speech_gate._DROP_TOKENS`` literal (until none is left, as the gate does). T(raw) is T-0121's oracle
``t_raw`` (imported). ``dropped_chars`` is ``len(T(raw)) - len(U(raw))`` and ``cut`` whether any round cut. The
premise row ``test_oracle_is_the_gate_rule`` checks the oracle against the gate's own ``_cut_and_drop`` (the task
says L is the gate's rule exactly), so the oracle and the gate cannot drift apart.

The oracle premises hold at the base and carry no marker; every row that needs the cleaner is pending.

Readings (in the test-author's report): ``_LiteralCleaner()`` takes no arguments (as ``_ThoughtCleaner()``);
``raw`` is the concatenation of what the literal stage was fed (T(raw), by the chain); after a cut the release
timing of a tail held before it is not pinned (the task says "may be released at once").

Regime: pure CPU, Hypothesis bounded (``max_examples`` 200, ``database=None`` so a mutation run never replays a saved
example, ``report_multiple_bugs=False``), strings of <= 14 tokens, <= 6 cuts; seed printed on failure.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from local_astra.conductor import session, speech_gate
from local_astra.conductor.speech_gate import normalise_for_tts
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS
from tests.unit.test_answer_cleaner_thoughts import CLOSE, OPEN, split
from tests.unit.test_thought_cleaner_tools_off import t_raw

BOUNDED = settings(
    max_examples=200,
    deadline=None,
    database=None,
    report_multiple_bugs=False,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)

TRIM = speech_gate._TRIM_TOKENS
DROP = speech_gate._DROP_TOKENS
#: "No literal" (Tests to write): none of SPECIAL_TOKENS and the gate's trim table
LITERALS = tuple(dict.fromkeys((*SPECIAL_TOKENS, *TRIM)))


# --- the oracle: L and U, from the task text ---------------------------------------------------------------


def l_rule(text: str) -> tuple[str, bool]:
    """L(s) and whether any round cut."""
    cut = False
    while True:
        before = text
        found = [i for i in (text.find(t) for t in TRIM) if i != -1]
        if found:
            text, cut = text[: min(found)], True
        while any(t in text for t in DROP):
            for t in DROP:
                text = text.replace(t, "")
        if text == before:
            return text, cut


def u_rule(raw: str) -> tuple[str, str, int, bool]:
    """(T(raw), U(raw), dropped_chars, cut)."""
    t = t_raw(raw)[0]
    u, cut = l_rule(t)
    return t, u, len(t) - len(u), cut


# --- generators ---------------------------------------------------------------------------------------------

#: every trim literal, drop literals, and the fragments the task names (``<end_of``/``_turn>``, ``<|endof``/
#: ``text|>``, ``<e``/``os>``), splices (``<pa``/``d>``, ``<|tool``/``_call>``, ``<|chan``/``nel>``), the channel
#: markers (so T(raw) matters), bare ``<``/``|``/``>`` and plain words (``f`` and ``x`` are hold characters only
#: because of ``<end_of_turn>``/``<|endoftext|>``)
TOKENS = (
    *TRIM,
    "<pad>",
    '<|"|>',
    "<bos>",
    "<|tool>",
    "<tool_call|>",
    "<|turn>",
    OPEN,
    CLOSE,
    "<end_of",
    "_turn>",
    "<|endof",
    "text|>",
    "<e",
    "os>",
    "<pa",
    "d>",
    "<|tool",
    "_call>",
    "<|chan",
    "nel>",
    "<b",
    "ack>",
    "<",
    "|",
    ">",
    "a",
    "f",
    "x",
    "now",
    " ",
    ". ",
)
raws = st.lists(st.sampled_from(TOKENS), min_size=1, max_size=14).map("".join)


@st.composite
def raw_and_split(draw: Any) -> tuple[str, list[str]]:
    raw = draw(raws)
    return raw, draw(split(raw))


def _stages() -> tuple[Any, Any]:
    thought = session._ThoughtCleaner
    cls = getattr(session, "_LiteralCleaner", None)
    assert cls is not None, (
        "T-0124 (owned files): session._LiteralCleaner (feed, flush, raw, dropped_chars, cut)"
    )
    return thought(), cls()


def chain(commits: list[str]) -> tuple[list[str], str, Any]:
    """Each commit through both stages, then the thought flush into the literal stage, then the literal flush."""
    th, lit = _stages()
    feeds = [lit.feed(th.feed(c)) for c in commits]
    end = lit.feed(th.flush())
    end += lit.flush()
    return feeds, end, lit


def _literal() -> Any:
    return _stages()[1]


#: whole and split forms of the task's rows, so the property rows impose them whatever is generated
RULED = (
    ("Press <|tool_call> now. ", ["Press <|tool", "_call> now. "]),
    ("Hi <end_of_turn> bye. ", ["Hi <end_of", "_turn> bye. "]),
    ("Hi <|endoftext|> bye. ", ["Hi <|endof", "text|> bye. "]),
    ("Sure, <|chan<eos>nel>thought SECRET<channel|>ok. ",
     ["Sure, <|chan", "<eos>nel>thought SEC", "RET<channel|>ok. "]),
    ("First. <eos>Second. Third. ", ["First. <eos>Second. ", "Third. "]),
    ("Here <|channel>a<eos>b<channel|> ok. ", ["Here <|chan", "nel>a<e", "os>b<channel|> ok. "]),
    ("Press <|tool<pad>_call> now. ", ["Press <|tool<pa", "d>_call> now. "]),
    ("Hi <pad>there. ", ["Hi <p", "ad>there. "]),
    ("Press <back> now. ", ["Press <b", "ack> now. "]),
)  # fmt: skip


def _ruled(test: Any) -> Any:
    for case in RULED:
        test = example(case)(test)
    return test


# =====================================================================================================
# Oracle premises (not pending: they check the test, and hold at the base)
# =====================================================================================================

#: commits -> (U(raw), dropped_chars, cut), by hand from the task's criteria 1, 2, 5 and 6
HAND: dict[tuple[str, ...], tuple[str, int, bool]] = {
    **{
        (f"Press {lit} now. ",): ("Press ", len(f"{lit} now. "), True)
        for lit in ("<eos>", "<turn|>", "<|tool_call>", "<|tool_response>", "<|think|>",
                    "<end_of_turn>", "<|endoftext|>")
    },
    ("Hi <pad>there. ",): ("Hi there. ", 5, False),
    ('Say <|"|>yes<|"|>. ',): ("Say yes. ", 10, False),
    # criterion 2: T(raw) deletes only the stray close (47 characters), L cuts at <eos>: 47 - 12 = 35
    ("Sure, <|chan<eos>nel>thought SECRET-THOUGHT<channel|>ok. ",): ("Sure, <|chan", 35, True),
    ("Sure, <|chan", "<eos>nel>thought SEC", "RET-THOUGHT<channel|>ok. "): ("Sure, <|chan", 35, True),
    # criterion 5: len("<eos>Second. Third. ") = 5 + 8 + 7
    ("First. <eos>Second. ", "Third. "): ("First. ", 20, True),
    # criterion 6: T(raw) removes the block, <eos> with it; L drops nothing
    ("Here <|channel>a<eos>b<channel|> ok. ",): ("Here  ok. ", 0, False),
    # criterion 6, spliced: len("<|tool<pad>_call> now. ") = 6 + 5 + 6 + 6
    ("Press <|tool<pad>_call> now. ",): ("Press ", 23, True),
    ("Press <|tool", "_call> now. "): ("Press ", 18, True),
    ("Press <back> now. ",): ("Press <back> now. ", 0, False),
}  # fmt: skip


def test_oracle_matches_the_task_examples():
    """``u_rule`` reproduces every U(raw), ``dropped_chars`` and ``cut`` the task writes out (by hand above)."""
    bad = {
        c: (u_rule("".join(c))[1:], want)
        for c, want in HAND.items()
        if u_rule("".join(c))[1:] != want
    }
    assert not bad, f"oracle vs hand: {bad}"


@BOUNDED
@given(raws)
def test_oracle_is_the_gate_rule(raw):
    """Implementation notes: L is the gate's rule exactly. The hand-written ``l_rule`` equals the gate's
    ``_cut_and_drop`` (text and whether it trimmed) on any generated string, so a drift in either is visible."""
    assert l_rule(raw) == speech_gate._cut_and_drop(raw), raw


# =====================================================================================================
# Criterion 8
# =====================================================================================================


@BOUNDED
@given(raw_and_split())
@_ruled
def test_release_is_chunk_independent(case):
    """Criterion 8, first item (mutations: L before T, clean per commit, one pass with no fixpoint, hold no tail):
    for any split of raw into commits, every release plus the flush joins to U(raw); ``raw`` is T(raw) (what the
    stage was fed); ``dropped_chars`` and ``cut`` are the oracle's."""
    raw, commits = case
    t, u, chars, cut = u_rule(raw)
    feeds, end, lit = chain(commits)
    assert "".join(feeds) + end == u, (commits, feeds, end, u)
    assert lit.raw == t, (lit.raw, t)
    assert (lit.dropped_chars, lit.cut) == (chars, cut), (
        f"T-0124 criterion 7 at the cleaner: {commits}: (dropped_chars, cut) {(lit.dropped_chars, lit.cut)}, "
        f"U(raw) gives {(chars, cut)}"
    )


@BOUNDED
@given(raw_and_split())
@_ruled
def test_released_text_only_grows(case):
    """Criterion 8, second item (I12; mutations: hold no tail, keep releasing after a cut): after every commit the
    text released so far is a prefix of U(raw)."""
    raw, commits = case
    u = u_rule(raw)[1]
    th, lit = _stages()
    released = ""
    for i, commit in enumerate(commits):
        released += lit.feed(th.feed(commit))
        assert u.startswith(released), (
            f"T-0124 criterion 8: after commit #{i} the release {released!r} is not a prefix of U(raw) {u!r}; "
            f"commits {commits}"
        )
    released += lit.feed(th.flush())
    released += lit.flush()
    assert released == u, (released, u)


@BOUNDED
@given(raw_and_split())
@_ruled
def test_no_release_holds_a_literal(case):
    """Criterion 8, third item (mutations: hold alphabet from ``SPECIAL_TOKENS`` only, clean only in ``flush``): no
    release, read when it is returned, and no cumulative release holds a literal (a released prefix that later
    completes one shows up in the cumulative release)."""
    _, commits = case
    th, lit = _stages()
    released = ""
    for i, commit in enumerate(commits):
        out = lit.feed(th.feed(commit))
        released += out
        bad = [x for x in LITERALS if x in out or x in released]
        assert not bad, (
            f"T-0124 criterion 8: commit #{i} released {out!r} (so far {released!r}): {bad}"
        )
    released += lit.feed(th.flush()) + lit.flush()
    bad = [x for x in LITERALS if x in released]
    assert not bad, (
        f"T-0124 criterion 8: the joined release {released!r} holds {bad}; commits {commits}"
    )


@BOUNDED
@given(raw_and_split())
@_ruled
def test_voice_is_unchanged(case):
    """Criterion 8, last item (mutation: delete trim literals instead of cutting): the voice of what the chain
    releases equals the voice of T(raw), the gate's input at the base: ``normalise_for_tts(U) ==
    normalise_for_tts(T)``."""
    raw, commits = case
    t = t_raw(raw)[0]
    feeds, end, _ = chain(commits)
    assert normalise_for_tts("".join(feeds) + end) == normalise_for_tts(t), (commits, feeds, end)


no_literal_raws = st.lists(
    st.sampled_from(tuple(x for x in TOKENS if x not in LITERALS)), min_size=1, max_size=14
).map("".join)


@BOUNDED
@given(st.data())
def test_no_literal_input_is_identity(data):
    """Negative (criterion 10 at the cleaner; mutations: delete or cut a fragment, insert anything): fed text with
    no literal (none spliced either: only literals are deleted), the literal stage releases it byte for byte, every
    cumulative release is a prefix of it, ``dropped_chars`` is 0 and ``cut`` false."""
    raw = data.draw(no_literal_raws)
    assume(not any(x in raw for x in LITERALS))
    commits = data.draw(split(raw))
    lit = _literal()
    released = ""
    for commit in commits:
        released += lit.feed(commit)
        assert raw.startswith(released), (commits, released)
    released += lit.flush()
    assert released == raw, f"T-0124: no literal, so the release is raw {raw!r}; got {released!r}"
    assert (lit.raw, lit.dropped_chars, lit.cut) == (raw, 0, False)


@pytest.mark.parametrize("commits", list(HAND), ids=[repr(c)[:60] for c in HAND])
def test_hand_rows(commits):
    """Criteria 1, 2, 5 and 6 by hand, through the chain: the joined release, ``dropped_chars`` and ``cut``."""
    want = HAND[commits]
    feeds, end, lit = chain(list(commits))
    assert ("".join(feeds) + end, lit.dropped_chars, lit.cut) == want, (feeds, end, want)


#: commits fed to the literal stage directly -> (each ``feed`` output, ``flush``), by hand from the hold rule: a tail
#: is held from the first ``<`` after the last character that is ``>`` or outside the alphabet of every literal L
#: cuts or deletes (``f`` and ``x`` included, from ``<end_of_turn>`` and ``<|endoftext|>``)
HOLDS: dict[str, tuple[tuple[str, ...], tuple[list[str], str]]] = {
    # released at once (too wide; mutation: hold every ``<``-led tail)
    "comparison": (("x < y",), (["x < y"], "")),
    "closed_bar": (("a <|>",), (["a <|>"], "")),
    "digit_barrier": (("Done <x9",), (["Done <x9"], "")),
    # held (absent; mutation: hold no tail), released at the end when nothing completes them
    "tool_call_prefix": (("Press <|tool",), (["Press "], "<|tool")),
    "tag_prefix": (("Press <b",), (["Press "], "<b")),
    # mutation: hold alphabet from SPECIAL_TOKENS only (no ``f``)
    "end_of_prefix": (("Hi <end_of",), (["Hi "], "<end_of")),
    # mutation: hold alphabet from SPECIAL_TOKENS only (no ``f``, no ``x``)
    "endof_prefix": (("Hi <|endof",), (["Hi "], "<|endof")),
    "bare_lt": (("<",), ([""], "<")),
    # mutation: release the held tail only at flush even after more text arrives
    "held_then_released": (("Press <b", "ack> now. "), (["Press ", "<back> now. "], "")),
    # mutation: keep releasing after a cut
    "after_cut": (("First. <eos>Second. ", "Third. "), (["First. ", ""], "")),
    # completed into a literal: cut there, nothing after
    "completed": (("Press <|tool", "_call> now. "), (["Press ", ""], "")),
}


@pytest.mark.parametrize("case", list(HOLDS))
def test_hold_rows(case):
    """Implementation notes, the hold (guard too wide and guard absent, ``.claude/rules/tests.md``): ``x < y``,
    ``a <|>`` and ``Done <x9`` are released at once; ``Press <|tool``, ``Press <b``, ``Hi <end_of``, ``Hi <|endof``
    and a bare ``<`` are held until a later commit or the end of the stream."""
    commits, (feeds_want, flush_want) = HOLDS[case]
    lit = _literal()
    feeds = [lit.feed(c) for c in commits]
    flushed = lit.flush()
    assert (feeds, flushed) == (feeds_want, flush_want), (
        f"T-0124 hold ({case}): feeds {feeds}, flush {flushed!r}; expected {feeds_want}, {flush_want!r}"
    )
