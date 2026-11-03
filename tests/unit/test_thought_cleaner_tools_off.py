"""T-0121 criterion 5 at the cleaner: ``session._ThoughtCleaner`` releases T(raw), chunk by chunk (tools off).

``_ThoughtCleaner`` does not exist at the base (791dc54); it is read with ``getattr(session, "_ThoughtCleaner")``
inside each row, behind a named assertion, so a row is red on that assertion and never on an import or
attribute error.

The oracle ``t_raw`` is the task's definition of T(raw) written out from its text, never a call into the cleaner
or the router: repeat until a round changes nothing: (1) remove each closed block ``<|channel>...<channel|>`` (the
first close after an open closes it: T-0117's non-greedy, left-to-right ``_BLOCK`` regex, imported) until stable,
then cut at a ``<|channel>`` that is left (nothing after it closes it); that is ``answer_channel`` applied until stable;
(2) delete every stray ``<channel|>``. Nothing is inserted and no other literal is deleted. ``chars`` is what step (1)
removed, markers included, which is ``len(raw) - len(T(raw))`` less the stray closes step (2) deleted; ``unclosed``
is whether any step (1) cut.

The generators are T-0117's (``tests/unit/test_answer_cleaner_thoughts``: tokens heavy in ``<``, ``|``, ``>``, marker
fragments, other literals and their fragments, the sentinel and plain words; random splits of up to six cuts),
imported, not copied.

The oracle premises hold at the base and carry no marker; every row that needs the cleaner is red until T-0121 lands.

Regime: pure CPU, Hypothesis bounded (``max_examples`` 200, ``database=None`` so a mutation run never replays a
saved example, ``report_multiple_bugs=False``), strings of <= 14 tokens, <= 6 cuts; seed printed on failure.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from local_astra.conductor import session
from tests.unit.test_answer_cleaner_thoughts import (
    _BLOCK,
    CLOSE,
    OPEN,
    SENTINEL,
    TOKENS,
    raw_and_split,
    split,
    thoughtful,
)

BOUNDED = settings(
    max_examples=200,
    deadline=None,
    database=None,
    report_multiple_bugs=False,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)


# --- the oracle: T(raw), from the task text ------------------------------------------------------------------


def _channel_until_stable(text: str) -> tuple[str, bool]:
    """Step (1): closed blocks removed until none is left, then the text from a remaining open cut."""
    while True:
        nxt = _BLOCK.sub("", text)
        if nxt == text:
            break
        text = nxt
    if OPEN in text:
        return text.split(OPEN, 1)[0], True
    return text, False


def t_raw(raw: str) -> tuple[str, int, bool]:
    """T(raw), the characters step (1) removed over every round, and whether any round cut."""
    text, strays, cut = raw, 0, False
    while True:
        channel, cut_now = _channel_until_stable(text)
        cut = cut or cut_now
        n = channel.count(CLOSE)
        nxt = channel.replace(CLOSE, "")
        if nxt == text:
            return text, len(raw) - len(text) - strays * len(CLOSE), cut
        strays += n
        text = nxt


#: raw -> (T(raw), chars, unclosed), by hand from the task's own examples
HAND: dict[str, tuple[str, int, bool]] = {
    "Here <|channel>thought SECRET-THOUGHT<channel|> it is. ": ("Here  it is. ", 42, False),
    "Okay <|channel>thought SECRET-THOUGHT": ("Okay ", 32, True),
    "Okay <|channel>xy<channel|> fine. ": ("Okay  fine. ", 22, False),
    "Done <channel|>here. ": ("Done here. ", 0, False),
    "Wait <|channel> there. ": ("Wait ", 18, True),
    "Wait <|chan": ("Wait <|chan", 0, False),
    "I thought so, and I still do. ": ("I thought so, and I still do. ", 0, False),
    "Press <|tool_call> now. ": ("Press <|tool_call> now. ", 0, False),
    "x < y and y > z. ": ("x < y and y > z. ", 0, False),
    # the stray close's deletion splices an open nothing closes (T-0117 F1): 10 + 14 cut, the stray not counted
    "Here <|chan<channel|>nel>SECRET-THOUGHT": ("Here ", 24, True),
    # the inner block's removal splices a block the next close closes (T-0117 F3): 21 + 26
    "Hi <|chan<|channel>x<channel|>nel>SECRET<channel|> ok. ": ("Hi  ok. ", 47, False),
    # step (2) deletes *every* stray close in a round (N9: one per round would leave the second as a block's
    # close): round 1 has no ``<|channel>``, so (1) changes nothing and (2) deletes both closes, giving
    # ``<|channel>AB``; round 2 cuts at that open (10 + 2 = 12 removed, unclosed); round 3 changes nothing.
    # With one close deleted per round, round 2 would remove the block ``<|channel>A<channel|>`` and keep ``B``.
    "<|chan<channel|>nel>A<channel|>B": ("", 12, True),
    # no other literal is deleted, so ``<eos>`` splices nothing: only the stray close goes (H-T121-5 territory)
    "Sure, <|chan<eos>nel>thought SECRET<channel|>ok. ": (
        "Sure, <|chan<eos>nel>thought SECRETok. ",
        0,
        False,
    ),
}


def _cleaner() -> Any:
    cls = getattr(session, "_ThoughtCleaner", None)
    assert cls is not None, (
        "T-0121 (owned files): session._ThoughtCleaner (feed, flush, raw, thought_chars, unclosed)"
    )
    return cls()


def _run(cleaner: Any, commits: list[str]) -> tuple[list[str], str]:
    feeds = [cleaner.feed(c) for c in commits]
    return feeds, cleaner.flush()


#: whole and split forms of the task's splices, so the property rows impose them whatever is generated
RULED = (
    ("Here <|chan<channel|>nel>SECRET", ["Here <|chan<chann", "el|>nel>SECRET"]),
    ("Hi <|chan<|channel>SECRET<channel|>nel>t<channel|> ok. ",
     ["Hi <|chan<|chann", "el>SECRET<channel|>ne", "l>t<channel|> ok. "]),
    ("Here <|channel>thought SECRET<channel|> it is. ", ["Here <|chan", "nel>thought SECRET<chan", "nel|> it is. "]),
    ("Okay <|channel>x SECRET", ["Okay <|channel>x", " SECRET"]),
    ("Okay <|channel>x SECRET<channel|> fine. ", ["Okay <|channel>x", " SECRET<channel|> fine. "]),
    ("Done <channel|>here. ", ["Done <chan", "nel|>here. "]),
    # N9: every stray close in a round is deleted, not one (T(raw) is ``""``, by hand in HAND)
    ("<|chan<channel|>nel>A<channel|>B", ["<|chan<chan", "nel|>nel>A<channel|>B"]),
)  # fmt: skip


def _ruled(test: Any) -> Any:
    for case in RULED:
        test = example(case)(test)
    return test


# =====================================================================================================
# Oracle premises (not pending: they check the test, and hold at the base)
# =====================================================================================================


def test_oracle_matches_the_task_examples():
    """``t_raw`` reproduces every T(raw), ``chars`` and ``unclosed`` the task writes out (hand-computed above)."""
    bad = {raw: (t_raw(raw), want) for raw, want in HAND.items() if t_raw(raw) != want}
    assert not bad, f"oracle vs hand: {bad}"


@BOUNDED
@given(thoughtful())
def test_sentinel_generator_premise(case):
    """Every sentinel the ``thoughtful`` generator writes is inside thought text, so T(raw) never holds it (a premise
    of ``test_no_release_holds_thought_text``)."""
    raw, _ = case
    assert SENTINEL not in t_raw(raw)[0], raw


# =====================================================================================================
# The cleaner's identity (owned files: must not be or construct ``_AnswerCleaner``)
# =====================================================================================================


def test_thought_cleaner_is_not_the_answer_cleaner(monkeypatch):
    """Owned files (mutation: route tools-off through ``_AnswerCleaner``): ``_ThoughtCleaner`` is not
    ``_AnswerCleaner`` nor a subclass, and builds none (``_AnswerCleaner`` replaced by a class whose construction
    fails by name); it cleans criterion 1's text to ``Here  it is. `` and exposes ``raw``, ``thought_chars`` and
    ``unclosed``."""
    cls = getattr(session, "_ThoughtCleaner", None)
    assert cls is not None, "T-0121 (owned files): session._ThoughtCleaner"
    assert cls is not session._AnswerCleaner and not issubclass(cls, session._AnswerCleaner), (
        cls.__mro__
    )

    class Forbidden(session._AnswerCleaner):  # type: ignore[misc, name-defined]
        """Delegates every class attribute to the real cleaner; only construction is refused, by name."""

        def __init__(self, *a: Any, **k: Any) -> None:
            raise AssertionError("T-0121: the tools-off cleaner built an _AnswerCleaner")

    monkeypatch.setattr(session, "_AnswerCleaner", Forbidden)
    cleaner = cls()
    raw = "Here <|channel>thought SECRET-THOUGHT<channel|> it is. Press <|tool_call> now. "
    feeds, flushed = _run(cleaner, [raw[:9], raw[9:30], raw[30:]])
    assert "".join(feeds) + flushed == "Here  it is. Press <|tool_call> now. ", (feeds, flushed)
    assert (cleaner.raw, cleaner.thought_chars, cleaner.unclosed) == (raw, 42, False)


# =====================================================================================================
# Criterion 5
# =====================================================================================================


@BOUNDED
@given(raw_and_split())
@_ruled
def test_release_is_chunk_independent(case):
    """Criterion 5, first item (mutations: strip per commit, one pass with no fixpoint, hold no tail): for any split
    of raw into commits, every ``feed`` output plus ``flush`` joins to T(raw), and ``raw`` is the concatenation."""
    raw, commits = case
    want = t_raw(raw)[0]
    cleaner = _cleaner()
    feeds, flushed = _run(cleaner, commits)
    assert "".join(feeds) + flushed == want, (commits, feeds, flushed, want)
    assert cleaner.raw == raw, (cleaner.raw, raw)


@BOUNDED
@given(raw_and_split())
@_ruled
def test_released_text_only_grows(case):
    """Criterion 5, second item (I12, tools-off form): after every ``feed`` the text released so far is a prefix of
    T(raw) (mutation: hold no tail)."""
    raw, commits = case
    want = t_raw(raw)[0]
    cleaner = _cleaner()
    released = ""
    for i, commit in enumerate(commits):
        released += cleaner.feed(commit)
        assert want.startswith(released), (
            f"T-0121 criterion 5: after feed #{i} the release {released!r} is not a prefix of T(raw) {want!r}; "
            f"commits {commits}"
        )
    released += cleaner.flush()
    assert released == want, (released, want)


@BOUNDED
@given(thoughtful())
def test_no_release_holds_thought_text(case):
    """Criterion 5, third item (mutations: clean only in ``flush``, release text after an open): every thought
    carries the ``SECRET`` sentinel (whose letters are in no marker), and no ``feed`` output, read when it is
    returned, nor ``flush`` holds it."""
    _, commits = case
    cleaner = _cleaner()
    for i, commit in enumerate(commits):
        out = cleaner.feed(commit)
        assert SENTINEL not in out, (
            f"T-0121 criterion 5: feed #{i} released {out!r}; commits {commits}"
        )
    out = cleaner.flush()
    assert SENTINEL not in out, f"T-0121 criterion 5: flush released {out!r}; commits {commits}"


@BOUNDED
@given(raw_and_split())
@_ruled
def test_thought_chars_and_unclosed(case):
    """Criterion 7 at the cleaner (mutation: counting stray closes): after ``flush``, ``thought_chars`` is what
    ``answer_channel`` removed over every round (the oracle's ``chars``) and ``unclosed`` whether a round cut."""
    raw, commits = case
    _, chars, unclosed = t_raw(raw)
    cleaner = _cleaner()
    _run(cleaner, commits)
    assert (cleaner.thought_chars, cleaner.unclosed) == (chars, unclosed), (
        f"T-0121 criterion 7: {commits}: (thought_chars, unclosed) {(cleaner.thought_chars, cleaner.unclosed)}, "
        f"T(raw) gives {(chars, unclosed)}"
    )


no_marker_raws = st.lists(st.sampled_from(tuple(t for t in TOKENS if t not in (OPEN, CLOSE))),
                          min_size=1, max_size=14).map("".join)  # fmt: skip


@BOUNDED
@given(st.data())
def test_other_literals_untouched(data):
    """Negative (mutations: run ``drop_special_literals``, delete other literals): with no channel marker in raw
    (none is ever spliced, since only markers are deleted), the joined release equals raw for any split, and
    every cumulative release is a prefix of raw; ``<|tool_call>``, ``<eos>`` and their fragments survive byte for
    byte."""
    raw = data.draw(no_marker_raws)
    assume(OPEN not in raw and CLOSE not in raw)
    commits = data.draw(split(raw))
    cleaner = _cleaner()
    released = ""
    for commit in commits:
        released += cleaner.feed(commit)
        assert raw.startswith(released), (commits, released)
    released += cleaner.flush()
    assert released == raw, (
        f"T-0121: no marker, so the release is raw {raw!r}; got {released!r} from {commits}"
    )


#: commits -> (each ``feed`` output, ``flush``), by hand from the hold rule: a tail is held only from a ``<`` whose
#: following characters are all from ``<|channel>`` other than ``>``; any other character is a barrier.
HOLDS = {
    # too wide (mutation: hold every ``<``-led tail): the literal, a comparison, a tag; released at once
    "tool_call_prefix": (("Press <|tool",), (["Press <|tool"], "")),
    "tool_call_split": (("Press <|tool", "_call> now. "), (["Press <|tool", "_call> now. "], "")),
    "comparison": (("x < y and y > z. ",), (["x < y and y > z. "], "")),
    "tag_prefix": (("Press <b",), (["Press <b"], "")),
    "eos": (("a <eos> b",), (["a <eos> b"], "")),
    "barrier_after_hold": (("Here <|", "x"), (["Here ", "<|x"], "")),
    # absent (mutation: hold no tail): a tail that can still become a marker is held, and released at the end
    "open_prefix": (("Here <|chan",), (["Here "], "<|chan")),
    "two_prefixes": (("Here <|chan<chann",), (["Here "], "<|chan<chann")),
    # N1 (too wide: ``>`` treated as a hold character): ``>`` ends every marker, so a tail ending in one cannot
    # grow into a marker and is released at once
    "closed_tag": (("x <a>",), (["x <a>"], "")),
    "closed_bar": (("a <|>",), (["a <|>"], "")),
    "bare_lt": (("Here <",), (["Here "], "<")),
    "stray_split": (("Done <chan", "nel|>here. "), (["Done ", "here. "], "")),
}


@pytest.mark.parametrize("case", list(HOLDS))
def test_hold_is_no_wider_than_the_marker_alphabet(case):
    """Implementation notes, the hold (guard too wide and guard absent, ``.claude/rules/tests.md``): ``Press <|tool``,
    ``x < y`` and ``Press <b`` are not held; ``<|chan`` is, until a later commit or the end of the stream."""
    commits, (feeds_want, flush_want) = HOLDS[case]
    feeds, flushed = _run(_cleaner(), list(commits))
    assert (feeds, flushed) == (feeds_want, flush_want), (
        f"T-0121 hold ({case}): feeds {feeds}, flush {flushed!r}; expected {feeds_want}, {flush_want!r}"
    )
