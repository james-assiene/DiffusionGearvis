"""T-0126 criterion 9 at the cleaner: with tools on, ``session._AnswerCleaner`` releases D(raw), chunk by chunk.

The oracle ``d_raw`` is the task's rule written out from its text, never a call into the cleaner: C(raw) is T-0117's
oracle ``c_raw`` (imported from ``test_answer_cleaner_thoughts``: criterion 1 of docs/tasks/T-0117.md as ruled), and
D(raw) is C(raw) cut at the earliest occurrence of any member of E (``speech_gate._TRIM_TOKENS`` minus
``SPECIAL_TOKENS``: ``<end_of_turn>`` and ``<|endoftext|>``), here an independent ``find``/``min`` over E. The premise
row ``test_oracle_matches_the_task_examples`` checks the oracle against the strings the task writes out by hand.

``_AnswerCleaner`` is read with ``getattr(session, "_AnswerCleaner")`` inside each row (the task's instruction). It
is fed only the ``thought_open=False`` answer texts (I10, I15), so the unit rows feed plain strings.

Pending rows (``pending(task="T-0126")``) are red at the base 4a25a85: every property row that involves an E literal
carries ``@example`` cases from the task, so the red does not depend on what Hypothesis draws. Unmarked rows hold at
the base: the oracle premise, ``test_without_eot_equals_c_raw`` (with no E literal in C(raw), D(raw) = C(raw): the
guard-too-wide half) and the ``HOLDS`` rows released at once (the hold must stay no wider than T-0124's rule).

Readings (in the test-author's report): "each cumulative release is a prefix of D(raw)" and "no release holds a
prefix that D(raw) later cuts" are checked for the split drawn, and the second also against every continuation in
``COMPLETIONS`` (the rest of an E literal, possibly behind the rest of a ``<pad>`` whose deletion splices it): a
release after feed i must be a prefix of D(raw_i + t) for each such t, which the "every split" form implies (take
raw = raw_i + t split at i); ``spoken_prefix`` is checked after every feed against D of what was fed so far.

Regime: pure CPU, Hypothesis bounded (``max_examples`` 200, ``database=None`` so a mutation run never replays a saved
example, ``report_multiple_bugs=False``), strings of <= 14 tokens, <= 6 cuts; seed printed on failure.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from local_astra.conductor import session, speech_gate
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS
from tests.unit.test_answer_cleaner_thoughts import SENTINEL, TOKENS, c_raw, split

BOUNDED = settings(
    max_examples=200,
    deadline=None,
    database=None,
    report_multiple_bugs=False,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)

#: E, as the task derives it
EOT = tuple(t for t in speech_gate._TRIM_TOKENS if t not in SPECIAL_TOKENS)


# --- the oracle: D(raw), from the task text ----------------------------------------------------------------


def cut_at_eot(text: str) -> str:
    """``text`` cut at the earliest occurrence of any member of E (nothing cut when none occurs)."""
    found = [i for i in (text.find(e) for e in EOT) if i != -1]
    return text[: min(found)] if found else text


def d_raw(raw: str) -> str:
    return cut_at_eot(c_raw(raw))


#: raw -> D(raw), by hand from the task's examples
HAND = {
    "Press <end_of_turn> now. ": "Press ",
    "Press <|endoftext|> now. ": "Press ",
    "Hi <end_of_turn> bye. ": "Hi ",
    "Hi <end_of<pad>_turn> bye. ": "Hi ",
    "First. <end_of_turn>Second. Third. ": "First. ",
    "Here <|channel>a<end_of_turn>b<channel|> ok. ": "Here  ok. ",
    f"Okay <end_of_turn> <|channel>thought {SENTINEL}-THOUGHT": "Okay ",
    "Found it: the café opens at nine. <end_of_turn>user: and tomorrow? ": "Found it: the café opens at nine. ",
    "Let me look. <end_of_turn>junk ": "Let me look. ",
    "<end_of_turn> bye. ": "",
    # I4 unchanged: tokenizer literals are deleted, not cut
    "Press <|tool_call> now. ": "Press  now. ",
    "Hi <eos> bye. ": "Hi  bye. ",
    # no E literal: D(raw) = C(raw)
    "x < y and y > z. ": "x < y and y > z. ",
    "Tap <xy> now. ": "Tap <xy> now. ",
}


def test_oracle_matches_the_task_examples():
    """Premise (not pending): the oracle reproduces every D(raw) the task writes out; E is the two strings."""
    assert sorted(EOT) == sorted(("<end_of_turn>", "<|endoftext|>")), EOT
    bad = {raw: (d_raw(raw), want) for raw, want in HAND.items() if d_raw(raw) != want}
    assert not bad, f"oracle vs hand: {bad}"


# --- generators ---------------------------------------------------------------------------------------------

#: T-0117's alphabet plus the task's: the two E literals and their fragments, the hold characters ``f`` and ``x``
#: (in no tokenizer literal), ``<pad>`` and its fragments (so a deletion can splice an E literal)
ALPHABET = tuple(
    dict.fromkeys(
        (
            *TOKENS,
            "<end_of_turn>",
            "<|endoftext|>",
            "<end_of",
            "_turn>",
            "<|endof",
            "text|>",
            "f",
            "x",
            "<pad>",
            "<pa",
            "d>",
        )
    )
)
raws = st.lists(st.sampled_from(ALPHABET), min_size=1, max_size=14).map("".join)


@st.composite
def raw_and_split(draw: Any) -> tuple[str, list[str]]:
    raw = draw(raws)
    return raw, draw(split(raw))


#: continuations that can complete an E literal begun in a held tail: the rest of an E literal, possibly behind the
#: rest of a ``<pad>`` (``<end_of<pa`` + ``d>`` + ``_turn>``)
COMPLETIONS = tuple(
    dict.fromkeys(
        pad[j:] + e[k:]
        for e in EOT
        for k in range(1, len(e) + 1)
        for pad in ("", "<pad>")
        for j in range(1 if pad else 0, max(len(pad), 1))
    )
)

#: the task's split and spliced rows, as explicit cases (so the pending rows are red at the base whatever is drawn)
CASES = (
    ("Hi <end_of_turn> bye. ", ["Hi <end_of_turn> bye. "]),
    ("Hi <end_of_turn> bye. ", ["Hi <end_of", "_turn> bye. "]),
    ("Hi <|endoftext|> bye. ", ["Hi <|endof", "text|> bye. "]),
    ("Hi <end_of<pad>_turn> bye. ", ["Hi <end_of<pa", "d>_turn> bye. "]),
    ("First. <end_of_turn>Second. Third. ", ["First. <end_of_turn>Second. ", "Third. "]),
)


def _cases(test: Any) -> Any:
    for case in CASES:
        test = example(case)(test)
    return test


def cleaner() -> Any:
    cls = getattr(session, "_AnswerCleaner", None)
    assert cls is not None, "session._AnswerCleaner is missing"
    return cls()


def run(c: Any, commits: list[str]) -> tuple[list[str], str]:
    feeds = [c.feed(t) for t in commits]
    return feeds, c.flush()


# =====================================================================================================
# Criterion 9: the properties
# =====================================================================================================


@BOUNDED
@given(raw_and_split())
@_cases
def test_release_is_chunk_independent(case):
    """Criterion 9, first item (mutations 1, delete E literals instead of cutting; 2, cut before C; 7, cut in flush
    only): for every split of raw into commits, the concatenation of every ``feed`` output plus ``flush`` is D(raw)."""
    raw, commits = case
    want = d_raw(raw)
    feeds, flushed = run(cleaner(), commits)
    assert "".join(feeds) + flushed == want, (commits, feeds, flushed, want)


@BOUNDED
@given(raw_and_split())
@_cases
def test_released_text_only_grows(case):
    """Criterion 9, second item (I12): after every ``feed`` the text released so far is a prefix of D(raw)."""
    raw, commits = case
    want = d_raw(raw)
    c = cleaner()
    released = ""
    for i, commit in enumerate(commits):
        released += c.feed(commit)
        assert want.startswith(released), (
            f"T-0126 criterion 9: after feed #{i} the release {released!r} is not a prefix of D(raw) {want!r}; "
            f"commits {commits}"
        )
    released += c.flush()
    assert released == want, (released, want)


@BOUNDED
@given(raw_and_split())
@_cases
def test_no_release_holds_an_eot_prefix(case):
    """Criterion 9, third item (mutations 4, hold alphabet ``_LITERAL_CHARS`` only; 5, no hold; 16, flush or reset on
    a commit boundary): no release holds an E literal, and none holds a prefix that a later commit could make D cut:
    after each feed the cumulative release is a prefix of D(fed so far + t) for every continuation t that completes
    an E literal (``COMPLETIONS``)."""
    _, commits = case
    c = cleaner()
    fed = released = ""
    for i, commit in enumerate(commits):
        out = c.feed(commit)
        fed += commit
        released += out
        hits = [e for e in EOT if e in out or e in released]
        assert not hits, (
            f"T-0126 criterion 9: feed #{i} released {hits} ({out!r}); commits {commits}"
        )
        bad = [t for t in COMPLETIONS if not d_raw(fed + t).startswith(released)]
        assert not bad, (
            f"T-0126 criterion 9: after feed #{i} the release {released!r} holds a prefix that a later commit "
            f"cuts (e.g. {bad[:3]}: D(fed + t) = {d_raw(fed + bad[0])!r}); commits {commits}"
        )
    out = c.flush()
    assert not [e for e in EOT if e in out], out


@BOUNDED
@given(raw_and_split())
@_cases
def test_spoken_prefix_is_d_raw(case):
    """Criterion 9, last item (H-T126-3; mutation 8, cut in ``feed`` but not in ``spoken_prefix``): after every feed,
    and after the flush, ``spoken_prefix`` is D of the text fed so far."""
    raw, commits = case
    c = cleaner()
    fed = ""
    for commit in commits:
        c.feed(commit)
        fed += commit
        assert c.spoken_prefix == d_raw(fed), (commits, fed, c.spoken_prefix, d_raw(fed))
    c.flush()
    assert c.spoken_prefix == d_raw(raw), (c.spoken_prefix, d_raw(raw))


@BOUNDED
@given(raw_and_split())
def test_without_eot_equals_c_raw(case):
    """Criterion 9 / 10 at unit level (negative, green at the base; mutation 3, route tools-on through
    ``_LiteralCleaner``, which cuts at tokenizer trim literals): when C(raw) holds no E literal, the release and
    ``spoken_prefix`` are C(raw), unchanged."""
    raw, commits = case
    want = c_raw(raw)
    assume(not any(e in want for e in EOT))
    c = cleaner()
    feeds, flushed = run(c, commits)
    assert "".join(feeds) + flushed == want, (commits, feeds, flushed, want)
    assert c.spoken_prefix == want, (c.spoken_prefix, want)


# =====================================================================================================
# The hold: no wider than T-0124's rule, wide enough for E (both directions, .claude/rules/tests.md)
# =====================================================================================================

#: raw (one feed) -> what that feed releases at once
HOLDS = {
    # held: the tail from ``<`` can still grow into an E literal (``f``/``x`` are hold characters only because of E)
    "Hi <end_of": "Hi ",
    "Hi <|endof": "Hi ",
    "Hi <end_of<pa": "Hi ",
    # criterion 10's allowance row
    "Tap <x": "Tap ",
    # released at once: a character outside the gate literals' alphabet, or a ``>``, follows the ``<``
    "x < y": "x < y",
    "Since 3 < 5, the first step": "Since 3 < 5, the first step",
    "Done <x9": "Done <x9",
    "a <|>": "a <|>",
}
HOLDS_GREEN = frozenset({"x < y", "Since 3 < 5, the first step", "Done <x9", "a <|>"})


@pytest.mark.parametrize("raw", [pytest.param(r) for r in HOLDS])
def test_holds(raw):
    """Criterion 9's hold (mutations 4 and 5 on the held rows; 6, hold every ``<``-led tail, on the released rows):
    one feed releases exactly the by-hand value; the flush releases the rest, and the two join to D(raw)."""
    c = cleaner()
    out = c.feed(raw)
    assert out == HOLDS[raw], (
        f"T-0126 criterion 9: feed({raw!r}) released {out!r}, expected {HOLDS[raw]!r}"
    )
    assert out + c.flush() == raw == d_raw(raw), "premise: D(raw) = raw for these rows"
