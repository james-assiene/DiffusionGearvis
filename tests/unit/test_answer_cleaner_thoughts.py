"""T-0117 criteria 1 and 7 at the cleaner: ``session._AnswerCleaner`` releases C(raw), chunk by chunk.

The oracle ``c_raw`` is criterion 1 written out from the task text, never a call into the cleaner or the router:

1. repeat until a whole round changes nothing: delete every special literal other than the two channel markers
   (itself to a fixpoint); remove each closed block ``<|channel>...<channel|>``, the first ``<channel|>`` after an
   open closing it (a non-greedy regex, left to right), repeated until stable (a removal can splice a new open
   that a later close closes); truncate at a ``<|channel>`` nothing after it closes; delete the unmatched
   ``<channel|>`` markers left (ruling F1);
2. then ``drop_special_literals`` (written here from ``SPECIAL_TOKENS``).

**Rulings on reviewer A's findings (docs/tasks/T-0117.md).** F1: a stray ``<channel|>`` is deleted inside the
fixpoint, so a marker its deletion splices counts as written in full: ``<|chan<channel|>nel>x`` is ``""`` (the
rejected reading, ``c_raw_literal``, deletes strays only in step 2 and voices ``x``; it is kept only for the
premise row that shows the two differ). F3: within a round, block removal repeats until stable before the
truncate and before strays are deleted: ``<|chan<|channel>x<channel|>nel>A<channel|>B`` is ``B``. The property
rows compare against ``c_raw`` with no filter, so both rulings are imposed on every generated case.

``test_equals_base_cleaner_without_markers`` (criterion 9 at unit level) and the two oracle premises hold at the
base; the other rows need the fix.

Regime: pure CPU, Hypothesis with the repo's default profile and the bounded settings below (``max_examples`` <= 200,
strings of <= 14 tokens, <= 6 cuts), ``report_multiple_bugs=False`` so a failure is one ``AssertionError``.
"""

from __future__ import annotations

import ast
import functools
import itertools
import re
import subprocess
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from local_astra.conductor.session import _AnswerCleaner
from local_astra.toolcall import special_tokens
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS
from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
#: T-0117's base (docs/tasks/T-0117.md, "Branch"). A literal, never a moving ref (tests.md).
BASE_SHA = "c7e04f0"

OPEN, CLOSE = "<|channel>", "<channel|>"
OTHER_LITERALS = tuple(lit for lit in SPECIAL_TOKENS if lit not in (OPEN, CLOSE))
SENTINEL = "SECRET"

BOUNDED = settings(
    max_examples=200,
    deadline=None,
    report_multiple_bugs=False,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)


# --- the oracle: criterion 1, from the task text ------------------------------------------------------------

_BLOCK = re.compile(re.escape(OPEN) + ".*?" + re.escape(CLOSE), flags=re.S)


def _delete(text: str, literals: tuple[str, ...]) -> str:
    while True:
        out = text
        for lit in literals:
            out = out.replace(lit, "")
        if out == text:
            return out
        text = out


def _round(text: str, *, strays: bool) -> str:
    text = _delete(text, OTHER_LITERALS)
    while (
        True
    ):  # each closed block, the first close after an open closing it, until stable (ruling F3)
        nxt = _BLOCK.sub("", text)
        if nxt == text:
            break
        text = nxt
    text = text.split(OPEN, 1)[0]  # what is left of an open has no close after it
    if strays:
        text = text.replace(CLOSE, "")
    return text


def _c(raw: str, *, strays: bool) -> str:
    text = raw
    while True:
        nxt = _round(text, strays=strays)
        if nxt == text:
            return _delete(text, SPECIAL_TOKENS)
        text = nxt


def c_raw(raw: str) -> str:
    """Criterion 1 as ruled (F1, F3): stray closes deleted inside the fixpoint, block removal until stable."""
    return _c(raw, strays=True)


def c_raw_literal(raw: str) -> str:
    """The reading ruling F1 rejected (stray closes deleted only by the final ``drop_special_literals``). Used only
    by the premise row that shows it differs; never an expected value."""
    return _c(raw, strays=False)


# --- generators ---------------------------------------------------------------------------------------------

#: heavy in ``<``, ``|``, ``>``, marker fragments, other literals and their fragments, the sentinel, plain words
TOKENS = (
    OPEN,
    CLOSE,
    "<|chan",
    "nel>",
    "<chan",
    "nel|>",
    "<",
    "|",
    ">",
    "<eos>",
    "<e",
    "os>",
    "<|tool_call>",
    "<|tool",
    "_call>",
    '<|"|>',
    SENTINEL,
    "thought",
    "a",
    "b ",
    " ",
    ". ",
    "x",
)
PLAIN_TOKENS = tuple(t for t in TOKENS if t != SENTINEL)
#: what a thought's inside is made of: no ``<`` (so nothing inside can close it early), the sentinel among words
INSIDE_TOKENS = ("a", "b ", " ", "thought", "x", "|", ">")

raws = st.lists(st.sampled_from(TOKENS), min_size=1, max_size=14).map("".join)


@st.composite
def split(draw: Any, text: str) -> list[str]:
    """``text`` cut into non-empty commits at up to six random points."""
    if len(text) < 2:
        return [text]
    cuts = sorted(draw(st.sets(st.integers(1, len(text) - 1), max_size=6)))
    bounds = [0, *cuts, len(text)]
    return [text[a:b] for a, b in itertools.pairwise(bounds)]


@st.composite
def raw_and_split(draw: Any) -> tuple[str, list[str]]:
    raw = draw(raws)
    return raw, draw(split(raw))


@st.composite
def thoughtful(draw: Any) -> tuple[str, list[str]]:
    """Answer text whose only sentinels are inside thought blocks: closed blocks between plain runs, and maybe an
    open block nothing closes at the end. Plain runs carry marker fragments, so splices happen around blocks."""
    inside = st.lists(st.sampled_from(INSIDE_TOKENS), max_size=3).map("".join)
    plain = st.lists(st.sampled_from(PLAIN_TOKENS), max_size=4).map("".join)
    parts: list[str] = []
    for _ in range(draw(st.integers(0, 3))):
        parts.append(draw(plain))
        parts.append(OPEN + draw(inside) + SENTINEL + draw(inside) + CLOSE)
    parts.append(draw(plain))
    if draw(st.booleans()):
        parts.append(OPEN + draw(inside) + SENTINEL + draw(inside))
    raw = "".join(parts)
    assume(SENTINEL in raw)
    return raw, draw(split(raw))


def run(cleaner: Any, commits: list[str]) -> tuple[list[str], str]:
    feeds = [cleaner.feed(c) for c in commits]
    return feeds, cleaner.flush()


# =====================================================================================================
# Oracle premises (not pending: they check the test, and hold at the base)
# =====================================================================================================

#: raw -> C(raw), by hand from the task's own examples (criteria 2, 3, 5, 6, H-T117-3, the splices)
HAND = {
    "Sure, <|channel>thought SECRET-THOUGHT<channel|>here it is. ": "Sure, here it is. ",
    "Let me look. <|channel>thought SECRET-THOUGHT<channel|>": "Let me look. ",
    "Okay. <|channel>thought SECRET-THOUGHT": "Okay. ",
    "Okay. <|channel>xy<channel|> fine. ": "Okay.  fine. ",
    "A <|channel>x SECRET<channel|> B <|channel>y SECRET<channel|> C. ": "A  B  C. ",
    "<|channel>a SECRET <|channel>b SECRET<channel|>c<channel|> d. ": "c d. ",
    "Sure, <|chan<eos>nel>thought SECRET-THOUGHT<channel|>ok. ": "Sure, ok. ",
    "Sure, <|chan<|channel>x<channel|>nel>SECRET-THOUGHT": "Sure, ",
    "Press <back> now. ": "Press <back> now. ",
    "Press <|tool_call> now. ": "Press  now. ",
    "I thought so. ": "I thought so. ",
    "Wait <|chan": "Wait <|chan",
    "Done <channel|>here. ": "Done here. ",
    # ruling F1: the stray close's deletion splices an open nothing closes
    "Sure, <|chan<channel|>nel>SECRET-THOUGHT": "Sure, ",
    # ruling F3: the inner block's removal splices an open the next close closes, within the same round
    "<|chan<|channel>x<channel|>nel>A<channel|>B": "B",
    "Hi <|chan<|channel>SECRET<channel|>nel>t<channel|> ok. ": "Hi  ok. ",
    # F5: round 1 removes a block and deletes two strays, which splices a new block that round 2 removes
    "A <|channel>thought SECRET-THOUGHT<channel|><|chan<channel|>nel>x<chan<channel|>nel|> B. ": "A  B. ",
}


def test_oracle_matches_the_task_examples():
    """The oracle reproduces every C(raw) the task and its rulings write out (hand-computed above)."""
    bad = {raw: (c_raw(raw), want) for raw, want in HAND.items() if c_raw(raw) != want}
    assert not bad, f"oracle vs hand: {bad}"
    # ruling F1 has teeth: the rejected reading voices the spliced thought here, the ruled one nothing
    assert (c_raw("<|chan<channel|>nel>x"), c_raw_literal("<|chan<channel|>nel>x")) == ("", "x")
    # ruling F3 has teeth: one block pass per round (then strays, then a cut) would drop the B
    assert c_raw("<|chan<|channel>x<channel|>nel>A<channel|>B") == "B"


@BOUNDED
@given(thoughtful())
def test_sentinel_generator_premise(case):
    """The ``thoughtful`` generator puts every sentinel inside thought text: C(raw) never holds it, in either
    reading. (A premise of ``test_no_feed_output_holds_thought_text``.)"""
    raw, _ = case
    assert SENTINEL not in c_raw(raw) and SENTINEL not in c_raw_literal(raw), raw


# =====================================================================================================
# Criterion 7
# =====================================================================================================


#: rulings F1 and F3 as explicit cases (whole and split), so the property rows impose them whatever is generated
RULED = (
    ("Sure, <|chan<channel|>nel>SECRET", ["Sure, <|chan<channel|>nel>SECRET"]),
    ("Sure, <|chan<channel|>nel>SECRET", ["Sure, <|chan<chann", "el|>nel>SECRET"]),
    ("<|chan<|channel>x<channel|>nel>A<channel|>B", ["<|chan<|channel>x<channel|>nel>A<channel|>B"]),
    ("<|chan<|channel>x<channel|>nel>A<channel|>B", ["<|chan", "<|channel>x<channel|>", "nel>A<chan", "nel|>B"]),
    ("Hi <|chan<|channel>SECRET<channel|>nel>t<channel|> ok. ",
     ["Hi <|chan<|channel>SECRET<channel|>nel>t<channel|> ok. "]),
    ("Hi <|chan<|channel>SECRET<channel|>nel>t<channel|> ok. ",
     ["Hi <|chan<|chann", "el>SECRET<channel|>ne", "l>t<channel|> ok. "]),
)  # fmt: skip


def _ruled(test: Any) -> Any:
    for case in RULED:
        test = example(case)(test)
    return test


@BOUNDED
@given(raw_and_split())
@_ruled
def test_release_is_chunk_independent(case):
    """Criterion 7, first item (reviewer mutation 3): for any split of raw into commits, the concatenation of every
    ``feed`` output and ``flush`` is C(raw), and so is ``spoken_prefix`` (H-T117-5, reviewer mutation 9). No
    filter: rulings F1 and F3 hold on every generated case."""
    raw, commits = case
    want = c_raw(raw)
    cleaner = _AnswerCleaner()
    feeds, flushed = run(cleaner, commits)
    assert "".join(feeds) + flushed == want, (commits, feeds, flushed, want)
    assert cleaner.spoken_prefix == want, (commits, cleaner.spoken_prefix, want)


@BOUNDED
@given(thoughtful())
def test_no_feed_output_holds_thought_text(case):
    """Criterion 7, second item (reviewer mutations 8 and 14): every thought carries the ``SECRET`` sentinel (whose
    letters are in no literal), and no single ``feed`` output, read when it is returned, nor ``flush`` holds it."""
    _, commits = case
    cleaner = _AnswerCleaner()
    for i, commit in enumerate(commits):
        out = cleaner.feed(commit)
        assert SENTINEL not in out, (
            f"T-0117 criterion 7: feed #{i} released thought text {out!r}; commits {commits}"
        )
    out = cleaner.flush()
    assert SENTINEL not in out, f"T-0117 criterion 7: flush released {out!r}; commits {commits}"
    assert SENTINEL not in cleaner.spoken_prefix, cleaner.spoken_prefix


@BOUNDED
@given(raw_and_split())
@_ruled
def test_released_text_only_grows(case):
    """Criterion 7, third item (I12): after every ``feed``, the text released so far is a prefix of C(raw)."""
    raw, commits = case
    want = c_raw(raw)
    cleaner = _AnswerCleaner()
    released = ""
    for i, commit in enumerate(commits):
        released += cleaner.feed(commit)
        assert want.startswith(released), (
            f"T-0117 criterion 7: after feed #{i} the release {released!r} is not a prefix of C(raw) {want!r}; "
            f"commits {commits}"
        )
    released += cleaner.flush()
    assert released == want, (released, want)


# =====================================================================================================
# Criterion 9 at unit level: no marker, byte-identical to the base cleaner
# =====================================================================================================


@functools.cache
def _base_cleaner_class() -> type:
    """``_AnswerCleaner`` (and ``_LITERAL_CHARS``) of ``c7e04f0:local_astra/conductor/session.py``, rebuilt by AST
    extraction (as ``_frozen_score_4c7dfdc.py`` freezes a base function); ``drop_special_literals`` is the base's
    too, since ``special_tokens.py`` is unchanged (``test_t0117_differential.test_unowned_files_equal_c7e04f0``)."""
    source = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{BASE_SHA}:local_astra/conductor/session.py"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    tree = ast.parse(source)
    keep: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_AnswerCleaner":
            keep.append(node)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "_LITERAL_CHARS" for t in targets):
                keep.append(node)
    assert {getattr(n, "name", "_LITERAL_CHARS") for n in keep} == {
        "_AnswerCleaner",
        "_LITERAL_CHARS",
    }, f"premise: {BASE_SHA}'s session.py defines _AnswerCleaner and _LITERAL_CHARS at module level"
    namespace: dict[str, Any] = {
        "__name__": "t0117_base_cleaner",
        "SPECIAL_TOKENS": SPECIAL_TOKENS,
        "drop_special_literals": special_tokens.drop_special_literals,
    }
    exec(
        compile(ast.Module(body=keep, type_ignores=[]), f"{BASE_SHA}:session.py", "exec"), namespace
    )
    cls: type = namespace["_AnswerCleaner"]
    return cls


no_marker_raws = st.lists(st.sampled_from(tuple(t for t in TOKENS if t not in (OPEN, CLOSE))),
                          min_size=1, max_size=14).map("".join)  # fmt: skip


@BOUNDED
@given(st.data())
def test_equals_base_cleaner_without_markers(data):
    """Criterion 9 at unit level (negative): for raw in which no ``<|channel>`` and no ``<channel|>`` appears, even
    after deleting the other literals repeatedly, ``"".join(feeds) + flush`` and ``spoken_prefix`` equal the c7e04f0
    cleaner's, for any split; every single ``feed`` output also equals the base's when ``"x" not in raw``. Skipped,
    naming the SHA, if the commit is absent.

    T-0126 re-scope R3 (docs/tasks/T-0126.md, "Owned files"; a spec change to criterion 9 at unit level): T-0126's
    hold covers ``<end_of_turn>`` / ``<|endoftext|>``, so ``x`` (in ``TOKENS``, in no tokenizer literal) becomes a
    hold character and a ``<x`` tail may be released one commit later than at c7e04f0; the per-feed split is
    therefore compared only for raw without ``x``. No ``TOKENS`` string can build either literal, so the totals are
    unchanged."""
    gr.require_commits(BASE_SHA)
    base_cls = _base_cleaner_class()
    raw = data.draw(no_marker_raws)
    cleaned = _delete(raw, OTHER_LITERALS)
    assume(OPEN not in cleaned and CLOSE not in cleaned)
    commits = data.draw(split(raw))
    now, base = _AnswerCleaner(), base_cls()
    got, want = run(now, commits), run(base, commits)
    assert "".join(got[0]) + got[1] == "".join(want[0]) + want[1], (
        f"T-0117 criterion 9: {commits}: now {got}, {BASE_SHA} {want}"
    )
    if "x" not in raw:  # T-0126 R3: the per-feed split only where the wider hold cannot move a tail
        assert got == want, f"T-0117 criterion 9: {commits}: now {got}, {BASE_SHA} {want}"
    assert now.spoken_prefix == base.spoken_prefix, (now.spoken_prefix, base.spoken_prefix)
