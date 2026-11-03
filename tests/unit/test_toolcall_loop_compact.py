"""T-0120: the tool loop's pure compaction helper, ``loop.compact(steps, budget)`` (ruling H-T115-0; H-T120-1..3).

Every expected value is hand-computed or built with ``render_call``/``render_response``/``step_prompt``, which
T-0113 proved against the pinned template oracle. ``continuation_chars`` is measured here through ``step_prompt``
(criterion 1 defines it as the length of the final assistant text that ``step_prompt`` builds), never through the
function under test.

**Fixture arithmetic (the six-step search loop, ``_six()``).** Each step is ``"Looking that up. "`` (17 chars), the
call ``google_search{query:"omelette recipe k"}`` (77 chars) and a 5-result ``search_payload`` with 320-char
snippets, which ``render_response`` caps to a 1 499-char block. One step is 1 593 chars, six are 9 558. The stub
``<|tool_response>response:google_search{note:<|"|>earlier result omitted<|"|>,results:5}<tool_response|>`` is
16 + 22 + 1 + 5 + 5 + 22 + 5 + 10 + 1 + 16 = 103 chars, so one elision saves 1 396:

    elided k   0      1      2      3      4 (all candidates; steps 4 and 5 are the kept tail)
    chars      9558   8162   6766   5370   3974

- budget 9 000 -> target 6 000: k = 3, the set {0, 1, 2} (newest-first among candidates would be {1, 2, 3});
  a mutant stopping at the budget stops at k = 1 (8 162 <= 9 000).
- budget 6 000 -> target 4 000: k = 4, {0, 1, 2, 3}; a mutant stopping at the budget stops at k = 3 (5 370).
- budget 3 900 -> target 2 600: all four candidates are elided and the result, 3 974, is still over the budget.

The stub's key order was checked against the existing renderer before the literal was relied on:
``render_response("google_search", {"note": ..., "results": 5})`` yields exactly the literal above (``dictsort``
puts ``note`` before ``results``).

API read from the task (the planner's text names the parameters, not their kind): ``compact(steps, budget, *,
keep_last=COMPACT_KEEP_LAST, target=<budget * 2 // 3>)`` returning a tuple of ``LoopStep``; ``elided_payload(original)
-> payload``; ``continuation_chars(steps) -> int``. A stub is recognised by ``note == ELIDED_NOTE``.
"""

from __future__ import annotations

import ast
import copy
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.contracts.llm import Message, Prompt, TextPart
from local_astra.contracts.perception import AssistFlags
from local_astra.contracts.toolcall import SearchResult, SearchResults
from local_astra.toolcall import loop
from local_astra.toolcall.loop import LoopStep, render_call, render_response
from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
NOTE = "earlier result omitted"
D = '<|"|>'
#: T-0120b re-scope (criterion 2, H-T120b-0/1): a search response's stub invites a re-search. 16 + 23 + 5 + 5 + 51 +
#: 5 + 1 + 9 + 1 + 16 = 132 chars (was 103), so one elision of a 1 499-char block saves 1 367 (was 1 396):
#: elided k 0..4 -> 9 558, 8 191, 6 824, 5 457, 4 090. The status and bare forms keep ``NOTE``.
SEARCH_NOTE = "earlier result omitted; search again if you need it"
STUB_SEARCH = (
    '<|tool_response>response:google_search{note:<|"|>earlier result omitted; search again if you need it<|"|>,'
    "results:5}<tool_response|>"
)
#: T-0120's search stub (the old note): ``_is_stub`` still recognises it (T-0120b criterion 2).
OLD_STUB_SEARCH = '<|tool_response>response:google_search{note:<|"|>earlier result omitted<|"|>,results:5}<tool_response|>'


def _api(name: str) -> Any:
    """A T-0120 name from ``loop``, or ``NotImplementedError`` while it does not exist (never a collection error)."""
    try:
        return getattr(loop, name)
    except AttributeError as exc:
        raise NotImplementedError(f"T-0120: loop.{name} does not exist yet") from exc


def _compact(steps: Sequence[LoopStep], budget: int, **kw: Any) -> tuple[LoopStep, ...]:
    out = _api("compact")(steps, budget, **kw)
    assert isinstance(out, tuple), f"compact returns a tuple, not {type(out).__name__}"
    return out


# --------------------------------------------------------------------------- fixtures and the oracle

TOOLS = (
    {"type": "function", "function": {"name": "google_search", "parameters": {"type": "object"}}},
)


def _prompt1() -> Prompt:
    return Prompt(
        messages=(
            Message(role="system", content=(TextPart(text="You are Astra."),)),
            Message(role="user", content=(TextPart(text="How do I make an omelette?"),)),
        ),
        prefix_len=2,
        thinking=True,
        assists=AssistFlags(),
        tools=TOOLS,
    )


P1 = _prompt1()


def _chars(steps: Sequence[LoopStep]) -> int:
    """The oracle for ``continuation_chars``: the final assistant text ``step_prompt`` builds (criterion 1)."""
    text = loop.step_prompt(P1, steps).messages[-1].content[0].text
    assert isinstance(text, str)
    return len(text)


def _big_search(k: int) -> dict[str, Any]:
    return loop.search_payload(
        SearchResults(
            query=f"omelette recipe {k}",
            results=tuple(
                SearchResult(
                    title=f"Omelette {k}.{i}",
                    url=f"https://example.org/{k}/{i}",
                    snippet="egg " * 80,
                )
                for i in range(5)
            ),
        )
    )


def _search_step(k: int, text: str = "Looking that up. ") -> LoopStep:
    return LoopStep(
        text=text,
        name="google_search",
        arguments={"query": f"omelette recipe {k}"},
        response=_big_search(k),
    )


def _six() -> tuple[LoopStep, ...]:
    steps = tuple(_search_step(k) for k in range(6))
    # fixture preconditions (module docstring): if these move, every hand number below moves with them
    assert [len(render_response(s.name, s.response)) for s in steps] == [1499] * 6
    assert len(render_call(steps[0].name, steps[0].arguments)) == 77
    assert len(STUB_SEARCH) == 132 and len(OLD_STUB_SEARCH) == 103
    assert _chars(steps) == 9558
    return steps


def _elided(before: Sequence[LoopStep], after: Sequence[LoopStep]) -> list[int]:
    """Indices whose response changed."""
    assert len(after) == len(before)
    return [
        i for i, (b, a) in enumerate(zip(before, after, strict=True)) if a.response != b.response
    ]


def _assert_kept(
    before: Sequence[LoopStep], after: Sequence[LoopStep], elided: Sequence[int]
) -> None:
    for i, (b, a) in enumerate(zip(before, after, strict=True)):
        if i not in elided:
            assert a is b, f"step {i} was not elided, so it must be the same object"


# --------------------------------------------------------------------------- criterion 1


def test_continuation_chars_is_the_step_prompt_text_length() -> None:
    """Criterion 1's definition, on a hand-typed one-step case (leading whitespace dropped as ``step_prompt``'s
    ``lstrip`` drops it) and on the six-step fixture (9 558, module docstring)."""
    continuation_chars = _api("continuation_chars")
    one = [LoopStep(text="  \nHi ", name="scroll", arguments={"direction": "down"}, response="ok")]
    by_hand = (
        "Hi "
        + '<|tool_call>call:scroll{direction:<|"|>down<|"|>}<tool_call|>'
        + '<|tool_response>response:scroll{value:<|"|>ok<|"|>}<tool_response|>'
    )
    assert continuation_chars(one) == len(by_hand) == 3 + 61 + 67
    assert continuation_chars(_six()) == 9558


def test_under_budget_is_identity() -> None:
    """Criterion 1 (the "once" timing). At ``budget == continuation_chars`` (the boundary) and well above it, the
    result is a tuple equal to ``tuple(steps)`` whose elements are the same objects, a list input included. One char
    under the boundary the same steps are compacted (the guard is not too wide)."""
    steps = _six()
    for budget in (9558, 50_000):
        out = _compact(list(steps), budget)
        assert out == tuple(steps)
        assert all(a is b for a, b in zip(out, steps, strict=True))
    assert _elided(steps, _compact(steps, 9557)) != []


# --------------------------------------------------------------------------- criterion 2


def test_oldest_first_and_kept_tail() -> None:
    """Criterion 2. Six maximal search steps, budget 9 000 -> target 6 000: exactly {0, 1, 2} (5 457 chars), the
    oldest first; budget 6 000 -> target 4 000: {0, 1, 2, 3} (4 090). Steps 4 and 5 are never elided.
    (T-0120b re-scope: 5 370 -> 5 457, 3 974 -> 4 090, the 132-char search stub.)"""
    steps = _six()
    out = _compact(steps, 9000)
    assert _elided(steps, out) == [0, 1, 2]
    assert _chars(out) == 5457
    _assert_kept(steps, out, [0, 1, 2])
    assert render_response(out[0].name, out[0].response) == STUB_SEARCH
    out = _compact(steps, 6000)
    assert _elided(steps, out) == [0, 1, 2, 3]
    assert _chars(out) == 4090
    assert out[4] is steps[4] and out[5] is steps[5]


def test_stops_at_target_not_at_budget() -> None:
    """Adversarial (criterion 2). A mutant that stops once under ``budget`` elides one fewer: at 6 000 it stops at
    k = 3 (5 457 <= 6 000), the target 4 000 needs k = 4. Explicit ``target=budget`` is accepted and gives the
    budget-stopping answer, so the two readings are told apart by the default alone. (T-0120b re-scope: the
    first-point boundary moves from 5 370 to 5 457.)"""
    steps = _six()
    assert _elided(steps, _compact(steps, 6000)) == [0, 1, 2, 3]
    assert _elided(steps, _compact(steps, 6000, target=6000)) == [0, 1, 2]
    assert _elided(steps, _compact(steps, 9000, target=6000)) == [0, 1, 2]
    # stops at the *first* point under the target: one elision more than needed is wrong too
    assert _elided(steps, _compact(steps, 9000, target=5457)) == [0, 1, 2]
    assert _elided(steps, _compact(steps, 9000, target=5456)) == [0, 1, 2, 3]


@pytest.mark.parametrize(("pad", "expected"), [(238, 5), (239, 6)])
def test_default_target_is_budget_times_two_floordiv_three(pad: int, expected: int) -> None:
    """Criterion 2's defaults at the shipped budget: ``target = 14 000 * 2 // 3 = 9 333``, ``keep_last = 2``. Ten
    search steps (15 930 chars) with the last step's text padded by ``pad`` chars; after five elisions the loop is
    15 930 - 5 * 1 367 = 9 095 + pad chars: 9 333 fits the target (stop at five), 9 334 does not (a sixth). A
    ceiling target (9 334) stops at five on both rows. (T-0120b re-scope: pads 383/384 -> 238/239, saving 1 396 ->
    1 367.)"""
    assert _api("COMPACT_KEEP_LAST") == 2
    steps = (
        *(_search_step(k) for k in range(9)),
        _search_step(9, "Looking that up. " + "p" * pad),
    )
    assert _chars(steps) == 15_930 + pad
    out = _compact(steps, 14_000)
    assert _elided(steps, out) == list(range(expected))
    assert _chars(out) == 15_930 + pad - 1367 * expected
    assert _compact(steps, 14_000) == _compact(steps, 14_000, keep_last=2, target=9333)


def test_keep_last_is_a_parameter() -> None:
    """Criterion 2: ``keep_last=1`` lets step 4 go (budget 3 900 -> target 2 600: all five candidates, 4 090 - 1 367 =
    2 723); ``keep_last=3`` keeps step 3 (three elisions at most, 5 457). (T-0120b re-scope: 2 578 -> 2 723, still
    all five; 5 370 -> 5 457.)"""
    steps = _six()
    out = _compact(steps, 3900, keep_last=1)
    assert _elided(steps, out) == [0, 1, 2, 3, 4]
    assert _chars(out) == 2723
    out = _compact(steps, 3900, keep_last=3)
    assert _elided(steps, out) == [0, 1, 2]
    assert _chars(out) == 5457


def test_bad_arguments_raise() -> None:
    """Criterion 2's ``ValueError``s: ``keep_last < 1``, ``budget < 1``, ``target > budget``; each raised whether or
    not the steps fit (the criterion states no condition). The edges ``keep_last == 1``, ``budget == 1`` and
    ``target == budget`` are valid (the guard is not too wide)."""
    compact = _api("compact")
    six = _six()
    small = (LoopStep(text="Hi ", name="scroll", arguments={"direction": "down"}, response="ok"),)
    for steps in (six, small):
        for kw in ({"keep_last": 0}, {"keep_last": -1}):
            with pytest.raises(ValueError):
                compact(steps, 50_000, **kw)
        for budget in (0, -5):
            with pytest.raises(ValueError):
                compact(steps, budget)
        with pytest.raises(ValueError):
            compact(steps, 50_000, target=50_001)
    _compact(six, 6000, keep_last=1)
    _compact(six, 1)
    _compact(six, 6000, target=6000)


# --------------------------------------------------------------------------- criterion 3


def test_calls_and_texts_byte_identical() -> None:
    """Criterion 3. Step 0's text carries a raw special literal (``compact`` must not sanitise it in place: the
    renderer does that), step 1 is a ``type_at`` with a 500-char argument and an already-sanitised text. With budget
    1 000 the four candidates are elided; every ``text``/``name``/``arguments`` is equal to the input, of the same
    type, every call renders byte-identically, the count is unchanged, and only elided responses change."""
    raw_text = "Let me check <|turn> this. "
    steps = (
        LoopStep(
            text=raw_text,
            name="google_search",
            arguments={"query": "omelette"},
            response=_big_search(0),
        ),
        LoopStep(
            text="Typing it out now. ",
            name="type_at",
            arguments={"text": "z" * 500, "x": 10, "y": 20},
            response="r" * 1400,
        ),
        *(_search_step(k) for k in (2, 3, 4, 5)),
    )
    before = copy.deepcopy(steps)
    out = _compact(steps, 1000)
    assert len(out) == 6
    assert _elided(steps, out) == [0, 1, 2, 3]
    for b, a in zip(steps, out, strict=True):
        assert a.text == b.text and type(a.text) is str
        assert a.name == b.name
        assert a.arguments == b.arguments and type(a.arguments) is type(b.arguments)
        assert render_call(a.name, a.arguments) == render_call(b.name, b.arguments)
        assert isinstance(a, LoopStep)
    assert out[0].text == raw_text
    assert out[1].arguments["text"] == "z" * 500
    assert out[0].response == {"note": SEARCH_NOTE, "results": 5}
    assert out[1].response == {"note": NOTE}
    _assert_kept(steps, out, [0, 1, 2, 3])
    assert steps == before, "compact must not mutate its input"


# --------------------------------------------------------------------------- criterion 4


STUB_CASES = {
    "search": (
        "google_search",
        _big_search(7),
        {"note": SEARCH_NOTE, "results": 5},
        STUB_SEARCH,
    ),
    "action": (
        "click",
        loop.action_payload("click the 'Buy now' button"),
        {"note": NOTE, "status": "dry_run"},
        '<|tool_response>response:click{note:<|"|>earlier result omitted<|"|>,status:<|"|>dry_run<|"|>}'
        "<tool_response|>",
    ),
    "refused": (
        "click",
        loop.status_payload("refused", "the user did not confirm the purchase"),
        {"note": NOTE, "status": "refused"},
        '<|tool_response>response:click{note:<|"|>earlier result omitted<|"|>,status:<|"|>refused<|"|>}'
        "<tool_response|>",
    ),
    # The "search not connected" response every live session gives until T-0111b: it has both a ``results`` list
    # and a ``status``, so by the recorded ruling it takes the ``results`` form (count 0). Its ``note`` is not
    # ``ELIDED_NOTE``, so it is not a stub. The literal was checked against ``render_response`` on the mapping
    # ``{"note": "earlier result omitted", "results": 0}`` (dictsort: note, results), not through ``elided_payload``.
    # T-0120b re-scope: the results form takes the search note (132 chars).
    "not_connected": (
        "google_search",
        loop.search_payload(None),
        {"note": SEARCH_NOTE, "results": 0},
        '<|tool_response>response:google_search{note:<|"|>earlier result omitted; search again if you need it<|"|>,'
        "results:0}<tool_response|>",
    ),
    "plain": (
        "type_at",
        "a plain string result from a tool that returns text",
        {"note": NOTE},
        '<|tool_response>response:type_at{note:<|"|>earlier result omitted<|"|>}<tool_response|>',
    ),
}


@pytest.mark.parametrize("case", list(STUB_CASES))
def test_stub_render_by_hand(case: str) -> None:
    """Criterion 4 (H-T120-3): the three stub forms and their hand-typed renderings, and ``ELIDED_NOTE``. T-0120b
    re-scope: the two results-form rows take the search note; action, refused and plain keep ``ELIDED_NOTE``."""
    name, original, payload, rendered = STUB_CASES[case]
    assert _api("ELIDED_NOTE") == NOTE
    snapshot = copy.deepcopy(original)
    stub = _api("elided_payload")(original)
    assert stub == payload
    assert render_response(name, stub) == rendered
    assert original == snapshot, "elided_payload must not mutate its input"


def test_elided_step_renders_the_stub_inside_the_continuation() -> None:
    """Criterion 4 end to end: the compacted continuation contains the hand-typed stub block where step 0's response
    was, preceded by step 0's text and call."""
    steps = _six()
    text = loop.step_prompt(P1, _compact(steps, 9000)).messages[-1].content[0].text
    head = (
        "Looking that up. "
        + render_call("google_search", {"query": "omelette recipe 0"})
        + STUB_SEARCH
    )
    assert text.startswith(head)
    assert text.count(STUB_SEARCH) == 3


def test_short_response_never_elided() -> None:
    """Adversarial, the too-wide direction (criterion 4's last sentence). Step 0's plain 21-char response renders to
    87 chars, exactly its stub's length (16 + 16 + 7 + 5 + 21 + 5 + 1 + 16 against 16 + 16 + 6 + 5 + 22 + 5 + 1 +
    16), so it is not shorter and survives; step 1's short refusal is shorter than its stub and survives. Budget
    1 000 wants every candidate, so the two big ones (2, 3) are elided and nothing else."""
    equal = "x" * 21
    assert len(render_response("type_at", equal)) == 87
    assert len(render_response("type_at", {"note": NOTE})) == 87
    short_refusal = loop.status_payload("refused", "no")
    steps = (
        LoopStep(text="Typing. ", name="type_at", arguments={"text": "hi"}, response=equal),
        LoopStep(
            text="Clicking. ", name="click", arguments={"x": 1, "y": 2}, response=short_refusal
        ),
        *(_search_step(k) for k in (2, 3, 4, 5)),
    )
    out = _compact(steps, 1000)
    assert _elided(steps, out) == [2, 3]
    _assert_kept(steps, out, [2, 3])


def test_one_char_longer_than_the_stub_is_elided() -> None:
    """The too-strict twin of the row above: a plain 22-char response renders to 88 chars, one more than its
    87-char stub, so with budget 1 000 it is elided with the two big candidates."""
    longer = "x" * 22
    assert len(render_response("type_at", longer)) == 88
    steps = (
        LoopStep(text="Typing. ", name="type_at", arguments={"text": "hi"}, response=longer),
        *(_search_step(k) for k in (1, 2, 3, 4)),
    )
    out = _compact(steps, 1000)
    assert _elided(steps, out) == [0, 1, 2]
    assert out[0].response == {"note": NOTE}


def _not_connected_step(k: int) -> LoopStep:
    return LoopStep(
        text="Looking that up. ",
        name="google_search",
        arguments={"query": f"omelette recipe {k}"},
        response=loop.search_payload(None),
    )


def test_not_connected_response_is_elided_not_mistaken_for_a_stub() -> None:
    """Adversarial (criteria 4 and 5). The "search not connected" response carries a ``note`` that is not
    ``ELIDED_NOTE``: a stub recogniser keyed on the presence of ``note`` (or on the key shape) would never elide it.
    It has both ``results`` and ``status``, so its stub is the ``results`` form with count 0.

    Arithmetic: the response renders to 16 + 22 + 1 + 5 + 5 + 39 + 5 + 1 + 8 + 2 + 1 + 7 + 5 + 7 + 5 + 1 + 16 = 146
    chars, its stub (T-0120b: the search note) to 132 < 146, so it is still elided and one elision saves 14. One
    step is 17 + 77 + 146 = 240 chars, four are 960; one elision gives 946. Budget 1: both candidates (0, 1) are
    elided (932). Budget 959 with ``target=946``: the oldest only. (T-0120b re-scope: 103 -> 132, 43 -> 14,
    917 -> 946.)"""
    nc_render = render_response("google_search", loop.search_payload(None))
    assert len(nc_render) == 146
    stub = (
        '<|tool_response>response:google_search{note:<|"|>earlier result omitted; search again if you need it<|"|>,'
        "results:0}<tool_response|>"
    )
    assert len(stub) == 132 < 146
    steps = tuple(_not_connected_step(k) for k in range(4))
    assert _chars(steps) == 960
    out = _compact(steps, 1)
    assert _elided(steps, out) == [0, 1]
    _assert_kept(steps, out, [0, 1])
    for s in out[:2]:
        assert s.response == {"note": SEARCH_NOTE, "results": 0}
        assert render_response(s.name, s.response) == stub
    assert _chars(out) == 960 - 2 * 14
    out = _compact(steps, 959, target=946)
    assert _elided(steps, out) == [0]
    assert _chars(out) == 946
    # mixed with maximal search steps: the not-connected step is an ordinary candidate in oldest-first order
    mixed = (_not_connected_step(0), *(_search_step(k) for k in (1, 2, 3)))
    out = _compact(mixed, 1)
    assert _elided(mixed, out) == [0, 1]
    assert render_response(out[0].name, out[0].response) == stub


# --------------------------------------------------------------------------- criterion 5


def test_a_stub_is_never_re_elided() -> None:
    """Criterion 5: a stub of each form that comes back to ``compact`` stays byte-identical even at budget 1 (where
    everything that can be elided is). The ``results`` stub carries an int, so a recogniser that only looks at the
    shape would turn it into the plain form; the plain stub would otherwise be compared with itself. T-0120b
    criterion 2: both notes are recognised -- the new search stub and an old-note search stub keep their bytes."""
    stubs = [
        ("google_search", {"note": SEARCH_NOTE, "results": 5}),
        ("google_search", {"note": NOTE, "results": 5}),
        ("click", {"note": NOTE, "status": "dry_run"}),
        ("type_at", {"note": NOTE}),
    ]
    steps = (
        *(LoopStep(text="Earlier. ", name=n, arguments={"q": "a"}, response=r) for n, r in stubs),
        *(_search_step(k) for k in (3, 4)),
    )
    out = _compact(steps, 1)
    for i, (name, payload) in enumerate(stubs):
        assert render_response(out[i].name, out[i].response) == render_response(name, payload)
    assert render_response(out[0].name, out[0].response) == STUB_SEARCH
    assert render_response(out[1].name, out[1].response) == OLD_STUB_SEARCH
    assert _compact(_compact(_six(), 1), 1) == _compact(_six(), 1)


_TEXT = st.text(alphabet="abc ", max_size=12)
_SMALL = st.text(alphabet="abcxyz", max_size=40)
_RESULT = st.fixed_dictionaries({"snippet": _SMALL, "title": _SMALL, "url": _SMALL})
_PAYLOAD = st.one_of(
    st.builds(lambda q, r: {"query": q, "results": r}, _SMALL, st.lists(_RESULT, max_size=3)),
    st.builds(lambda a: loop.action_payload(a), _SMALL),
    st.builds(
        lambda s, r: loop.status_payload(s, r),
        st.sampled_from(["refused", "failed", "timeout"]),
        _SMALL,
    ),
    _SMALL,
    st.builds(loop.search_payload, st.none()),  # "search not connected": fixed size, so bounded
    st.just({"note": NOTE, "results": 3}),
    st.just({"note": SEARCH_NOTE, "results": 3}),
    st.just({"note": NOTE, "status": "dry_run"}),
    st.just({"note": NOTE}),
)
_STEP = st.builds(
    lambda t, n, q, r: LoopStep(text=t, name=n, arguments={"query": q}, response=r),
    _TEXT,
    st.sampled_from(["google_search", "click", "type_at"]),
    _TEXT,
    _PAYLOAD,
)
_STEPS = st.lists(_STEP, min_size=1, max_size=10).map(tuple)


def _is_stub(payload: Any) -> bool:
    return isinstance(payload, Mapping) and payload.get("note") in (NOTE, SEARCH_NOTE)


def _brute_elided(steps: tuple[LoopStep, ...], budget: int) -> list[int]:
    """Independent oracle, by brute force over prefixes: the candidates are ``steps[:-2]`` that are not stubs and
    whose stub renders shorter; the answer is the shortest oldest-first prefix of them that brings the
    continuation (measured by ``step_prompt``) to ``budget * 2 // 3`` or under, or all of them; nothing if the
    steps already fit ``budget``."""
    elided_payload = _api("elided_payload")
    if _chars(steps) <= budget:
        return []
    target = budget * 2 // 3
    cands = [
        i
        for i in range(max(0, len(steps) - 2))
        if not _is_stub(steps[i].response)
        and len(render_response(steps[i].name, elided_payload(steps[i].response)))
        < len(render_response(steps[i].name, steps[i].response))
    ]
    for k in range(len(cands) + 1):
        trial = [
            LoopStep(s.text, s.name, s.arguments, elided_payload(s.response))
            if i in cands[:k]
            else s
            for i, s in enumerate(steps)
        ]
        if _chars(trial) <= target:
            return cands[:k]
    return cands


def test_idempotent_and_monotone() -> None:
    """Criteria 2, 3 and 5 as properties (Hypothesis, 150 examples, at most 10 steps of tiny payloads, budgets
    1..1 500, no example database): the elided set equals the brute-force oracle; idempotence; and for
    ``s2 = compact(s, b) + (x,)`` every step elided in ``compact(s, b)`` is still elided in ``compact(s2, b)`` with
    identical rendered bytes (the prefix stability the cache depends on). The API is resolved before Hypothesis
    starts, so a missing name fails in milliseconds instead of being shrunk."""
    _api("compact")
    _api("elided_payload")
    _property()


@settings(
    max_examples=150,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(steps=_STEPS, x=_STEP, budget=st.integers(min_value=1, max_value=1500))
def _property(steps: tuple[LoopStep, ...], x: LoopStep, budget: int) -> None:
    elided_payload = _api("elided_payload")
    out = _compact(steps, budget)
    elided = _brute_elided(steps, budget)
    assert _elided(steps, out) == elided
    for i, (b, a) in enumerate(zip(steps, out, strict=True)):
        assert (a.text, a.name, a.arguments) == (b.text, b.name, b.arguments)
        if i in elided:
            assert a.response == elided_payload(b.response)
        else:
            assert a is b
    assert _compact(out, budget) == out
    s2 = (*out, x)
    out2 = _compact(s2, budget)
    for i in elided:
        assert render_response(out2[i].name, out2[i].response) == render_response(
            out[i].name, out[i].response
        )


# --------------------------------------------------------------------------- criterion 6


def test_still_over_returns_fully_elided() -> None:
    """Criterion 6: budget 3 900 (target 2 600). Eliding all four candidates reaches 4 090, still over the budget:
    the result is the fully elided steps, steps 4 and 5 untouched (a ``keep_last`` off by one would take step 4).
    (T-0120b re-scope: 3 974 -> 4 090.)"""
    steps = _six()
    out = _compact(steps, 3900)
    assert _elided(steps, out) == [0, 1, 2, 3]
    assert _chars(out) == 4090 > 3900
    assert out[4] is steps[4] and out[5] is steps[5]
    assert all(render_response(s.name, s.response) == STUB_SEARCH for s in out[:4])


# --------------------------------------------------------------------------- the public names


def test_new_names_are_exported() -> None:
    """Owned files: ``loop.__all__`` gains the five T-0120 names."""
    for name in (
        "compact",
        "elided_payload",
        "continuation_chars",
        "COMPACT_KEEP_LAST",
        "ELIDED_NOTE",
    ):
        _api(name)
        assert name in loop.__all__, name


def test_search_note_exported_and_the_old_note_kept() -> None:
    """T-0120b criterion 2 / owned files: ``ELIDED_SEARCH_NOTE`` is the hand-typed invitation and is exported;
    ``ELIDED_NOTE`` is unchanged (the status and bare forms keep it: re-calling an action repeats it)."""
    assert _api("ELIDED_NOTE") == NOTE
    try:
        value = loop.ELIDED_SEARCH_NOTE  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise NotImplementedError("T-0120b: loop.ELIDED_SEARCH_NOTE does not exist yet") from exc
    assert value == SEARCH_NOTE
    assert "ELIDED_SEARCH_NOTE" in loop.__all__
    stub = _api("elided_payload")(loop.action_payload("click the 'Buy now' button"))
    assert stub == {"note": NOTE, "status": "dry_run"} and "search" not in str(stub), stub


# --------------------------------------------------------------------------- criteria 7 and 8 (negative; hold today)

#: Criterion 7: ``loop.py`` may import the stdlib (minus I/O and clocks), ``local_astra.contracts`` and
#: ``toolcall.special_tokens``. Stricter than T-0113's row, which also admits ``router``.
_NO_IO_STDLIB = frozenset(
    {
        "asyncio", "datetime", "http", "io", "logging", "os", "pathlib", "random", "requests", "select",
        "shutil", "socket", "subprocess", "sys", "tempfile", "threading", "time", "urllib",
    }
)  # fmt: skip
_NO_IO_CALLS = frozenset({"open", "print", "input", "exec", "eval", "__import__", "import_module"})


def _allowed(mod: str) -> bool:
    top = mod.split(".")[0]
    if top in sys.stdlib_module_names:
        return top not in _NO_IO_STDLIB
    return any(
        mod == a or mod.startswith(a + ".")
        for a in ("local_astra.contracts", "local_astra.toolcall.special_tokens")
    )


def _violations(source: str) -> list[str]:
    bad: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if not _allowed(a.name)]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                mod = "local_astra.toolcall." + mod if node.level == 1 else "?" + mod
            bad += [
                f"{mod}.{a.name}"
                for a in node.names
                if not (_allowed(mod) or _allowed(f"{mod}.{a.name}"))
            ]
        elif isinstance(node, ast.Call):
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fname in _NO_IO_CALLS:
                bad.append(f"call {fname}")
    return bad


def test_imports_and_purity() -> None:
    """Criterion 7 (negative; holds before the implementation too). The predicate is shown to catch each forbidden form and
    pass each allowed one on a synthetic source, then applied to ``loop.py``."""
    probe = "\n".join(
        [
            "from __future__ import annotations",
            "from collections.abc import Mapping",
            "from local_astra.contracts.llm import Prompt",
            "from .special_tokens import drop_special_literals",
            "import time",
            "from datetime import datetime",
            "from local_astra.toolcall import router",
            "from .native import parse",
            "import pydantic",
            "open('x')",
            "print('x')",
        ]
    )
    assert sorted(_violations(probe)) == sorted(
        [
            "time",
            "datetime.datetime",
            "local_astra.toolcall.router",
            "local_astra.toolcall.native.parse",
            "pydantic",
            "call open",
            "call print",
        ]
    )
    assert _violations((REPO / "local_astra/toolcall/loop.py").read_text()) == []


#: T-0120's fork point: main at c7e04f0 (the task text names 4f84b04; the two commits between them add task files
#: only; the branch never merged main, so this is also the merge-base). The range is closed: BASE..TIP, TIP the
#: branch tip that merges (.claude/rules/tests.md, T-0094 "pin both ends"), so later tasks that edit these paths
#: on main (T-0117 changes ``session.py``) do not redden it.
BASE = "c7e04f0c511a38f78d2cb3af6754534fb36e8df3"
TIP = "b92151243969799ea54f4681b1174f125cf50eb0"
#: Power pair: T-0114's merge a54e5f8 and its first parent, a range known to change ``session.py``.
POWER_BASE = "09b3ad2fb705d3fbe87b5e5fd97d60df1894f3b8"
POWER_TIP = "a54e5f838b1100fd913dfa2cc4591bb16884b7e1"
UNTOUCHED = (
    "local_astra/conductor/session.py",
    "local_astra/conductor/turn_machine.py",
    "local_astra/contracts",
    "tests/unit/test_toolcall_loop_render.py",
    "tests/unit/test_toolcall_loop_injection.py",
    "tests/cd/unit/test_toolcall_loop_oracle.py",
)


def test_imports_and_session_untouched() -> None:
    """Criteria 7-8 (negative; holds before the implementation too): ``session.py`` (so ``MAX_CONTINUATION``),
    ``turn_machine.py``, ``contracts/**`` and the T-0113 test modules (its rows pass *unedited*) are unchanged since
    BASE..TIP (closed range); ``MAX_CONTINUATION`` is still 14 000 in the source. The power twin shows the same diff
    predicate sees T-0114's ``session.py`` change. Skips, naming the SHA, if a commit is absent."""
    gr.require_commits(BASE, TIP, POWER_BASE, POWER_TIP)
    assert gr.changed(BASE, TIP, *UNTOUCHED) == []
    assert gr.changed(POWER_BASE, POWER_TIP, *UNTOUCHED) == ["local_astra/conductor/session.py"]
    tree = ast.parse((REPO / "local_astra/conductor/session.py").read_text())
    values = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "MAX_CONTINUATION" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert values == [14_000]
