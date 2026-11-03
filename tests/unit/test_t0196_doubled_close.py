"""T-0196 criteria 1-3, 9, 10: a valid call followed only by extra ``<tool_call|>`` closes (and whitespace) is judged
by ``live.classify`` as the same call with one close; nothing else moves (docs/tasks/T-0196.md).

**The oracle** is ``local_astra/toolcall/live.py`` as committed at the literal BASE ``611068bb`` (the task's
Definitions), executed as its own module over today's ``native``/``score``/``tools`` by
``test_toolcall_live._live_at`` (skips when the commit is absent). Expected names and arguments of the accepted rows
are also written out by hand.

**The shape** (Definitions, "Doubled-close shape") is checked by ``doubled_split`` below: an independent string-aware
walk over the body (the text after the leading ``<|tool_call>``), written from the definition and never importing the
implementation. ``C`` ends at the first ``<tool_call|>`` outside every ``<|"|>`` string; ``T`` (the rest) is non-empty,
only ``<tool_call|>`` literals and whitespace, with at least one ``<tool_call|>``.

Readings (reported to the orchestrator):

- Criterion 2 lists "extra closes then ``<turn|>``" among the tails that keep BASE's verdict, but ``<turn|>`` is a stop
  text and the Definitions take the shape *after* the stop-text drop: ``VALID + CLOSE + "<turn|>"`` is a doubled-close
  text by the Definitions (and BASE already judges ``VALID + "<turn|>"`` valid). No row pins a trailing ``<turn|>``
  either way; the ``<turn|>`` row here puts it before a newline (``<turn|>\\n``), which is not dropped.
- The negative rows (criteria 2, 3, 9, 10) are green at BASE by construction; they are guards, not pending rows.

Regime: CPU, pure, deterministic; Hypothesis 400 examples per property, ``database=None``.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_astra.toolcall import live, native, score, tools
from tests import _git_range as gr
from tests.unit import test_toolcall_live as tl
from tests.unit._toolcall_live_support import CLOSE, OPEN, Q

#: The literal base T-0196 branched from (docs/tasks/T-0196.md "Definitions": BASE is live.py at 611068bb).
T0196_BASE = tl.T0196_BASE
assert T0196_BASE == "611068bb"
#: ``llm.commit`` seq 6757, ``logs/sessions/20261029-121653-2974784.jsonl``, turn t1, request ``t1-gen-s7``, verbatim
INCIDENT = "<|tool_call>call:click_at{x:458,y:38}<tool_call|><tool_call|>"


def base() -> Any:
    return tl._live_at(T0196_BASE)


def as_tuple(v: Any) -> tuple[Any, ...]:
    return (v.status, v.name, v.arguments, v.reason, v.query_problem)


#: the shape walk (module docstring) lives beside the widened base-comparison properties it also serves
doubled_split = tl.t0196_doubled_split
single_close = tl.t0196_single_close


#: (single-close call, its name, its arguments written out by hand, tool set name)
VALID_CALLS: dict[str, tuple[str, str, dict[str, Any], str]] = {
    "click_at": (OPEN + "call:click_at{x:458,y:38}" + CLOSE, "click_at", {"x": 458, "y": 38}, "v1"),
    "type_at": (
        OPEN + f"call:type_at{{press_enter:false,text:{Q}ada@example.com{Q},x:305,y:448}}" + CLOSE,
        "type_at",
        {"press_enter": False, "text": "ada@example.com", "x": 305, "y": 448},
        "v1",
    ),
    "scroll": (
        OPEN + f"call:scroll{{direction:{Q}down{Q}}}" + CLOSE,
        "scroll",
        {"direction": "down"},
        "v1",
    ),
    "press_key": (
        OPEN + f"call:press_key{{combo:{Q}Ctrl+T{Q}}}" + CLOSE,
        "press_key",
        {"combo": "Ctrl+T"},
        "v1",
    ),
    "google_search": (
        OPEN + f"call:google_search{{query:{Q}weather in Paris{Q}}}" + CLOSE,
        "google_search",
        {"query": "weather in Paris"},
        "v1",
    ),
    "page_overview": (OPEN + "call:page_overview{}" + CLOSE, "page_overview", {}, "overview"),
}
WHITESPACE = ["", " ", "\n", " \n\t"]


def tool_set(which: str) -> Any:
    return None if which == "v1" else tools.load(tools.TOOLS_V1_OVERVIEW_PATH)


def doubled(single: str, extra: int, ws: str) -> str:
    """``single`` with ``extra`` more closes; ``ws`` before the first extra close, between closes and after the last."""
    return single + "".join(ws + CLOSE for _ in range(extra)) + ws


# --------------------------------------------------------------------------------------------- criterion 1


def test_incident_text_is_valid() -> None:
    """Criterion 1 (the incident, verbatim from seq 6757): ``valid``, ``click_at``, ``{"x":458,"y":38}``, no reason --
    BASE's verdict on the single-close text, field for field. BASE said ``repairable`` "text after call"."""
    v = live.classify(INCIDENT)
    assert as_tuple(v) == ("valid", "click_at", {"x": 458, "y": 38}, None, None), v
    assert as_tuple(v) == as_tuple(base().classify(INCIDENT[: -len(CLOSE)])), v


@pytest.mark.parametrize("ws", WHITESPACE, ids=["none", "space", "newline", "mixed"])
@pytest.mark.parametrize("extra", [1, 2, 3])
@pytest.mark.parametrize("tool", sorted(VALID_CALLS))
def test_doubled_close_equals_single_close(tool: str, extra: int, ws: str) -> None:
    """Criterion 1: for the five v1 tools and ``page_overview`` (T-0140 overview set), 1-3 extra closes with
    whitespace before, between and after, ``classify`` equals BASE's ``classify(single-close text)`` in every field
    (``valid``, the hand name and arguments, reason None)."""
    single, name, args, which = VALID_CALLS[tool]
    ts = tool_set(which)
    text = doubled(single, extra, ws)
    assert single_close(text) == single, "premise: the text has the shape"
    want = base().classify(single, tool_set=ts)
    assert (want.status, want.name, want.arguments, want.reason) == ("valid", name, args, None), (
        "premise: BASE judges the single-close text valid",
        want,
    )
    got = live.classify(text, tool_set=ts)
    assert as_tuple(got) == as_tuple(want), (text, got, want)


# --------------------------------------------------------------------------------------------- criterion 2

CLICK = VALID_CALLS["click_at"][0]
SIDEWAYS = OPEN + f"call:scroll{{direction:{Q}sideways{Q}}}" + CLOSE
#: criterion 2's negative rows: (text, a hand note of BASE's verdict status)
OTHER_TAILS: dict[str, str] = {
    "closes_then_dot": CLICK + CLOSE + ".",
    "closes_then_dot_space": CLICK + CLOSE + " .",
    "closes_then_channel_close": CLICK + CLOSE + "<channel|>",
    "closes_then_channel_open": CLICK + CLOSE + CLOSE + "<|channel>",
    "closes_then_turn_newline": CLICK + CLOSE + "<turn|>\n",
    "closes_then_word": CLICK + CLOSE + " ok",
    "closes_then_wait": CLICK + CLOSE + CLOSE + "\nWait",
    "closes_then_second_call": CLICK + CLOSE + OPEN + "call:click_at{x:1,y:2}" + CLOSE,
    "closes_then_second_tag": CLICK + CLOSE + " " + OPEN,
    "closes_then_quote": CLICK + CLOSE + Q,
    "closes_then_brace": CLICK + CLOSE + "}",
    "missing_brace_doubled": OPEN + "call:click_at{x:458,y:38" + CLOSE + CLOSE,
    "missing_brace_tripled": OPEN + "call:click_at{x:458,y:38" + CLOSE + " " + CLOSE + CLOSE,
    "x1200_doubled": OPEN + "call:click_at{x:1200,y:5}" + CLOSE + CLOSE,
    "ctrl_s_doubled": OPEN + f"call:press_key{{combo:{Q}Ctrl+S{Q}}}" + CLOSE + CLOSE,
    "sideways_doubled": SIDEWAYS + CLOSE,
    "sideways_tripled_ws": SIDEWAYS + "\n" + CLOSE + " " + CLOSE,
    "extra_key_doubled": OPEN + f"call:click_at{{x:12,y:5,button:{Q}left{Q}}}" + CLOSE + CLOSE,
    "navigate_doubled": OPEN + f"call:navigate{{url:{Q}x{Q}}}" + CLOSE + CLOSE,
    "overview_v1_doubled": OPEN + "call:page_overview{}" + CLOSE + CLOSE,
    "single_close_no_brace": OPEN + "call:click_at{x:458,y:38" + CLOSE,
    "no_close": OPEN + "call:click_at{x:458,y:38}",
    "no_close_ws": OPEN + "call:click_at{x:458,y:38} \n",
    "only_closes_after_tag": OPEN + CLOSE + CLOSE,
    "no_call_prefix": OPEN + "click_at{x:1,y:2}" + CLOSE + CLOSE,
}


@pytest.mark.parametrize("row", sorted(OTHER_TAILS))
def test_other_tails_keep_base_verdict(row: str) -> None:
    """Criterion 2 (negative; adversarial, the guard too wide): a text not of the shape, or of the shape but whose
    single-close text BASE does not judge ``valid`` (missing ``}``, T-0100's ``x:1200`` and ``Ctrl+S`` refusals with
    ``arguments=None``, the schema-invalid ``sideways`` scroll, an unknown name), gets BASE's ``classify(text)`` field
    for field; a close is never invented (no close, or a close inside an unclosed brace, is never ``valid``). Kills
    "strip everything after the first close" and "adopt the trimmed verdict when not valid". Guard: green at BASE."""
    text = OTHER_TAILS[row]
    want = base().classify(text)
    assert want.status != "valid", ("premise: BASE does not accept the row", want)
    got = live.classify(text)
    assert as_tuple(got) == as_tuple(want), (row, got, want)


def test_other_tails_premise_the_shape_rows() -> None:
    """Premise of criterion 2's second half: the doubled rows *have* the shape (so the rule is reached) and BASE's
    single-close verdict on them is not ``valid``; the tail rows do not have it."""
    shaped = {k for k, t in OTHER_TAILS.items() if single_close(t) is not None}
    assert shaped == {
        "missing_brace_doubled",
        "missing_brace_tripled",
        "x1200_doubled",
        "ctrl_s_doubled",
        "sideways_doubled",
        "sideways_tripled_ws",
        "extra_key_doubled",
        "navigate_doubled",
        "overview_v1_doubled",
        "only_closes_after_tag",
        "no_call_prefix",
    }, sorted(shaped)
    for k in shaped:
        single = single_close(OTHER_TAILS[k])
        assert single is not None and base().classify(single).status != "valid", k
    x1200 = base().classify(single_close(OTHER_TAILS["x1200_doubled"]) or "")
    assert (x1200.status, x1200.arguments) == ("refused", {"x": 1200, "y": 5}), (
        "premise: the single-close x:1200 refusal carries arguments; the doubled one must keep None",
        x1200,
    )


# --------------------------------------------------------------------------------------------- string contents

STRING_CLOSES = (
    OPEN + f"call:type_at{{text:{Q}a{CLOSE}{CLOSE}{Q},x:5,y:5,press_enter:false}}" + CLOSE
)
STRING_CLOSES_ARGS = {"text": "a" + CLOSE + CLOSE, "x": 5, "y": 5, "press_enter": False}
UNTERMINATED = OPEN + f"call:type_at{{text:{Q}a{CLOSE}{CLOSE}"


def test_close_inside_string_is_content() -> None:
    """Adversarial (string awareness): ``<tool_call|>`` inside ``<|"|>...<|"|>`` is string content. The call whose
    ``text`` holds two closes stays ``valid`` with that text; with one extra close after it, it is ``valid`` with the
    same arguments (a walk that took the first close inside the string would find prose after it and keep BASE's
    ``repairable``); an unterminated string followed by closes keeps BASE's verdict (no close outside a string)."""
    v = live.classify(STRING_CLOSES)
    assert (v.status, v.name, v.arguments) == ("valid", "type_at", STRING_CLOSES_ARGS), v
    assert single_close(STRING_CLOSES + CLOSE) == STRING_CLOSES, "premise"
    assert base().classify(STRING_CLOSES + CLOSE).status != "valid", "premise: BASE refuses it"
    for extra in (CLOSE, " " + CLOSE + "\n" + CLOSE):
        w = live.classify(STRING_CLOSES + extra)
        assert as_tuple(w) == as_tuple(base().classify(STRING_CLOSES)), (extra, w)
    for t in (UNTERMINATED, UNTERMINATED + CLOSE, UNTERMINATED + " " + CLOSE + CLOSE):
        assert single_close(t) is None, "premise: no close outside a string"
        assert as_tuple(live.classify(t)) == as_tuple(base().classify(t)), t


# --------------------------------------------------------------------------------------------- criterion 3

ALL_ROWS = [
    INCIDENT,
    *(
        doubled(VALID_CALLS[k][0], n, ws)
        for k in sorted(VALID_CALLS)
        for n in (1, 2, 3)
        for ws in WHITESPACE
    ),
    *OTHER_TAILS.values(),
    STRING_CLOSES,
    STRING_CLOSES + CLOSE,
    UNTERMINATED + CLOSE,
]


def test_repair_verdict_unchanged() -> None:
    """Criterion 3 (negative; H-T196-3, ``repair_verdict`` stays strict): on every row of this module, bare and behind
    a closed thought, and with the T-0140 overview set, ``repair_verdict`` equals BASE's field for field -- so a doubled
    close in a repair's output is still refused. Kills "change repair_verdict too". Guard: green at BASE."""
    b, ov = base(), tool_set("overview")
    bad = []
    for text in ALL_ROWS:
        for t in (text, "<|channel>thought\nok<channel|>" + text):
            for ts in (None, ov):
                got, want = live.repair_verdict(t, tool_set=ts), b.repair_verdict(t, tool_set=ts)
                if as_tuple(got) != as_tuple(want):
                    bad.append((t, ts is not None, as_tuple(want), as_tuple(got)))
    assert bad == []
    assert b.repair_verdict(INCIDENT).status == "refused", (
        "premise: the strict verdict refuses the incident"
    )


# --------------------------------------------------------------------------------------------- the property

_ALPHABET = [*tl._FRAGMENTS, CLOSE, " ", "\n"]
_BUILT = [VALID_CALLS[k][0] for k in sorted(VALID_CALLS) if VALID_CALLS[k][3] == "v1"]


@settings(max_examples=400, deadline=None, database=None)
@given(
    name=st.sampled_from(tl._NAMES),
    parts=st.lists(st.sampled_from(_ALPHABET), max_size=10),
    single=st.sampled_from(_BUILT),
    tail=st.lists(st.sampled_from([CLOSE, " ", "\n", "\t"]), min_size=1, max_size=6),
)
def test_property_only_the_delta_moves(
    name: str, parts: list[str], single: str, tail: list[str]
) -> None:
    """Criteria 1-3 as a property (Hypothesis, 400 examples over the T-0100 names and fragments plus closes and
    whitespace), against ``live.py`` at 611068bb. (a) For a fragment text, ``classify`` is BASE's, or else BASE's was
    not ``valid``, the text has the shape (the walk above) and the verdict is BASE's verdict on the single-close text,
    which is ``valid``; ``repair_verdict`` is BASE's field for field. (b) The positive direction, by construction: a
    valid v1 call followed by a run of closes and whitespace holding at least one close gets BASE's single-close
    verdict (BASE did not)."""
    b = base()
    text = OPEN + "call:" + name + "{" + "".join(parts)
    new, old = live.classify(text), b.classify(text)
    if as_tuple(new) != as_tuple(old):
        single_text = single_close(text)
        assert old.status != "valid" and single_text is not None, (text, old, new)
        want = b.classify(single_text)
        assert want.status == "valid" and as_tuple(new) == as_tuple(want), (text, old, new, want)
    assert as_tuple(live.repair_verdict(text)) == as_tuple(b.repair_verdict(text)), text

    if CLOSE not in tail:
        tail = [*tail, CLOSE]
    built = single + "".join(tail)
    assert single_close(built) == single, ("premise: the built text has the shape", built)
    want_b = b.classify(single)
    assert want_b.status == "valid", ("premise", want_b)
    assert as_tuple(live.classify(built)) == as_tuple(want_b), (built, live.classify(built))
    assert as_tuple(live.repair_verdict(built)) == as_tuple(b.repair_verdict(built)), built


def test_doubled_split_oracle_hand_rows() -> None:
    """The test's own walk, against hand-written cases (so the property's oracle is not vacuous)."""
    assert doubled_split("call:a{}" + CLOSE + CLOSE) == ("call:a{}" + CLOSE, CLOSE)
    assert doubled_split("call:a{}" + CLOSE + " \n" + CLOSE + "\t") == (
        "call:a{}" + CLOSE,
        " \n" + CLOSE + "\t",
    )
    assert doubled_split("call:a{}" + CLOSE) is None  # T empty
    assert doubled_split("call:a{}" + CLOSE + "  ") is None  # no extra close
    assert doubled_split("call:a{}" + CLOSE + CLOSE + ".") is None
    assert doubled_split("call:a{}") is None
    s = f"call:a{{t:{Q}{CLOSE}{Q}}}"
    assert doubled_split(s + CLOSE + CLOSE) == (s + CLOSE, CLOSE)
    assert doubled_split(f"call:a{{t:{Q}{CLOSE}{CLOSE}") is None


# --------------------------------------------------------------------------------------------- criterion 9


def test_native_parse_call_untouched() -> None:
    """Criterion 9: ``native.parse_call`` on the incident body is still ``ParseFailure("text after call")`` (the eval
    parser, ReGround and the cd oracles keep the strict reading). Guard: green at BASE."""
    got = native.parse_call(INCIDENT[len(OPEN) :])
    assert got == score.ParseFailure("text after call"), got


#: Criterion 9's frozen paths (docs/tasks/T-0196.md), relative to the repo root.
FROZEN = (
    "local_astra/toolcall/native.py",
    "local_astra/toolcall/router.py",
    "local_astra/toolcall/loop.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/tools_v1_overview.json",
    "local_astra/toolcall/payloads",
    "local_astra/toolcall/reground.py",
    "local_astra/toolcall/thought_call.py",
    "local_astra/conductor/session.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/conductor/turn_machine.py",
    "local_astra/eval",
    "local_astra/decoding",
    "local_astra/contracts",
    "tests/fakes",
    "plugin",
    "services",
    "docs/cd",
)
#: T-0196's own final commit: M^2 of the merge d4fa820f (``Merge T-0196``). The branch has no merge from main, and
#: ``merge-base M^1 M^2`` (a9858a5c, the planning commit) descends from BASE with no frozen path changed in between,
#: so BASE..TIP is exactly T-0196's work (tests.md: after merge a scope guard compares ``BASE..TIP``, both literals;
#: later tasks such as T-0195 A may legitimately edit these paths).
T0196_TIP = "d0363212"
#: The power row's range: T-0193 A's merge changed ``session.py`` and ``contracts/protocol.py``.
POWER = ("8ca5592a^1", "8ca5592a")


def test_scope_byte_identical() -> None:
    """Criterion 9 (negative): T-0196 (BASE 611068bb .. its tip d0363212) changed no frozen path."""
    gr.require_commits(T0196_BASE, T0196_TIP)
    assert gr.changed(T0196_BASE, T0196_TIP, *FROZEN) == []


def test_scope_guard_has_power() -> None:
    """The scope row's power: the same path set over a range known to edit two of its files reports them."""
    gr.require_commits(POWER[1])
    moved = gr.changed(*POWER, *FROZEN)
    assert (
        "local_astra/conductor/session.py" in moved and "local_astra/contracts/protocol.py" in moved
    ), moved
    assert len(FROZEN) == 18
