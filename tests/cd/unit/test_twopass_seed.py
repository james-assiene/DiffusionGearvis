"""T-0081b criteria 2 and 5: ``twopass.seed_ids`` (the draft seed of a ``perstep_mar+seed`` pass-2 cell) and the
``executed_steps`` fix, against the real pinned tokenizer and the real ``compile_toolcall`` automata.

The seed, as docs/tasks/T-0081b.md specifies it (every expected value below is built from that formula by hand,
with ``vocab.encode`` -- the pinned HF tokenizer, no BOS -- as the only oracle for tokenisation)::

    seed(thought) = (H(thought) + encode(T(thought) + draft_call_text) + S)[:CL]

- H(T1) = pass-1 ``token_ids`` verbatim up to and including the first ``101``, if they start with ``100`` and
  contain a ``101``; otherwise the empty-thought header ``(100, 45518, 107, 101)``. H(T2) = that header. H(T4) = ().
- T(T1) = T(T2) = ``"<cc>"``; T(T4) = ``""``.
- S = ``[s]``, ``s`` the first stop id (``replay.STOP_IDS``: 1, 50, 106) in the pass-1 ids; empty without one.
- CL = 256 (the engine-global canvas).

The alignment proof (criterion 2) replays the seed through ``compile_toolcall(scope=<tool>, thought=<thought>)``
with ``replay.accepts``. The adversarial controls are the four wrong constructions the task names; each row checks
that the oracle rejects the control *and* that ``seed_ids`` returns the accepted spec seed instead.

Regime: CPU; pinned tokenizer (model rev 2ea8372); ``tools_v1.json`` at its pinned sha; JSON whitespace; CL 256.
Pass-1 ids are ``vocab.encode(text)`` plus a stop id the row chooses (106, ``<turn|>``, unless stated), so a
seed whose S is a constant EOS is caught.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from tests.cd import _twopass_support as ts

CL = 256
HEADER = [100, 45518, 107, 101]  # <|channel> thought \n <channel|>
TURN_CLOSE, EOS, TOOL_RESPONSE = 106, 1, 50
SPOKEN = "Sure, doing it. "
#: one schema-valid call per v1 tool (keys in declared order, ``json.dumps`` spacing: the drafts' own format)
CALLS: dict[str, dict[str, Any]] = {
    "google_search": {"query": "british museum hours"},
    "click_at": {"x": 500, "y": 380},
    "type_at": {"x": 500, "y": 300, "text": "usb-c hub", "press_enter": True},
    "scroll": {"direction": "down"},
    "press_key": {"combo": "Ctrl+Z"},
}


# --------------------------------------------------------------------------- helpers (oracles, not the code under test)


def encode(text: str) -> list[int]:
    from local_astra.decoding import vocab

    return [int(t) for t in vocab.encode(text)]


def accepts(scope: str, thought: str, ids: Sequence[int]) -> bool:
    from local_astra.eval.cd import replay

    return bool(replay.accepts(ts.compiled(scope, thought), list(ids)))


def seed_ids(
    thought: str, pass1_ids: Sequence[int], draft_call_text: str, canvas_length: int = CL
) -> list[int]:
    """``twopass.seed_ids`` (criterion 2's signature, positional). Its absence is an assertion, not an error."""
    tp = ts.twopass()
    fn = getattr(tp, "seed_ids", None)
    if fn is None:
        raise AssertionError("T-0081b not implemented: twopass.seed_ids is missing")
    out = fn(thought, list(pass1_ids), draft_call_text, canvas_length)
    return [int(t) for t in out]


def executed_steps(steps: Sequence[int], k0: int) -> list[int]:
    return [int(s) for s in ts.twopass().executed_steps(list(steps), k0)]


def call_of(tool: str) -> str:
    return ts.call_text(tool, CALLS[tool])


def pass1(tool: str, *, thought: str | None = None, stop: int | None = TURN_CLOSE) -> list[int]:
    """Pass-1 ids: ``<|channel>thought\\n<thought><channel|>`` + spoken prefix + ``<cc>`` + the call, then ``stop``.
    ``thought=None`` uses a short default thought."""
    body = thought if thought is not None else f"The user wants {tool}."
    text = f"<|channel>thought\n{body}<channel|>{SPOKEN}<cc>" + call_of(tool)
    return encode(text) + ([stop] if stop is not None else [])


def through_first_101(ids: Sequence[int]) -> list[int]:
    i = list(ids).index(101)
    return list(ids[: i + 1])


def spec_seed(
    thought: str, tool: str, ids: Sequence[int], stop: int | None = TURN_CLOSE
) -> list[int]:
    """The criterion-2 seed for a pass-1 that opens with a closed thought, built by hand from the formula."""
    tail = [stop] if stop is not None else []
    if thought == "T1":
        return (through_first_101(ids) + encode("<cc>" + call_of(tool)) + tail)[:CL]
    if thought == "T2":
        return (HEADER + encode("<cc>" + call_of(tool)) + tail)[:CL]
    return (encode(call_of(tool)) + tail)[:CL]


def test_premises_of_the_oracle() -> None:
    """The fixtures are what the rows assume: the empty-thought header tokenises to ``(100, 45518, 107, 101)``
    (``decoding/report.py``'s ``_PHASE0_HEADER``), every call is schema-valid, and the router reads exactly the
    call as the draft of each pass-1 text (so ``draft_call_text`` in the rows is the router's)."""
    from local_astra.decoding import report
    from local_astra.eval.cd import replay
    from local_astra.toolcall import router, score

    assert encode("<|channel>thought\n<channel|>") == HEADER
    assert list(report._PHASE0_HEADER) == HEADER
    assert {EOS, TOOL_RESPONSE, TURN_CLOSE} == replay.STOP_IDS
    for tool in ts.TOOLS:
        parsed = score.parse_call(call_of(tool))
        assert not isinstance(parsed, score.ParseFailure), tool
        assert score.schema_valid(parsed, ts.toolset()), tool
        ids = pass1(tool)
        r = router.route(replay.detokenize(ids[:-1]))
        assert (r.kind, r.draft_name, r.draft_call_text) == ("toolcall", tool, call_of(tool)), r


# =========================================================================== criterion 2: alignment proof


@pytest.mark.parametrize("thought", ["T1", "T2", "T4"])
@pytest.mark.parametrize("tool", list(CALLS))
def test_the_seed_is_the_spec_seed_and_the_thoughts_grammar_accepts_it(
    tool: str, thought: str
) -> None:
    """Criterion 2 (alignment proof), 5 tools x 3 thoughts: a schema-valid draft ending on a stop id gives exactly
    the hand-built seed, and ``replay.accepts(compile_toolcall(scope=<tool>, thought=<thought>), seed)``."""
    ids = pass1(tool)
    got = seed_ids(thought, ids, call_of(tool))
    want = spec_seed(thought, tool, ids)
    assert got == want, (tool, thought, got, want)
    assert accepts(tool, thought, got), f"{tool}/{thought}: the grammar rejects the seed {got}"


@pytest.mark.parametrize("stop", [EOS, TOOL_RESPONSE, TURN_CLOSE])
def test_s_is_the_first_stop_id_of_pass_1_not_a_constant(stop: int) -> None:
    """S = the first stop id in pass-1 ``token_ids``, whichever of the three it is; a stop after it (here EOS,
    then ``<turn|>``) is not used. Each seed is still accepted (every stop id ends the language)."""
    tool = "scroll"
    ids = [*pass1(tool, stop=stop), EOS, TURN_CLOSE]
    for thought in ("T1", "T2", "T4"):
        got = seed_ids(thought, ids, call_of(tool))
        assert got == spec_seed(thought, tool, ids, stop=stop), (thought, stop, got)
        assert got[-1] == stop
        assert accepts(tool, thought, got), (thought, stop)


@pytest.mark.parametrize("thought", ["T1", "T2", "T4"])
def test_a_truncated_pass_1_has_no_s(thought: str) -> None:
    """Criterion 2/4: pass 1 that ended without a stop (out of budget) gives a seed without S -- no stop id is
    appended. The draft is the router's truncated text, used raw."""
    from local_astra.eval.cd import replay

    tool = "type_at"
    full = pass1(tool, stop=None)
    ids = full[:-6]  # cut inside the call
    draft = replay.detokenize(ids).split("<cc>", 1)[1]
    assert draft and call_of(tool).startswith(draft) and draft != call_of(tool), draft
    got = seed_ids(thought, ids, draft)
    head = {"T1": through_first_101(ids), "T2": HEADER, "T4": []}[thought]
    tag = "" if thought == "T4" else "<cc>"
    assert got == head + encode(tag + draft.rstrip()), (
        thought,
        got,
    )  # R2: trailing whitespace trimmed
    assert not replay.STOP_IDS & set(got), got


# =========================================================================== H(T1)


def test_h_t1_is_pass_1_verbatim_even_when_it_does_not_re_tokenise_to_itself() -> None:
    """H(T1) copies pass-1 ids, never a re-tokenisation of their text: the thought's ``The`` (818) is sent as
    ``T`` + ``he`` (236774, 499), which ``encode`` would merge back. The prompt is pass 1's, so these positions
    line up exactly, and the T1 grammar takes any id inside the thought."""
    tool = "click_at"
    canonical = pass1(tool)
    i = canonical.index(818)
    assert i == 3, canonical[:6]  # premise: the thought's first token is "The"
    assert encode("T") == [236774] and encode("he") == [499]
    ids = [*canonical[:i], 236774, 499, *canonical[i + 1 :]]
    got = seed_ids("T1", ids, call_of(tool))
    want = through_first_101(ids) + encode("<cc>" + call_of(tool)) + [TURN_CLOSE]
    assert got == want, got
    assert got[:6] == [100, 45518, 107, 236774, 499, ids[5]]
    assert accepts(tool, "T1", got)


def test_h_t1_longer_than_the_canvas_is_cut_at_cl() -> None:
    """A thought longer than CL: the seed is the first 256 pass-1 ids, verbatim (no tag, no call, no S fits)."""
    tool = "google_search"
    ids = pass1(tool, thought="Let me think about which search query fits best. " * 40)
    assert ids.index(101) > CL, "premise: the thought is longer than the canvas"
    got = seed_ids("T1", ids, call_of(tool))
    assert got == ids[:CL], (len(got), got[:8])


@pytest.mark.parametrize(
    "case",
    ["no_thought", "unclosed_thought", "101_without_a_leading_100"],
)
def test_h_t1_falls_back_to_the_empty_header(case: str) -> None:
    """The T1 fallback: pass-1 ids that do not start with ``100``, or contain no ``101``, give the empty-thought
    header ``(100, 45518, 107, 101)``; the seed is then exactly T2's. ``no_thought``: the model answered without a
    thought; ``unclosed_thought``: ``100`` but no ``101``; ``101_without_a_leading_100``: a ``101`` but the ids open
    with the spoken text (both conditions are needed)."""
    tool = "press_key"
    call = call_of(tool)
    if case == "no_thought":
        ids = [*encode(SPOKEN + "<cc>" + call), TURN_CLOSE]
    elif case == "unclosed_thought":
        ids = [*encode("<|channel>thought\nI could press it " + "<cc>" + call), TURN_CLOSE]
        assert 100 in ids and 101 not in ids
    else:
        ids = [*encode(SPOKEN + "<|channel>thought\nok<channel|><cc>" + call), TURN_CLOSE]
        assert ids[0] != 100 and 101 in ids
    got = seed_ids("T1", ids, call)
    want = HEADER + encode("<cc>" + call) + [TURN_CLOSE]
    assert got == want, (case, got)
    assert accepts(tool, "T1", got)


def test_h_t1_does_not_fall_back_when_pass_1_has_a_thought() -> None:
    """The other side of the fallback (too wide): a pass 1 that opens with a closed, non-empty thought keeps it --
    the seed is not the empty header."""
    tool = "press_key"
    ids = pass1(tool)
    got = seed_ids("T1", ids, call_of(tool))
    assert got[:4] != HEADER and got[: ids.index(101) + 1] == through_first_101(ids), got[:10]


# =========================================================================== the cut at CL


@pytest.mark.parametrize("thought", ["T2", "T4"])
def test_the_cut_at_cl_is_exact(thought: str) -> None:
    """``[:CL]`` with CL at, one below and well below the full seed length: the full seed at its own length, S
    dropped one below, the first 10 ids at CL 10."""
    tool = "type_at"
    ids = pass1(tool)
    full = spec_seed(thought, tool, ids)
    n = len(full)
    assert seed_ids(thought, ids, call_of(tool), n) == full
    assert seed_ids(thought, ids, call_of(tool), n + 5) == full
    assert seed_ids(thought, ids, call_of(tool), n - 1) == full[:-1]
    assert seed_ids(thought, ids, call_of(tool), 10) == full[:10]


# =========================================================================== the draft is re-tokenised, raw


def test_the_call_is_re_tokenised_not_sliced_from_pass_1() -> None:
    """The pass-2 canvas starts at a fresh token boundary: the call part is ``encode`` of the draft text, never pass
    1's ids after the tag. Here pass 1 spelled ``{"`` (14937) as ``{`` + ``"`` (236782, 236775): a slice would carry
    the split pair, the spec seed carries the merged id."""
    tool = "scroll"
    canonical = pass1(tool)
    j = canonical.index(14937)
    assert encode("{") == [236782] and encode('"') == [236775]
    ids = [*canonical[:j], 236782, 236775, *canonical[j + 1 :]]
    for thought in ("T2", "T4"):
        got = seed_ids(thought, ids, call_of(tool))
        assert got == spec_seed(thought, tool, ids), (thought, got)
        assert 14937 in got and got[got.index(14937) - 2 : got.index(14937)] != [236782, 236775]


@pytest.mark.parametrize(
    "draft",
    [
        '{"name": "click_at", "arguments": {"x": 1500, "y": 380}}',  # out of range
        '{"name": "google_search", "arguments": {"query": "museum", "lang": "en"}}',  # unknown argument
        '{"name":"click_at","arguments":{"y":380,"x":500}}',  # re-ordered keys, no spaces
        ' {"name": "scroll", "arguments": {"direction": "down"}} Done.',  # leading space, trailing prose
    ],
    ids=["x1500", "unknown_arg", "reordered", "trailing_prose"],
)
def test_the_draft_text_is_used_raw(draft: str) -> None:
    """Criterion 4 (no host repair), at the function: whatever the draft -- out of schema, unknown argument,
    keys out of order, odd spacing, prose after the call -- the seed is ``encode`` of its raw text (T4) or of
    ``"<cc>" + raw`` (T2), never a parsed, clamped, re-ordered or re-serialised call."""
    ids = [*encode(SPOKEN + "<cc>" + draft), TURN_CLOSE]
    assert seed_ids("T4", ids, draft) == [*encode(draft), TURN_CLOSE]
    assert seed_ids("T2", ids, draft) == HEADER + encode("<cc>" + draft) + [TURN_CLOSE]


# =========================================================================== adversarial controls (the oracle discriminates)


def _control(name: str, tool: str) -> tuple[str, list[int], list[int]]:
    """(thought, control ids the grammar must reject, pass-1 ids)."""
    ids = pass1(tool)
    call = call_of(tool)
    if name == "t4_with_leading_cc":
        return "T4", [*encode("<cc>" + call), TURN_CLOSE], ids
    if name == "t2_without_header":
        return "T2", [*encode("<cc>" + call), TURN_CLOSE], ids
    if name == "t1_keeping_the_spoken_prefix":
        return "T1", through_first_101(ids) + encode(SPOKEN + "<cc>" + call) + [TURN_CLOSE], ids
    if name == "t2_with_pass_1s_full_thought":
        return "T2", through_first_101(ids) + encode("<cc>" + call) + [TURN_CLOSE], ids
    raise AssertionError(name)


@pytest.mark.parametrize("tool", list(CALLS))
@pytest.mark.parametrize(
    "name",
    [
        "t4_with_leading_cc",
        "t2_without_header",
        "t1_keeping_the_spoken_prefix",
        "t2_with_pass_1s_full_thought",
    ],
)
def test_wrong_alignments_are_rejected_and_seed_ids_avoids_them(name: str, tool: str) -> None:
    """Adversarial (the failure mode: a misaligned seed). The four wrong constructions the task names -- T4 with a
    leading ``<cc>``, T2 without the header, T1 keeping the spoken prefix before ``<cc>``, T2 with pass 1's full
    thought -- are each rejected by the thought's own grammar (so the accepts rows above can tell them apart), and
    ``seed_ids`` returns the accepted spec seed, not the control."""
    thought, control, ids = _control(name, tool)
    assert not accepts(tool, thought, control), (
        f"control {name} is accepted: the oracle cannot discriminate"
    )
    got = seed_ids(thought, ids, call_of(tool))
    assert got != control, name
    assert got == spec_seed(thought, tool, ids)
    assert accepts(tool, thought, got)


# =========================================================================== criterion 5: executed steps


def test_executed_steps_subtracts_k0_from_the_first_canvas_only() -> None:
    """Criterion 5 (T-0081 c4-5: only the first canvas is seeded, later canvases restart at step 0):
    ``[30]`` at k0 24 -> ``[6]``; ``[30, 12, 7]`` at k0 24 -> ``[6, 12, 7]`` (subtracting from every canvas would give
    ``[6, -12, -17]``); k0 0 leaves every list unchanged."""
    assert executed_steps([30], 24) == [6]
    assert executed_steps([30, 12, 7], 24) == [6, 12, 7]
    assert executed_steps([30, 12, 7], 0) == [30, 12, 7]
    assert executed_steps([30], 0) == [30]
    assert [int(s) for s in ts.twopass().executed_steps([30, 12, 7])] == [
        30,
        12,
        7,
    ]  # k0 defaults to 0


def test_executed_steps_stays_total() -> None:
    """``executed_steps`` stays the one place and total (implementation note): no records -> ``[]``; a canvas at or
    below k0 is returned as computed (0, -4), never raised on -- the ``steps <= k0`` check is a stop (criterion
    6a), not this function's."""
    assert executed_steps([], 24) == []
    assert executed_steps([24, 30], 24) == [0, 30]
    assert executed_steps([20], 24) == [-4]


def test_the_rows_json_round_trip() -> None:
    """``seed_ids`` returns plain ints (the row stores them as JSON)."""
    tool = "google_search"
    tp = ts.twopass()
    fn = getattr(tp, "seed_ids", None)
    if fn is None:
        raise AssertionError("T-0081b not implemented: twopass.seed_ids is missing")
    out = fn("T4", pass1(tool), call_of(tool), CL)
    assert json.loads(json.dumps(list(out))) == spec_seed("T4", tool, pass1(tool))


# =========================================================================== ruling R2: trailing whitespace trimmed

TRAILING = {"nl": "\n", "sp_nl": " \n", "nl_nl": "\n\n"}


@pytest.mark.parametrize("ws", list(TRAILING))
@pytest.mark.parametrize("thought", ["T1", "T2", "T4"])
@pytest.mark.parametrize("tool", list(CALLS))
def test_trailing_whitespace_is_trimmed_and_the_seed_is_accepted(
    tool: str, thought: str, ws: str
) -> None:
    """R2 (criterion 2 extended), 5 tools x 3 thoughts x 3 endings: a schema-valid draft ending in ``"\\n"``,
    ``" \\n"`` or ``"\\n\\n"`` before the stop gives the seed of the trimmed call, which the grammar accepts; the
    untrimmed seed (the control) is rejected -- no grammar admits whitespace between the call and the stop."""
    tail = TRAILING[ws]
    call = call_of(tool)
    text = f"<|channel>thought\nThe user wants {tool}.<channel|>{SPOKEN}<cc>" + call + tail
    ids = [*encode(text), TURN_CLOSE]
    got = seed_ids(thought, ids, call + tail)
    head = {"T1": through_first_101(ids), "T2": HEADER, "T4": []}[thought]
    tag = "" if thought == "T4" else "<cc>"
    want = [*head, *encode(tag + call), TURN_CLOSE]
    assert got == want, (tool, thought, ws, got[-6:], want[-6:])
    assert accepts(tool, thought, got), (tool, thought, ws)
    control = [*head, *encode(tag + call + tail), TURN_CLOSE]
    assert control != want and not accepts(tool, thought, control), (tool, thought, ws)


def test_leading_and_inner_whitespace_is_kept() -> None:
    """R2: only ``str.rstrip()`` -- a leading space, a double space inside the call and a tab inside it stay."""
    draft = ' {"name":  "scroll",\t"arguments": {"direction": "down"}} \n'
    ids = [*encode(SPOKEN + "<cc>" + draft), TURN_CLOSE]
    kept = ' {"name":  "scroll",\t"arguments": {"direction": "down"}}'
    assert seed_ids("T4", ids, draft) == [*encode(kept), TURN_CLOSE]
    assert seed_ids("T2", ids, draft) == [*HEADER, *encode("<cc>" + kept), TURN_CLOSE]
    assert seed_ids("T4", ids, draft) != [*encode(draft.strip()), TURN_CLOSE]


def test_an_invalid_draft_is_unrepaired_apart_from_the_trim() -> None:
    """R2 + criterion 4: x 1500 and an unknown argument stay as written; only the trailing newline goes."""
    for draft in (
        '{"name": "click_at", "arguments": {"x": 1500, "y": 380}}\n',
        '{"name": "google_search", "arguments": {"query": "museum", "lang": "en"}} \n\n',
    ):
        ids = [*encode(SPOKEN + "<cc>" + draft), TURN_CLOSE]
        assert seed_ids("T4", ids, draft) == [*encode(draft.rstrip()), TURN_CLOSE], draft


@pytest.mark.parametrize("thought", ["T1", "T2", "T4"])
def test_a_truncated_draft_ending_in_whitespace_is_trimmed_and_gets_no_s(thought: str) -> None:
    """R2: pass 1 cut by its budget right after ``"x": 500, `` (no stop): the seed is the trimmed draft (ending at
    the comma) and carries no stop id."""
    draft = '{"name": "type_at", "arguments": {"x": 500, '
    ids = encode(f"<|channel>thought\nThe user wants type_at.<channel|>{SPOKEN}<cc>" + draft)
    assert not {1, 50, 106} & set(ids)
    got = seed_ids(thought, ids, draft)
    head = {"T1": through_first_101(ids), "T2": HEADER, "T4": []}[thought]
    tag = "" if thought == "T4" else "<cc>"
    assert got == [*head, *encode(tag + draft.rstrip())], (thought, got[-6:])
    assert got != [*head, *encode(tag + draft)]
    assert not {1, 50, 106} & set(got)


# =========================================================================== T-0082a criterion 8: seeds follow the strip

CLOSE_ENDINGS = {"close": "</cc>", "close_nl": "</cc>\n"}


@pytest.mark.parametrize("ending", list(CLOSE_ENDINGS))
@pytest.mark.parametrize("thought", ["T1", "T2", "T4"])
@pytest.mark.parametrize("tool", list(CALLS))
def test_a_draft_closed_by_cc_gives_a_seed_without_the_close(
    tool: str, thought: str, ending: str
) -> None:
    """T-0082a criterion 8 (T-0081b criterion 2 and R2), 5 tools x 3 thoughts x {``}</cc>``, ``}</cc>\\n``} + stop:
    the driver's path -- decode pass 1 up to its stop, route it, seed from ``Route.draft_call_text`` with the
    *unchanged* ``seed_ids`` -- gives the hand-built seed of the bare call, whose decoded text holds no ``</cc>``
    and which the thought's grammar accepts. The raw seed, built with the close (the T-0080c behaviour), is the
    control: the grammar rejects it, so the accept above is the strip's doing."""
    from local_astra.eval.cd import replay
    from local_astra.toolcall import router

    tail = CLOSE_ENDINGS[ending]
    call = call_of(tool)
    text = f"<|channel>thought\nThe user wants {tool}.<channel|>{SPOKEN}<cc>" + call + tail
    ids = [*encode(text), TURN_CLOSE]
    assert replay.detokenize(ids[:-1]) == text, "premise: the pass-1 text round-trips"
    draft = router.route(replay.detokenize(ids[:-1])).draft_call_text
    got = seed_ids(thought, ids, draft)
    head = {"T1": through_first_101(ids), "T2": HEADER, "T4": []}[thought]
    tag = "" if thought == "T4" else "<cc>"
    want = [*head, *encode(tag + call), TURN_CLOSE]
    assert got == want, (tool, thought, ending, got[-8:], want[-8:])
    assert "</cc>" not in replay.detokenize(got), replay.detokenize(got)[-40:]
    assert accepts(tool, thought, got), (tool, thought, ending)
    raw = [*head, *encode(tag + (call + tail).rstrip()), TURN_CLOSE]
    assert raw != want and not accepts(tool, thought, raw), (tool, thought, ending)


# =========================================================================== T-0084c criterion 5: native seeds (QN6)
#
#     seed = (H(thought) + encode(T + draft_call_text.rstrip()) + S)[:CL],  T = "<|tool_call>" (T1/T2), "" (T4)
#
# H and S as in T-0081b. Pass 1 is the pinned template's own model turn for the call (``tests/cd/oracle/
# gemma4_template.model_turn``: ``<|channel>thought\n{reasoning}\n<channel|><|tool_call>call:...<tool_call|>`` then
# ``<|tool_response>``, stop id 50) with a spoken sentence inserted before ``<|tool_call>``, as a native pass 1 writes
# it. The draft is the router's text after ``<|tool_call>`` (ending in ``<tool_call|>``, id 49). The grammars are
# ``compile_toolcall(scope=<tool>, thought=<T>, call_format="native")``. API pin (test-author, T-0084c): the format
# is the keyword ``fmt`` of ``seed_ids`` (``"cc"``, the default, or ``"native"``), in the rows' own vocabulary.

NATIVE_CALLS: dict[str, dict[str, Any]] = {
    "google_search": {"query": "british museum hours"},
    "click_at": {"x": 500, "y": 380},
    "type_at": {"x": 500, "y": 300, "text": "usb-c hub", "press_enter": True},
    "scroll": {"direction": "down"},
    "press_key": {"combo": "Ctrl+Z"},
}
REASONING = "The user wants an action.\nI will call one tool."
OPEN, CLOSE = "<|tool_call>", "<tool_call|>"
CALL_OPEN_ID, CALL_CLOSE_ID = 48, 49


def seed_ids_fmt(
    thought: str,
    pass1_ids: Sequence[int],
    draft_call_text: str,
    canvas_length: int = CL,
    *,
    fmt: str,
) -> list[int]:
    """``twopass.seed_ids(..., fmt=fmt)``; a missing ``fmt`` keyword is an assertion, not a ``TypeError``."""
    import inspect

    fn = getattr(ts.twopass(), "seed_ids", None)
    assert fn is not None, "twopass.seed_ids is missing"
    if "fmt" not in inspect.signature(fn).parameters:
        raise AssertionError("T-0084c not implemented: twopass.seed_ids has no fmt keyword")
    return [int(t) for t in fn(thought, list(pass1_ids), draft_call_text, canvas_length, fmt=fmt)]


def accepts_native(scope: str, thought: str, ids: Sequence[int]) -> bool:
    from local_astra.eval.cd import replay

    return bool(replay.accepts(ts.compiled_native(scope, thought), list(ids)))


def native_pass1(
    tool: str, *, reasoning: str = REASONING, spoken: str = SPOKEN
) -> tuple[str, list[int]]:
    """(the pass-1 text up to and including the stop, its ids) from the template oracle."""
    from tests.cd.oracle import gemma4_template as g

    turn = g.model_turn(tool, NATIVE_CALLS[tool], reasoning)
    i = turn.index(OPEN)
    text = turn[:i] + spoken + turn[i:]
    return text, encode(text)


def native_draft(tool: str) -> str:
    """The template's call after ``<|tool_call>`` (what the router's ``draft_call_text`` is, see the premise row)."""
    return ts.native_call(tool, NATIVE_CALLS[tool])[len(OPEN) :]


def native_spec_seed(
    thought: str, ids: Sequence[int], draft: str, stop: int | None = 50
) -> list[int]:
    tail = [stop] if stop is not None else []
    if thought == "T1":
        return (through_first_101(ids) + encode(OPEN + draft.rstrip()) + tail)[:CL]
    if thought == "T2":
        return (HEADER + encode(OPEN + draft.rstrip()) + tail)[:CL]
    return (encode(draft.rstrip()) + tail)[:CL]


def test_t0084c_premises_of_the_native_oracle() -> None:
    """The markers are single ids (48, 49), the template turn ends in ``<|tool_response>`` (50), the router's draft
    of the native pass 1 is the template call after 48 (tag ``native``), and each call parses as schema-valid with
    ``native.parse_call``."""
    from local_astra.eval.cd import replay
    from local_astra.toolcall import native, router, score

    assert encode(OPEN) == [CALL_OPEN_ID] and encode(CLOSE) == [CALL_CLOSE_ID]
    assert encode("<|tool_response>") == [TOOL_RESPONSE]
    for tool in NATIVE_CALLS:
        _text, ids = native_pass1(tool)
        assert ids[-1] == TOOL_RESPONSE and ids.count(TOOL_RESPONSE) == 1, tool
        assert ids[0] == 100 and 101 in ids, tool
        r = router.route(replay.detokenize(ids[:-1]))
        assert (r.kind, r.tag, r.draft_call_text) == ("toolcall", "native", native_draft(tool)), r
        assert r.spoken_prefix == SPOKEN
        parsed = native.parse_call(native_draft(tool))
        assert not isinstance(parsed, score.ParseFailure), (tool, parsed)
        assert score.schema_valid(parsed, ts.toolset()), tool


@pytest.mark.parametrize("thought", ["T1", "T2", "T4"])
@pytest.mark.parametrize("tool", list(NATIVE_CALLS))
def test_t0084c_the_native_seed_is_the_spec_seed_and_the_native_grammar_accepts_it(
    tool: str, thought: str
) -> None:
    """Criterion 5 alignment proof, native x T1/T2/T4 x 5 tools: a template-rendered draft ending in 49 with stop 50
    gives exactly the hand-built seed, and the native grammar of that tool and thought accepts it."""
    _text, ids = native_pass1(tool)
    draft = native_draft(tool)
    want = native_spec_seed(thought, ids, draft)
    assert accepts_native(tool, thought, want), (
        "oracle premise: the spec seed is in the native language"
    )
    got = seed_ids_fmt(thought, ids, draft, fmt="native")
    assert got == want, (thought, tool, got[:12], want[:12])
    assert want[-2:] == [CALL_CLOSE_ID, TOOL_RESPONSE]
    if thought in ("T1", "T2"):
        k = want.index(101)
        assert want[k + 1] == CALL_OPEN_ID, "nothing between 101 and 48"
    else:
        assert CALL_OPEN_ID not in want, "the T4 prompt already ends in 48"


def test_t0084c_native_t1_keeps_the_multi_line_thought_verbatim() -> None:
    """H(T1) is pass 1's own thought, through the first 101, also for the native format (a real multi-line thought)."""
    _text, ids = native_pass1("click_at", reasoning="Line one.\nLine two.\n\nLine four.")
    got = seed_ids_fmt("T1", ids, native_draft("click_at"), fmt="native")
    assert got[: ids.index(101) + 1] == ids[: ids.index(101) + 1]
    assert accepts_native("click_at", "T1", got)


@pytest.mark.parametrize("thought", ["T1", "T2", "T4"])
def test_t0084c_a_truncated_native_pass_1_has_no_s(thought: str) -> None:
    """S is pass 1's first stop id; a native pass 1 cut by its budget (no 50) gives a seed without one."""
    _text, ids = native_pass1("scroll")
    cut = ids[:-1]
    draft = native_draft("scroll")
    got = seed_ids_fmt(thought, cut, draft, fmt="native")
    assert got == native_spec_seed(thought, cut, draft, stop=None)
    assert not set(got) & {EOS, TOOL_RESPONSE, TURN_CLOSE}


def _native_control(name: str, thought: str, tool: str) -> tuple[list[int], list[int], str]:
    _text, ids = native_pass1(tool)
    draft = native_draft(tool)
    head = {"T1": through_first_101(ids), "T2": HEADER, "T4": []}[thought]
    lead = {"T1": OPEN, "T2": OPEN, "T4": ""}[thought]
    if name == "spoken_prefix_kept":
        bad = head + encode(SPOKEN + lead + draft) + [TOOL_RESPONSE]
    else:  # cc_in_place_of_48
        bad = head + encode("<cc>" + draft) + [TOOL_RESPONSE]
    return ids, bad, draft


@pytest.mark.parametrize("tool", list(NATIVE_CALLS))
@pytest.mark.parametrize("thought", ["T1", "T2", "T4"])
@pytest.mark.parametrize("name", ["spoken_prefix_kept", "cc_in_place_of_48"])
def test_t0084c_native_controls_are_rejected_and_seed_ids_avoids_them(
    name: str, thought: str, tool: str
) -> None:
    """Criterion 5 controls: a seed keeping the spoken prefix, and one with ``<cc>`` in place of 48, are rejected by
    the native grammar; ``seed_ids`` returns neither."""
    ids, bad, draft = _native_control(name, thought, tool)
    assert not accepts_native(tool, thought, bad), (
        f"the control {name} is accepted: the oracle is blind"
    )
    got = seed_ids_fmt(thought, ids, draft, fmt="native")
    assert got != bad
    assert accepts_native(tool, thought, got)


@pytest.mark.parametrize("thought", ["T1", "T2", "T4"])
def test_t0084c_native_trailing_whitespace_is_trimmed(thought: str) -> None:
    """R2 holds for the native format: a draft ending in ``"\\n"`` after ``<tool_call|>`` is trimmed (accepted), its
    untrimmed spelling is rejected."""
    _text, ids = native_pass1("press_key")
    draft = native_draft("press_key") + "\n"
    got = seed_ids_fmt(thought, ids, draft, fmt="native")
    assert got == native_spec_seed(thought, ids, draft)
    assert accepts_native("press_key", thought, got)
    head = {"T1": through_first_101(ids), "T2": HEADER, "T4": []}[thought]
    lead = {"T1": OPEN, "T2": OPEN, "T4": ""}[thought]
    assert not accepts_native("press_key", thought, head + encode(lead + draft) + [TOOL_RESPONSE])


@pytest.mark.parametrize("thought", ["T1", "T2", "T4"])
@pytest.mark.parametrize("tool", list(CALLS))
def test_t0084c_cc_seeds_are_unchanged(tool: str, thought: str) -> None:
    """Criterion 5 (negative): ``fmt="cc"`` and the default give the T-0081b seed, byte for byte (built by hand)."""
    ids = pass1(tool)
    want = spec_seed(thought, tool, ids)
    assert seed_ids_fmt(thought, ids, call_of(tool), fmt="cc") == want
    assert seed_ids(thought, ids, call_of(tool)) == want


def test_t0084c_an_unknown_format_is_refused() -> None:
    """``fmt`` is ``cc`` or ``native``; anything else (the grammar's ``json`` included) raises ``ValueError``."""
    ids = pass1("scroll")
    for bad in ("json", "xml", ""):
        with pytest.raises(ValueError):
            seed_ids_fmt("T4", ids, call_of("scroll"), fmt=bad)
