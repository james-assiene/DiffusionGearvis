"""T-0088 criteria 5, 6, 7 and 9: the client reads a tools-on stream by its token ids.

Every stream is built by the wire fake (``tests/fakes/vllm_sse.py``) in the recorded arm-(b) shape and read by the
real ``VllmClient`` over ``httpx.MockTransport``, decoding with a ``HostDetokenizer`` over T-0088a's subset fixture.

Expected values are hand-written strings, or come from the eval's own definition (``router.route`` +
``special_tokens.drop_special_literals``, the criterion-6 parity statement) applied to the whole decode — never
from the client.

Mutants this module is aimed at (task file, "Blast radius" and "Tests to write"):
- reading ``content``/``reasoning`` instead of the ids → ``test_served_fields_are_never_read`` and the replay rows;
- splitting on a tag only when it lies inside one chunk → the plain-text-tag rows and the parity property;
- dropping specials before the tag search → ``test_a_tag_spliced_by_a_special_does_not_route``;
- emitting the tool text as ``answer`` → every row checking ``answer``;
- taking the stop from ``stop_reason`` → ``test_the_stop_comes_from_the_ids_not_stop_reason``.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import hashlib
import itertools
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import local_astra
from local_astra.toolcall import native, router
from local_astra.toolcall.special_tokens import (
    SPECIAL_IDS,
    SPECIAL_TOKENS,
    STOP_IDS,
    drop_special_literals,
)
from tests.fakes import vllm_sse as V
from tests.unit import _llm_id_support as S
from tests.unit._llm_id_support import (
    CHANNEL_CLOSE,
    EOS,
    TOOL_CALL,
    TOOL_CALL_CLOSE,
    TOOL_RESPONSE,
    TURN_CLOSE,
)
from tests.unit._work_count import WorkCapExceeded, line_events

CALL_TEXT = "call:click_at{x:1,y:2}"
CALL = f"<|tool_call>{CALL_TEXT}<tool_call|>"


def _cuts(seq: Sequence[int]) -> list[list[list[int]]]:
    """Every two-chunk partition of ``seq``."""
    return [[list(seq[:k]), list(seq[k:])] for k in range(1, len(seq))]


async def _until_error(cl: Any, prompt: Any) -> tuple[list[Any], BaseException | None]:
    got: list[Any] = []
    try:
        async for c in cl.generate(prompt, "r1"):
            got.append(c)
    except Exception as exc:
        return got, exc
    return got, None


async def _served_by(lines: list[str]) -> tuple[list[Any], BaseException | None, V.Wire]:
    wire = V.Wire(streams=[lines])
    cl = S.client(wire.transport())
    got, exc = await _until_error(cl, S.tools_prompt())
    await cl.aclose()
    return got, exc, wire


def _rewrite(lines: list[str], edit: Any) -> list[str]:
    """Apply ``edit(chunk_index, chunk_dict)`` to every non-role chunk and re-serialise."""
    out = [lines[0]]
    for k, ln in enumerate(lines[1:]):
        if ln == V.DONE_LINE:
            out.append(ln)
            continue
        chunk = json.loads(ln[len("data: ") :])
        edit(k, chunk)
        out.append("data: " + V.compact(chunk))
    return out


# =========================================================================== criterion 5: reading the wire


async def test_id_mode_without_a_detokenizer_raises_before_any_socket_opens() -> None:
    wire = V.Wire(streams=[S.stream_lines([[*S.thought("t"), *S.prose("Hi."), TURN_CLOSE]])])
    cl = S.client(wire.transport(), detokenizer=None)
    with pytest.raises(ValueError):
        async for _ in cl.generate(S.tools_prompt(), "r1"):
            pass
    await cl.aclose()
    assert wire.paths == [], "criterion 5: the refusal comes before any request"


async def test_tools_off_needs_no_detokenizer() -> None:
    """The refusal is id mode's alone: a tools-off prompt on a client without a detokeniser reads as on main."""
    sse = 'data: {"choices":[{"delta":{"content":"Four."}}]}\n\ndata: [DONE]\n\n'
    cl = S.client(
        httpx.MockTransport(
            lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)
        ),
        detokenizer=None,
    )
    got = await S.collect(cl, S.make_prompt())
    await cl.aclose()
    assert [(c.segment, c.text) for c in got] == [("answer", "Four.")]


@pytest.mark.parametrize(
    ("chunks", "drop", "absent"),
    [
        pytest.param(
            [[*S.thought("Tap it."), *S.prose("Done here.")], [*S.prose(" More."), TURN_CLOSE]],
            0,
            ("Tap it.", "Done here."),
            id="content-chunk",
        ),
        pytest.param(
            [[*S.header(), *S.prose("Tap it.")], [CHANNEL_CLOSE, *S.prose("Ok."), TURN_CLOSE]],
            0,
            ("Tap it.",),
            id="reasoning-only-chunk",
        ),
        pytest.param(
            [
                [*S.thought("Tap it."), *S.prose("Ok.")],
                [*S.prose(" Bye."), *S.call(), TOOL_RESPONSE],
            ],
            1,
            ("Bye", "call:"),
            id="final-chunk",
        ),
    ],
)
async def test_a_chunk_without_token_ids_is_a_wire_error_and_yields_nothing(
    chunks: list[list[int]], drop: int, absent: tuple[str, ...]
) -> None:
    lines = S.stream_lines(chunks, drop_token_ids=frozenset({drop}))
    chunk = json.loads(lines[1 + drop][len("data: ") :])
    assert "token_ids" not in chunk["choices"][0], "the fake dropped the ids"
    got, exc, _ = await _served_by(lines)
    assert isinstance(exc, S.wire_error()), f"criterion 5: expected LlmWireError, got {exc!r}"
    said = "".join(c.text for c in got)
    for text in absent:
        assert text not in said, f"nothing from the id-less chunk is yielded; {text!r} was"
    assert S.tool_commits(got) == []


async def test_the_role_chunk_carries_no_ids_and_is_accepted() -> None:
    lines = S.stream_lines([[*S.thought("t"), *S.prose("Hi."), TURN_CLOSE]])
    role = json.loads(lines[0][len("data: ") :])
    assert "token_ids" not in role["choices"][0] and role["choices"][0]["delta"] == {
        "role": "assistant",
        "content": "",
    }
    got, exc, _ = await _served_by(lines)
    assert exc is None
    assert S.texts(got, "answer") == "Hi."


@pytest.mark.parametrize("bad", [5, 10**7], ids=["hole-in-vocab", "out-of-range"])
async def test_an_unknown_id_is_a_wire_error_with_no_tool_commit(bad: int) -> None:
    lines = S.stream_lines(
        [[*S.thought("t"), *S.prose("Ok.")], [*S.call(), TOOL_RESPONSE]],
        foreign_ids={0: [bad]},  # before the stop: ids after it are ignored
    )
    got, exc, _ = await _served_by(lines)
    assert isinstance(exc, S.wire_error()), f"criterion 5: expected LlmWireError, got {exc!r}"
    assert S.tool_commits(got) == []


@pytest.mark.parametrize("done", [True, False], ids=["with-DONE", "stream-just-ends"])
async def test_a_stream_without_a_finish_reason_is_a_wire_error_with_no_tool_commit(
    done: bool,
) -> None:
    lines = S.stream_lines(
        [[*S.thought("t"), *S.prose("Ok."), *S.call(), TOOL_RESPONSE]], truncate=True, done=done
    )
    got, exc, _ = await _served_by(lines)
    assert isinstance(exc, S.wire_error()), f"criterion 5: expected LlmWireError, got {exc!r}"
    assert S.tool_commits(got) == []


async def test_ids_after_the_stop_are_ignored() -> None:
    seq = [
        *S.thought("t"),
        *S.prose("Ok."),
        TURN_CLOSE,
        *S.prose(" leaked"),
        *S.call(),
        TOOL_RESPONSE,
    ]
    got, _ = await S.run_chunks([seq])
    assert S.texts(got, "answer") == "Ok."
    assert "leaked" not in "".join(c.text for c in got)
    assert S.tool_commits(got) == [], "the call after the stop is not this response's"


async def test_the_stop_comes_from_the_ids_not_stop_reason() -> None:
    """Mutant: ``stop_id = stop_reason``. The ids say 50; the chunk says 106."""
    lines = S.stream_lines([[*S.thought("t"), *S.prose("Ok."), *S.call(), TOOL_RESPONSE]])

    def lie(k: int, chunk: dict[str, Any]) -> None:
        if chunk["choices"][0].get("finish_reason"):
            chunk["choices"][0]["stop_reason"] = TURN_CLOSE

    got, exc, _ = await _served_by(_rewrite(lines, lie))
    assert exc is None
    (tool,) = S.tool_commits(got)
    assert S.stop_id(tool) == TOOL_RESPONSE


async def test_served_fields_are_never_read() -> None:
    """Mutant: reading ``content``/``reasoning``. They are garbage here; the ids are right."""
    seq = [*S.thought("Look."), *S.prose("Clicking."), *S.call(), TOOL_RESPONSE]
    lines = S.stream_lines(_cuts(seq)[len(seq) // 2])

    def garble(k: int, chunk: dict[str, Any]) -> None:
        delta = chunk["choices"][0]["delta"]
        delta["content"] = "SERVED-CONTENT"
        delta["reasoning"] = "SERVED-REASONING"

    got, exc, _ = await _served_by(_rewrite(lines, garble))
    assert exc is None
    assert S.texts(got, "thought") == "Look."
    assert S.texts(got, "answer") == "Clicking."
    assert "SERVED" not in "".join(c.text for c in got)


async def test_an_abort_yields_no_tool_commit() -> None:
    seq_a = [*S.thought("t"), *S.prose("I will.")]
    seq_b = [*S.call(), TOOL_RESPONSE]
    wire = V.Wire(streams=[S.stream_lines([seq_a, seq_b])], delay_s=0.01)
    cl = S.client(wire.transport())
    got = []
    async for c in cl.generate(S.tools_prompt(), "r1"):
        got.append(c)
        await cl.abort("r1")
    await cl.aclose()
    assert got, "the first canvas was read"
    assert S.tool_commits(got) == []


# =========================================================================== criterion 6: the split


@pytest.mark.parametrize("cut", [1, 2, 3])
async def test_the_thought_header_split_at_each_of_its_ids(cut: int) -> None:
    seq = [*S.thought("Find it."), *S.prose("Here."), TURN_CLOSE]
    got, _ = await S.run_chunks([seq[:cut], seq[cut:]])
    assert S.texts(got, "thought") == "Find it.", "the header and markers are not thought text"
    assert S.texts(got, "answer") == "Here."
    assert S.tool_commits(got) == []


async def test_a_canvas_ending_inside_the_thought_is_thought_open() -> None:
    """Whole-word pieces (not byte runs, which the detokeniser holds), so the first canvas emits thought text."""
    first = [*S.header(), *S.ids("▁the", "▁a")]
    second = [*S.ids("▁b"), CHANNEL_CLOSE, *S.ids("Here", "."), TURN_CLOSE]
    got, _ = await S.run_chunks([first, second])
    c0 = [c for c in got if c.canvas_idx == got[0].canvas_idx]
    assert [(c.segment, c.text) for c in c0] == [("thought", " the a")]
    assert all(c.thought_open for c in c0), "the canvas ends inside the thought"
    assert S.texts(got, "thought") == " the a b"
    assert [(c.segment, c.text, c.thought_open) for c in got if c.segment == "answer"] == [
        ("answer", "Here.", False)
    ]


@pytest.mark.parametrize("plain", [False, True], ids=["special-48", "spelled-in-plain-text"])
async def test_a_tag_at_every_chunk_boundary_of_a_two_canvas_stream(plain: bool) -> None:
    tag = S.ids("<", "|", "tool", "_", "call", ">") if plain else [TOOL_CALL]
    seq = [
        *S.thought("t"),
        *S.prose("I will."),
        *tag,
        *S.prose(CALL_TEXT),
        TOOL_CALL_CLOSE,
        TOOL_RESPONSE,
    ]
    for chunks in _cuts(seq):
        got, _ = await S.run_chunks(chunks)
        assert S.texts(got, "answer") == "I will.", chunks
        tools_ = S.tool_commits(got)
        assert len(tools_) == 1, chunks
        assert tools_[0].text == CALL, chunks
        assert S.stop_id(tools_[0]) == TOOL_RESPONSE


async def test_e_key_02s_stray_tool_call_close_is_dropped_from_the_answer() -> None:
    (row,) = [r for r in S.recorded() if (r["arm"], r["case"]) == ("e", "key-02")]
    seq = S.recorded_ids(row)
    assert TOOL_CALL_CLOSE in seq[: seq.index(TOOL_CALL)], "the recorded stray precedes the call"
    want = "I will press Ctrl+C to copy the highlighted text."
    for chunks in [[seq], *_cuts(seq)[60:]]:
        got, _ = await S.run_chunks(chunks)
        assert S.texts(got, "answer") == want
        (tool,) = S.tool_commits(got)
        assert tool.text == '<|tool_call>call:press_key{combo:<|"|>Ctrl+C<|"|>}<tool_call|>'


async def test_a_tag_inside_an_open_ui_fence_is_ui_text() -> None:
    seq = [
        *S.thought("t"),
        *S.prose("Here."),
        *S.ids("<", "a", "2", "ui", ">"),
        *S.prose('{"a":"'),
        TOOL_CALL,
        *S.prose('"}'),
        *S.ids("</", "a", "2", "ui", ">"),
        *S.prose(" Done."),
        TURN_CLOSE,
    ]
    got, _ = await S.run_chunks([seq])
    assert S.tool_commits(got) == []
    ui = [c for c in got if c.segment == "ui"]
    # The tag is UI text, so it does not route; ruling E8 then strips its literal from the ui piece.
    assert [(c.text, c.fence_close) for c in ui] == [('{"a":""}', "written")]
    assert S.texts(got, "answer") == "Here. Done."


async def test_a_tag_inside_a_thought_is_thought() -> None:
    seq = [
        *S.header(),
        *S.prose("Maybe "),
        TOOL_CALL,
        *S.prose(" no."),
        CHANNEL_CLOSE,
        *S.prose("Sure."),
        TURN_CLOSE,
    ]
    got, _ = await S.run_chunks([seq])
    assert S.texts(got, "thought") == "Maybe <|tool_call> no."
    assert S.texts(got, "answer") == "Sure."
    assert S.tool_commits(got) == []


async def test_a2ui_after_the_tag_belongs_to_the_tool_text() -> None:
    fence = [*S.ids("<", "a", "2", "ui", ">"), *S.prose("{}")]
    seq = [*S.thought("t"), *S.prose("Ok."), *S.call(), *fence, TOOL_RESPONSE]
    got, _ = await S.run_chunks([seq])
    assert [c for c in got if c.segment == "ui"] == []
    (tool,) = S.tool_commits(got)
    assert tool.text == CALL + "<a2ui>{}"
    assert S.texts(got, "answer") == "Ok."


async def test_a_second_closed_thought_after_the_answer() -> None:
    seq = [*S.thought("First."), *S.prose("Ans."), *S.thought("Second."), *S.call(), TOOL_RESPONSE]
    got, _ = await S.run_chunks([seq])
    assert [c.text for c in got if c.segment == "thought"] in (["First.", "Second."],) or (
        S.texts(got, "thought") == "First.Second."
    )
    assert S.texts(got, "answer") == "Ans."
    (tool,) = S.tool_commits(got)
    assert tool.text == CALL


async def test_an_unclosed_thought_after_answer_text_ends_the_answer() -> None:
    seq = [*S.thought("t"), *S.prose("Ans."), *S.header(), *S.prose("more"), TURN_CLOSE]
    got, _ = await S.run_chunks([seq[:-3], seq[-3:]])
    assert S.texts(got, "answer") == "Ans."
    assert "more" not in S.texts(got, "answer")
    last = [c for c in got if c.canvas_idx == got[-1].canvas_idx and c.segment != "tool"]
    assert all(c.thought_open for c in last)


async def test_the_tool_text_is_cut_at_an_unclosed_channel() -> None:
    seq = [
        *S.thought("t"),
        *S.prose("Ok."),
        TOOL_CALL,
        *S.prose("call:click_at{x:1"),
        *S.header(),
        *S.prose("hmm"),
    ]
    got, _ = await S.run_chunks([[*seq, TOOL_RESPONSE]])
    (tool,) = S.tool_commits(got)
    assert tool.text == "<|tool_call>call:click_at{x:1"


async def test_no_spoken_sentence_when_the_tag_comes_first() -> None:
    seq = [*S.thought("t"), *S.call(), TOOL_RESPONSE]
    got, _ = await S.run_chunks([seq])
    assert S.texts(got, "answer") == ""
    (tool,) = S.tool_commits(got)
    assert tool.text == CALL


async def test_cc_first_is_a_tool_commit_with_cc() -> None:
    body = '{"name":"click_at"}'
    seq = [*S.thought("t"), *S.ids("<", "cc", ">"), *S.prose(body), TURN_CLOSE]
    got, _ = await S.run_chunks([seq])
    (tool,) = S.tool_commits(got)
    assert tool.text == "<cc>" + body
    assert S.stop_id(tool) == TURN_CLOSE
    assert S.texts(got, "answer") == ""


async def test_a_tag_spliced_by_a_special_does_not_route() -> None:
    """``<|tool<|"|>_call>``: removing ``<|"|>`` would make a tag; the removal comes after the search (Q2).

    Expected, from the parity definition: ``route`` sees no tag, and the answer is
    ``drop_special_literals('I<|tool<|"|>_call>.') == 'I.'`` (the removal repeats until no literal is left)."""
    spliced = [*S.ids("<", "|", "tool"), S.LITERAL_ID['<|"|>'], *S.ids("_", "call", ">")]
    seq = [*S.thought("t"), *S.prose("I"), *spliced, *S.prose("."), TURN_CLOSE]
    want = drop_special_literals('I<|tool<|"|>_call>.')
    assert want == "I."
    assert router.route(S.decode(seq[:-1])).tag is None
    for chunks in [[seq], *_cuts(seq)]:
        got, _ = await S.run_chunks(chunks)
        assert S.tool_commits(got) == [], chunks
        assert S.texts(got, "answer") == want, chunks


async def test_thought_markers_in_answer_text_make_it_thought_m9() -> None:
    seq = [*S.thought("t"), *S.prose("Hi "), SPECIAL_ID("<|think|>"), *S.prose("x."), TURN_CLOSE]
    got, _ = await S.run_chunks([seq])
    assert "Hi" not in S.texts(got, "answer")
    assert "Hi" in S.texts(got, "thought")


def SPECIAL_ID(lit: str) -> int:
    return S.LITERAL_ID[lit]


async def test_special_literals_are_removed_from_answer_pieces() -> None:
    seq = [
        *S.thought("t"),
        *S.prose("A"),
        CHANNEL_CLOSE,
        *S.prose("B"),
        SPECIAL_ID("<pad>"),
        *S.prose("C."),
        TURN_CLOSE,
    ]
    got, _ = await S.run_chunks([seq])
    answer = S.texts(got, "answer")
    assert answer == "ABC."
    for c in got:
        if c.segment == "answer":
            assert not any(lit in c.text for lit in SPECIAL_TOKENS)


async def test_special_literals_are_removed_from_ui_pieces_e8() -> None:
    """Ruling E8: in id mode ``ui`` pieces go through ``drop_special_literals``, the same as ``answer``.

    The fence holds ``<pad>`` and ``<tool|>`` (neither a tag, a channel marker nor a stop). Expected by hand:
    the fence body ``{"a":<pad>1<tool|>}`` loses both literals, so the written surface is ``{"a":1}``; every
    two-canvas cut, including the ones that publish an open fence, carries no literal on any ``ui`` commit."""
    seq = [
        *S.thought("t"),
        *S.prose("Here."),
        *S.ids("<", "a", "2", "ui", ">"),
        *S.prose('{"a":'),
        SPECIAL_ID("<pad>"),
        *S.prose("1"),
        SPECIAL_ID("<tool|>"),
        *S.prose("}"),
        *S.ids("</", "a", "2", "ui", ">"),
        *S.prose(" Done."),
        TURN_CLOSE,
    ]
    for chunks in [[seq], *_cuts(seq)]:
        got, _ = await S.run_chunks(chunks)
        ui = [c for c in got if c.segment == "ui"]
        for c in ui:
            assert not any(lit in c.text for lit in SPECIAL_TOKENS), (chunks, c.text)
        assert [c.text for c in ui if c.fence_close == "written"] == ['{"a":1}'], chunks
        assert S.texts(got, "answer") == "Here. Done.", chunks
        assert S.tool_commits(got) == [], chunks


async def test_a_held_fragment_that_never_becomes_a_marker_is_text() -> None:
    seq = [*S.thought("t"), *S.prose("a <"), *S.prose(" b."), TURN_CLOSE]
    k = len(S.thought("t")) + len(S.prose("a <"))
    got, _ = await S.run_chunks([seq[:k], seq[k:]])
    assert S.texts(got, "answer") == "a < b.", "parity: the split must not change the answer"


def _plain_sse(contents: Sequence[str]) -> httpx.MockTransport:
    sse = "".join(
        "data: " + json.dumps({"choices": [{"delta": {"content": c}}]}) + "\n\n" for c in contents
    )
    return httpx.MockTransport(
        lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)
    )


@pytest.mark.parametrize("frag", ["<", "</"])
@pytest.mark.parametrize("with_detok", [False, True], ids=["no-detokenizer", "with-detokenizer"])
async def test_tools_off_still_drops_a_held_fragment_that_never_becomes_a_marker_e7(
    frag: str, with_detok: bool
) -> None:
    """Ruling E7, the tools-off half: main's ``split_canvas`` holds a trailing ``<`` (a prefix of ``<a2ui>``) and
    drops it when the next canvas does not complete a marker (L10). That stays byte for byte (criterion 10),
    whether or not the client was given a detokeniser; only id mode gives the fragment back as text (the row
    above). Expected by hand: ``"a " + " b."``."""
    from local_astra.conductor import llm_client

    transport = _plain_sse(["a " + frag, " b."])
    cl = S.client(transport) if with_detok else llm_client.VllmClient(transport=transport)
    got = await S.collect(cl, S.make_prompt())
    await cl.aclose()
    assert [(c.segment, c.text) for c in got] == [("answer", "a "), ("answer", " b.")]


async def test_a_fragment_followed_by_a_fragment_across_three_canvases_is_kept() -> None:
    """Found by the parity property against a reference that let ``split_canvas`` hold ``<`` (main's L10 drops a held
    fragment that does not become a marker): ``' I<' | '<' | ' I'`` must still answer ``' I<< I'``."""
    head = [*S.thought("t")]
    got, _ = await S.run_chunks(
        [[*head, *S.ids("▁I", "<")], S.ids("<"), [*S.ids("▁I"), TURN_CLOSE]]
    )
    assert S.texts(got, "answer") == " I<< I"


async def test_a_fragment_held_when_the_response_ends_is_dropped_l10() -> None:
    seq = [*S.thought("t"), *S.prose("Hi <"), TURN_CLOSE]
    got, _ = await S.run_chunks([seq])
    assert S.texts(got, "answer") == "Hi "


# =========================================================================== criterion 7: the tool commit


async def test_exactly_one_tool_commit_even_with_two_calls_and_it_is_last() -> None:
    seq = [*S.thought("t"), *S.prose("Ok."), *S.call(), *S.call(), TOOL_RESPONSE]
    for chunks in [[seq], _cuts(seq)[len(seq) // 2]]:
        got, _ = await S.run_chunks(chunks)
        tools_ = S.tool_commits(got)
        assert len(tools_) == 1
        assert got[-1].segment == "tool"
        assert tools_[0].text == CALL + CALL
        assert tools_[0].thought_open is False


@pytest.mark.parametrize(
    ("stop", "finish"),
    [(TOOL_RESPONSE, "stop"), (TURN_CLOSE, "stop"), (EOS, "stop"), (None, "length")],
)
async def test_the_tool_commit_carries_the_requests_stop_id(stop: int | None, finish: str) -> None:
    seq = [*S.thought("t"), *S.prose("Ok."), *S.call(), *([stop] if stop is not None else [])]
    lines = S.stream_lines([seq])
    assert json.loads(lines[1][len("data: ") :])["choices"][0]["finish_reason"] == finish
    got, exc, _ = await _served_by(lines)
    assert exc is None
    (tool,) = S.tool_commits(got)
    assert S.stop_id(tool) == stop


async def test_canvas_idx_never_decreases_and_the_tool_is_on_the_final_canvas() -> None:
    seq = [*S.thought("Plan."), *S.prose("I will do it now."), *S.call(), TOOL_RESPONSE]
    for size in (4, 7, len(seq)):
        got, _ = await S.run_chunks(V.partition(seq, size))
        idx = [c.canvas_idx for c in got]
        assert idx == sorted(idx), (size, idx)
        tool = S.tool_commits(got)[0]
        assert tool.canvas_idx == max(idx)
    got, _ = await S.run_chunks([seq])
    assert {c.canvas_idx for c in got} == {got[0].canvas_idx}, (
        "one canvas: every commit carries its index"
    )


async def test_a_continuation_starts_in_the_tool_state_with_no_answer() -> None:
    prompt = S.continuation_prompt("I will click it.<|tool_call>")
    out = [*S.prose(CALL_TEXT), TOOL_CALL_CLOSE, TOOL_RESPONSE]
    for chunks in [[out], *_cuts(out)[::5]]:
        got, _ = await S.run_chunks(chunks, prompt)
        assert [c for c in got if c.segment == "answer"] == []
        (tool,) = S.tool_commits(got)
        # Ruling E4: the tag (taken from the prompt) leads the text, so every tool commit has one shape.
        assert tool.text == "<|tool_call>" + CALL_TEXT + "<tool_call|>", chunks
        assert S.stop_id(tool) == TOOL_RESPONSE


async def test_a_continuation_removes_closed_thoughts_from_the_tool_text() -> None:
    prompt = S.continuation_prompt("<|tool_call>")
    out = [
        *S.prose("call:click_at{"),
        *S.thought("hm"),
        *S.prose("x:1,y:2}"),
        TOOL_CALL_CLOSE,
        TOOL_RESPONSE,
    ]
    got, _ = await S.run_chunks([out], prompt)
    (tool,) = S.tool_commits(got)
    assert tool.text == "<|tool_call>" + CALL_TEXT + "<tool_call|>", (
        "ruling E4: the tag leads the text"
    )
    assert [c for c in got if c.segment == "answer"] == []


async def test_a_length_cut_continuation_still_leads_with_the_tag() -> None:
    """Ruling E4 on the length-cut path: the tag leads the text whatever ends the response; ``stop_id`` is None."""
    prompt = S.continuation_prompt("I will click it.<|tool_call>")
    got, _ = await S.run_chunks([S.prose("call:click_at{x:1")], prompt)
    assert [c for c in got if c.segment == "answer"] == []
    (tool,) = S.tool_commits(got)
    assert tool.text == "<|tool_call>call:click_at{x:1"
    assert S.stop_id(tool) is None


async def test_a_continuation_not_ending_in_the_tag_splits_as_usual() -> None:
    prompt = S.continuation_prompt("Sure,")
    got, _ = await S.run_chunks([[*S.prose(" done."), TURN_CLOSE]], prompt)
    assert S.texts(got, "answer") == " done."
    assert S.tool_commits(got) == []


async def test_the_tool_commit_is_refused_by_the_speech_gate() -> None:
    """T-0088a criterion 3: a ``tool`` commit fed to the gate raises ``GateInputError``."""
    from local_astra.conductor import speech_gate

    got, _ = await S.run_chunks([[*S.thought("t"), *S.call(), TOOL_RESPONSE]])
    (tool,) = S.tool_commits(got)
    gate = speech_gate.SpeechGate()
    with pytest.raises(speech_gate.GateInputError):
        gate.feed(tool)


# =========================================================================== criterion 6: router parity property

#: plain pieces for t and s: prose and the fragment pieces markers can be spelled from
PROSE = [
    "▁I",
    "▁the",
    "▁a",
    "▁b",
    ".",
    "A",
    "B",
    "é",
    "Hi",
    "Here",
    "call",
    ":",
    "click",
    "_",
    "at",
    "{",
    "}",
]
FRAGMENTS = ["<", "|", "|>", ">", "tool", "cc", "gen", "ui", "▁<"]
#: specials allowed in s: every special except ``<|channel>`` (excluded by the parity set), the stops (D ends at
#: the first stop) and ``<|think|>`` (M9 makes a canvas thought, which ``route`` does not model; reported)
S_SPECIALS = [
    lit
    for i, lit in SPECIAL_IDS.items()
    if i not in STOP_IDS and lit not in ("<|channel>", "<|think|>")
]
TAG_SPELLINGS: list[list[str]] = [
    ["<|tool_call>"],
    ["<", "|", "tool", "_", "call", ">"],
    ["<", "cc", ">"],
    ["<", "gen", "ui", ">"],
]
RECOGNISED = (
    *router.TAGS.values(),
    router.CHANNEL_OPEN,
    router.CHANNEL_CLOSE,
    "<a2ui>",
    "</a2ui>",
    *SPECIAL_TOKENS,
)


def _proper_prefix_tail(text: str) -> bool:
    return any(text.endswith(m[:k]) for m in RECOGNISED for k in range(1, len(m)))


@st.composite
def _parity_case(draw: st.DrawFn) -> tuple[list[int], list[list[int]]]:
    t = draw(st.lists(st.sampled_from(PROSE), max_size=6))
    s_pieces = draw(st.lists(st.sampled_from([*PROSE, *FRAGMENTS, *S_SPECIALS]), max_size=14))
    if draw(st.booleans()):
        at = draw(st.integers(0, len(s_pieces)))
        # the tag as its special id, or spelled in plain text (route finds either; a tag split across chunks must
        # still be found: the "split on a tag only within a chunk" mutant)
        tag = draw(st.sampled_from(TAG_SPELLINGS))
        s_pieces[at:at] = [
            *tag,
            *draw(st.lists(st.sampled_from(PROSE), max_size=4)),
            "<tool_call|>",
        ]
    stop = draw(st.sampled_from(sorted(STOP_IDS)))
    seq = [*S.ids(*S.HEADER, *t), CHANNEL_CLOSE, *S.ids(*s_pieces), stop]
    cuts = sorted(draw(st.sets(st.integers(1, len(seq) - 1), max_size=len(seq) - 1)))
    bounds = [0, *cuts, len(seq)]
    return seq, [seq[a:b] for a, b in itertools.pairwise(bounds)]


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_parity_case())
def test_router_parity_for_any_partition(case: tuple[list[int], list[list[int]]]) -> None:
    """Criterion 6 (T-0085's eval definition). D = header + t + ``<channel|>`` + s, decoded up to the stop.

    Domain, as written plus three exclusions the spec forces (each reported as an ambiguity): s has no
    ``<|channel>`` and no ``<a2ui>`` (as written), no ``</a2ui>`` (``_strip_markup`` drops it from answers), no
    ``THOUGHT_MARKERS`` (M9), and a tag-free D does not end in a proper prefix of a recognised marker (L10 drops
    it at the end of the response, while ``route`` keeps it)."""
    seq, chunks = case
    d = S.decode(seq[:-1])
    s_text = d.split("<channel|>", 1)[1]
    assume("<|channel>" not in s_text and "<a2ui>" not in s_text and "</a2ui>" not in s_text)
    assume(not any(m in d for m in ("<|think|>", "<|/think|>", "<think>", "</think>")))
    r = router.route(d)
    assume(r.tag is not None or not _proper_prefix_tail(drop_special_literals(r.spoken_prefix)))
    got = S.run(chunks)
    assert S.texts(got, "answer") == drop_special_literals(r.spoken_prefix)
    tools_ = S.tool_commits(got)
    if r.tag is None:
        assert tools_ == []
    else:
        assert len(tools_) == 1
        assert tools_[0].text == router.TAGS[r.tag] + r.draft_call_text + r.draft_close
        assert S.stop_id(tools_[0]) == seq[-1]


# =========================================================================== criterion 9: linear host time


def _long_response(n: int = 4096) -> list[int]:
    """An ``n``-id response: thought, answer and call, built from recorded pieces, ending in 50. The
    channel close, the tag and the call close sit at 1200, 3000 and 4094 for n = 4096, scaled to
    ``n`` (T-0129)."""
    rec = {(r["arm"], r["case"]): S.recorded_ids(r) for r in S.recorded()}
    body = [t for t in rec[("b", "type-01")] if t not in SPECIAL_IDS]
    close_at, tag_at, end_at = 1200 * n // 4096, 3000 * n // 4096, n - 2
    out = [*S.header()]
    while len(out) < close_at:
        out.extend(body)
    out = [*out[:close_at], CHANNEL_CLOSE]
    while len(out) < tag_at:
        out.extend(body)
    out = [*out[:tag_at], TOOL_CALL]
    while len(out) < end_at:
        out.extend(body)
    return [*out[:end_at], TOOL_CALL_CLOSE, TOOL_RESPONSE]


def seq_e11(n: int) -> list[int]:
    """Ruling E11's sequence at ``n`` ids: a short thought, ``<`` and ``<pad>`` alternating, ``<turn|>``."""
    head = S.thought("t")
    return [*head, *([LT, PAD] * n)[: n - len(head) - 1], TURN_CLOSE]


def seq_e15(n: int) -> list[int]:
    """Ruling E15's sequence at ``n`` ids: a short thought, ``<|tool``, ``<pad>`` to the end, ``<turn|>``."""
    head = S.thought("t")
    return [*head, *([*S.ids("<", "|", "tool"), *([PAD] * n)])[: n - len(head) - 1], TURN_CLOSE]


#: T-0129 premise: sha256 of ``json.dumps`` of each builder's 4096-id sequence, computed from the
#: builders as they were at the fork point 1e24faf (``_long_response()`` and the two inline
#: sequences of the E11 and E15 rows), so the timed input is the one the rulings were written
#: against.
SEQ_4096_SHA256 = {
    "long": "b0210ce181b54c4a023f6448f6ec65069ef5f53bae6f5e39eb535b813eedfae5",
    "e11": "80b26d44e4bbb51caf13951f88945b82013698a0d02034bb936e9f1d8469a4db",
    "e15": "17125af8155ded217e9ab2a19af7c8df031a709ec188e8e7bf595157e73559e3",
}

#: T-0129 criterion 4 (work): sizes in one-id chunks and the bar (linear 2, quadratic 4).
ID_WORK_SIZES = (1024, 2048)
ID_WORK_RATIO = 2.25
#: T-0129 criterion 5 (CPU): repetitions (min of 7, gc off) and bars. 5.0 x the raw reader restates
#: T-0088's 50 ms (the raw reader is 9.0-9.5 ms of CPU idle on this box, so 50 ms is 5.3-5.5 x);
#: 8 is the geometric middle of linear (4) and quadratic (16) for 1024 -> 4096.
ID_CPU_RUNS = 7
ID_HOST_OVER_RAW = 5.0
ID_HOST_RATIO_4X = 8.0


def _premise_4096(name: str, build: Any) -> list[int]:
    seq = build(4096)
    assert len(seq) == 4096
    digest = hashlib.sha256(json.dumps(seq).encode()).hexdigest()
    assert digest == SEQ_4096_SHA256[name], (
        f"T-0129 premise: the size-parametrised {name} builder at n = 4096 is not the sequence the "
        f"ruling was written against (sha256 {digest})"
    )
    return seq


_BUILDERS: dict[str, Any] = {"long": _long_response, "e11": seq_e11, "e15": seq_e15}


@functools.cache
def _one_id_lines(name: str, n: int) -> list[str]:
    """The SSE lines of ``_BUILDERS[name](n)`` as one-id chunks, built once per (name, n): building 4096 lines costs
    about 0.14 s, more than one client run (T-0129 criterion 10). The wire fake copies the list it serves."""
    seq = _BUILDERS[name](n)
    assert len(seq) == n
    return S.stream_lines([[t] for t in seq])


def _package_root() -> Path:
    """The ``local_astra`` package directory, from the imported package (a copy counts itself)."""
    return Path(local_astra.__file__).resolve().parent


def _id_work(row: str, name: str, check: Any) -> list[int]:
    """Criterion 4 (T-0129): traced line events under ``local_astra/`` during ``generate`` over
    ``_BUILDERS[name](n)`` as one-id chunks, n = 1024 then 2048; the 2048 run stops once it passes
    2.25 x L(1024) + 1. ``check(commits, n)`` is the row's expected answer at each size."""
    root = _package_root()
    counts: list[int] = []
    for n in ID_WORK_SIZES:
        lines = _one_id_lines(name, n)
        got: list[Any] = []
        cap = None if not counts else int(ID_WORK_RATIO * counts[-1]) + 1
        try:
            counts.append(line_events(functools.partial(_run_client, lines, got), root, cap=cap))
        except WorkCapExceeded as exc:
            raise AssertionError(
                f"{row} (work, T-0129): at {n} one-id chunks the client passed {exc.cap} traced line "
                f"events under {root} = {ID_WORK_RATIO} x L({n // 2}) + 1; host work is not linear"
            ) from None
        check(got, n)
    small, large = counts
    assert small > 0, f"{row} (work): no line under {root} was traced; the count would be vacuous"
    assert large <= ID_WORK_RATIO * small, (
        f"{row} (work, T-0129): L(2048)/L(1024) = {large}/{small} = {large / small:.3f} > "
        f"{ID_WORK_RATIO} traced line events; host work is not linear"
    )
    return counts


def _run_client(lines: list[str], sink: list[Any] | None = None) -> None:
    got = asyncio.run(_consume_client(lines))
    if sink is not None:
        sink.extend(got)


def _run_raw(lines: list[str]) -> None:
    asyncio.run(_consume_raw(lines))


def _thread_cpu(fn: Any, lines: list[str]) -> float:
    start = time.thread_time()
    fn(lines)
    return time.thread_time() - start


def _id_cpu(row: str, name: str, with_1024: bool) -> None:
    """Criterion 5 (T-0129): raw reader and client interleaved, ``thread_time``, gc off, min of 7;
    (min client - min raw)/min raw <= 5.0 at 4096, and, when ``with_1024``,
    host(4096)/host(1024) <= 8."""
    sizes = {n: _one_id_lines(name, n) for n in ((4096, 1024) if with_1024 else (4096,))}
    raw = dict.fromkeys(sizes, float("inf"))
    cli = dict.fromkeys(sizes, float("inf"))
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for _ in range(ID_CPU_RUNS):
            for n, lines in sizes.items():
                raw[n] = min(raw[n], _thread_cpu(_run_raw, lines))
                cli[n] = min(cli[n], _thread_cpu(_run_client, lines))
    finally:
        if enabled:
            gc.enable()
    host = {n: cli[n] - raw[n] for n in sizes}
    assert host[4096] / raw[4096] <= ID_HOST_OVER_RAW, (
        f"{row} (CPU, T-0129): host work {host[4096] * 1e3:.2f} ms of thread CPU for 4096 one-id "
        f"chunks = {host[4096] / raw[4096]:.2f} x the raw reader ({raw[4096] * 1e3:.2f} ms; min of "
        f"{ID_CPU_RUNS}, gc off); bound {ID_HOST_OVER_RAW} x (T-0088's 50 ms)"
    )
    if with_1024:
        assert host[1024] > 0, (
            f"{row} (CPU): the client took no more CPU than the raw reader at 1024 chunks "
            f"({cli[1024] * 1e3:.3f} vs {raw[1024] * 1e3:.3f} ms); the ratio is not measurable"
        )
        ratio = host[4096] / host[1024]
        assert ratio <= ID_HOST_RATIO_4X, (
            f"{row} (CPU, T-0129): host(4096)/host(1024) = {ratio:.2f} > {ID_HOST_RATIO_4X} "
            f"({host[1024] * 1e3:.2f} -> {host[4096] * 1e3:.2f} ms thread CPU, min of "
            f"{ID_CPU_RUNS}); host work is not linear"
        )


async def _consume_raw(lines: list[str]) -> None:
    """Baseline: the same body read by a trivial reader (lines + ``json.loads``), with no decode and no split."""
    wire = V.Wire(streams=[lines])
    async with (
        httpx.AsyncClient(transport=wire.transport(), base_url="http://127.0.0.1:8001/v1") as http,
        http.stream("POST", "/chat/completions", json={}) as resp,
    ):
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and line != V.DONE_LINE:
                json.loads(line[len("data: ") :])


async def _consume_client(lines: list[str]) -> list[Any]:
    wire = V.Wire(streams=[lines])
    cl = S.client(wire.transport())
    got = [c async for c in cl.generate(S.tools_prompt(), "r1")]
    await cl.aclose()
    return got


def test_host_work_in_id_mode_is_linear_4096_one_id_chunks() -> None:
    """Criterion 9, gated row (T-0129; was ``..._under_50_ms``): the recorded-pieces response as one-id chunks.

    - work: traced line events under ``local_astra/`` at 1024 and 2048 chunks, L(2048)/L(1024) <= 2.25;
    - CPU: at 4096 chunks, (client - raw reader)/raw reader <= 5.0 (thread_time, gc off, interleaved, min of 7),
      which restates the old 50 ms.

    Both replace a ``perf_counter`` median of 5 under 50 ms, red in 44 of 360 trials under load (T-0129). Regime:
    CPU, the fast suite (serial or ``-n 12``). Re-decoding or re-splitting the prefix on every chunk is ~n²/2 piece
    operations and fails both."""
    _premise_4096("long", _long_response)
    got = asyncio.run(_consume_client(_one_id_lines("long", 4096)))
    assert len(S.tool_commits(got)) == 1, "the response is read correctly before it is timed"

    def one_tool_commit(commits: list[Any], n: int) -> None:
        assert len(S.tool_commits(commits)) == 1, (
            f"the {n}-id response is read correctly while counted"
        )

    _id_work("criterion 9", "long", one_tool_commit)
    _id_cpu("criterion 9", "long", with_1024=False)


def test_parse_call_accepts_the_tool_text_after_the_tag() -> None:
    """A guard on this module's own fixture strings: the call text the rows use is a valid native call."""
    assert native.parse_call(CALL.removeprefix("<|tool_call>")) == {
        "name": "click_at",
        "arguments": {"x": 1, "y": 2},
    }


# =========================================================================== rulings after review: E11, E12
#
# Sequences are reviewer 1's (``t0088-r1-…/probe/``: ``p_abort.py``, ``p_fence.py``, ``p_time.py``, ``p_dbg.py``,
# ``fuzz_text.py``). Each row names the mutant it was written against.

LT, PAD = S.ids("<")[0], S.LITERAL_ID["<pad>"]


class _Overrun(Exception):
    pass


async def _consume_client_within(lines: list[str], cutoff_s: float) -> None:
    """``_consume_client`` that gives up once ``cutoff_s`` has passed: the wire raises between two chunks, so a
    quadratic client fails in about ``cutoff_s`` rather than in minutes (the fast suite's budget)."""
    t0 = time.perf_counter()

    def check(k: int) -> None:
        if time.perf_counter() - t0 > cutoff_s:
            raise _Overrun(f"chunk {k}")

    wire = V.Wire(streams=[lines], on_chunk=check)
    cl = S.client(wire.transport())
    try:
        async for _ in cl.generate(S.tools_prompt(), "r1"):
            pass
    finally:
        await cl.aclose()


def _guarded_first_run(row: str, name: str, cutoff_s: float = 1.0) -> None:
    """One guarded run of the 4096-id sequence as one-id chunks before the CPU measure: a client still busy after
    ``cutoff_s`` (about 20x the old 50 ms bound) is reported as such, not timed."""
    try:
        asyncio.run(_consume_client_within(_one_id_lines(name, 4096), cutoff_s))
    except _Overrun as exc:
        raise AssertionError(
            f"{row}: id-mode host work passed {cutoff_s * 1000:.0f} ms on one run ({exc}); bound 50 ms"
        ) from None


def test_host_work_is_linear_on_alternating_lt_and_pad_4096_one_id_chunks_e11() -> None:
    """Ruling E11, the adversarial criterion-9 row: one-id chunks alternating ``<`` and ``<pad>`` after a short
    thought, ending in ``<turn|>``. Every ``<`` is a marker prefix, and so is every ``<`` left once ``<pad>`` is
    dropped; a hold that re-walks the whole run on each chunk is quadratic (reviewer 1: ~1 s and more at 625cadc).

    T-0129: work L(2048)/L(1024) <= 2.25 (traced line events under ``local_astra/``); a guarded run at 4096 (1 s);
    CPU at 4096 (thread_time, gc off, interleaved, min of 7) host/raw <= 5.0 and host(4096)/host(1024) <= 8. These
    replace a ``perf_counter`` median of 5 under 50 ms, red in 160 of 360 trials under load. Regime: CPU, the fast
    suite (serial or ``-n 12``)."""
    _premise_4096("e11", seq_e11)

    def every_lt_but_the_held_last(commits: list[Any], n: int) -> None:
        # E11 + L10, by hand: ``<pad>`` is dropped, ``<<`` cannot begin a marker, so every ``<`` is answer text
        # except the last, a held fragment when the response ends
        want = "<" * (seq_e11(n).count(LT) - 1)
        assert S.texts(commits, "answer") == want, (
            f"ruling E11: the {n}-id answer lost or gained text"
        )

    _id_work("ruling E11", "e11", every_lt_but_the_held_last)
    _guarded_first_run("ruling E11", "e11")
    _id_cpu("ruling E11", "e11", with_1024=True)


@pytest.mark.parametrize("per_id", [False, True], ids=["one-chunk", "one-id-chunks"])
async def test_a_run_of_lt_never_loses_text_e11(per_id: bool) -> None:
    """Ruling E11 (reviewer 1's ``p_dbg.py``): ``("a", "<") x 10`` then ``<turn|>``. Only the last ``<`` is a held
    fragment when the response ends (L10); every earlier ``<`` is followed by ``a`` and ``<a`` then ``<`` cannot
    begin a marker, so it is text. Expected by hand."""
    seq = [*S.thought("t"), *(S.ids("a", "<") * 10), TURN_CLOSE]
    got, _ = await S.run_chunks([[t] for t in seq] if per_id else [seq])
    assert S.texts(got, "answer") == "a<" * 9 + "a"


async def test_an_abort_after_the_tag_was_read_yields_no_tool_commit_m21() -> None:
    """Mutant M21 (tool commit emitted on abort). The tag is in chunk 1, so the tool text has begun when the consumer
    aborts after the first commit; criterion 5 still says no ``tool`` commit (reviewer 1's ``p_abort.py``)."""
    seq_a = [*S.thought("t"), *S.prose("I will."), TOOL_CALL, *S.prose("call:click")]
    seq_b = [*S.prose("_at{x:1,y:2}"), TOOL_CALL_CLOSE, TOOL_RESPONSE]
    wire = V.Wire(streams=[S.stream_lines([seq_a, seq_b])], delay_s=0.01)
    cl = S.client(wire.transport())
    got = []
    async for c in cl.generate(S.tools_prompt(), "r1"):
        got.append(c)
        await cl.abort("r1")
    await cl.aclose()
    assert got, "the first canvas was read"
    assert S.tool_commits(got) == []


async def test_a_tag_inside_a_fence_then_a_real_tag_routes_on_the_second_m25() -> None:
    """Mutant M25 (the first tag inside an open fence ends the search). Reviewer 1's ``p_fence.py``:
    ``Hi <a2ui>{"a":"<|tool_call>"}</a2ui> ok <|tool_call>call:x{}<tool_call|>``. Expected by hand: the fenced tag is
    UI text (its literal stripped, E8), the later one is the call."""
    seq = [
        *S.thought("t"),
        *S.prose("Hi "),
        *S.ids("<", "a", "2", "ui", ">"),
        *S.prose('{"a":"'),
        TOOL_CALL,
        *S.prose('"}'),
        *S.ids("</", "a", "2", "ui", ">"),
        *S.prose(" ok "),
        TOOL_CALL,
        *S.prose("call:x{}"),
        TOOL_CALL_CLOSE,
        TOOL_RESPONSE,
    ]
    assert S.decode(seq[len(S.thought("t")) : -1]) == (
        'Hi <a2ui>{"a":"<|tool_call>"}</a2ui> ok <|tool_call>call:x{}<tool_call|>'
    )
    for chunks in [[seq], *_cuts(seq)]:
        got, _ = await S.run_chunks(chunks)
        assert S.texts(got, "answer") == "Hi  ok ", chunks
        assert [c.text for c in got if c.segment == "ui" and c.fence_close == "written"] == [
            '{"a":""}'
        ], chunks
        (tool,) = S.tool_commits(got)
        assert tool.text == "<|tool_call>call:x{}<tool_call|>", chunks


#: reviewer 1's ``fuzz_text.py`` counterexample to M12, as ids: the tool text ends in ``<|`` when the stop arrives
M12_SEQ = [
    *S.header(),
    *S.prose("x  "),
    CHANNEL_CLOSE,
    *S.ids("cc"),
    CHANNEL_CLOSE,
    *S.prose(" "),
    *S.ids("<", "gen", "ui", ">"),
    *S.ids("gen", "call"),
    *S.ids("<", "|"),
    TURN_CLOSE,
]


async def test_the_tool_text_keeps_a_marker_prefix_tail_at_the_stop_m12() -> None:
    """Mutant M12 (the thought split never flushes a held ``<|channel>`` prefix at the end). D is
    ``<|channel>thought\\nx  <channel|>cc<channel|> <genui>gencall<|``: the tool text is the router's,
    ``<genui>gencall<|``, tail included (criterion 7). Expected by hand and checked against ``router.route``."""
    d = S.decode(M12_SEQ[:-1])
    assert d == "<|channel>thought\nx  <channel|>cc<channel|> <genui>gencall<|"
    r = router.route(d)
    assert router.TAGS[r.tag] + r.draft_call_text + r.draft_close == "<genui>gencall<|"
    for chunks in [[M12_SEQ], [[t] for t in M12_SEQ], *_cuts(M12_SEQ)]:
        got, _ = await S.run_chunks(chunks)
        assert S.texts(got, "answer") == "cc ", chunks
        (tool,) = S.tool_commits(got)
        assert tool.text == "<genui>gencall<|", chunks
        assert S.stop_id(tool) == TURN_CLOSE


async def test_an_unclosed_thought_at_the_stop_keeps_its_tail_m12b() -> None:
    """Mutant M12b (the thought split never flushes a held ``<channel|>`` prefix at the end). The response stops
    inside a thought ending in ``<``. The thought text is the served ``reasoning`` the engine derives (the wire
    fake's ``served_regions``: an unclosed thought runs to the end; criterion 8's agreement), ``Plan <``."""
    seq = [*S.header(), *S.prose("Plan <"), TURN_CLOSE]
    lines = S.stream_lines([seq])
    served = json.loads(lines[1][len("data: ") :])["choices"][0]["delta"].get("reasoning")
    assert served == "Plan <"
    got, _ = await S.run_chunks([seq])
    assert S.texts(got, "thought") == "Plan <"
    assert S.texts(got, "answer") == ""


#: the tails a per-id partition must carry to the end of a tool text (prefixes of ``<|channel>``/``<channel|>``)
TAILS: list[list[str]] = [[], ["<"], ["<", "|"], ["</"], ["▁<"]]


@st.composite
def _parity_case_per_id(draw: st.DrawFn) -> tuple[list[int], list[list[int]]]:
    """Criterion 6's parity set, every id its own chunk (ruling E12: per-character partitions), with s optionally
    ending in a marker-prefix tail just before the stop."""
    seq, _ = draw(_parity_case())
    tail = S.ids(*draw(st.sampled_from(TAILS)))
    seq = [*seq[:-1], *tail, seq[-1]]
    return seq, [[t] for t in seq]


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_parity_case_per_id())
def test_router_parity_for_one_id_chunks(case: tuple[list[int], list[list[int]]]) -> None:
    """Criterion 6 over per-id partitions (ruling E12; kills M12). Same oracle and exclusions as
    :func:`test_router_parity_for_any_partition`."""
    test_router_parity_for_any_partition.hypothesis.inner_test(case)  # type: ignore[attr-defined]


async def test_e8_holds_on_the_end_of_response_open_fence_flush_m7b() -> None:
    """Mutant M7b (E8 skipped on the flush of a fence still open when the response ends). The fence holds ``<pad>``
    and never closes; no ``ui`` commit may carry a literal, and the last one is ``{"a":1`` (hand)."""
    seq = [
        *S.thought("t"),
        *S.prose("Here."),
        *S.ids("<", "a", "2", "ui", ">"),
        *S.prose('{"a":'),
        SPECIAL_ID("<pad>"),
        *S.prose("1"),
        TURN_CLOSE,
    ]
    for chunks in [[seq], *_cuts(seq)]:
        got, _ = await S.run_chunks(chunks)
        ui = [c for c in got if c.segment == "ui"]
        assert ui, chunks
        for c in ui:
            assert not any(lit in c.text for lit in SPECIAL_TOKENS), (chunks, c.text)
        assert (ui[-1].text, ui[-1].fence_close) == ('{"a":1', "open"), chunks
        assert S.texts(got, "answer") == "Here.", chunks


@pytest.mark.parametrize(
    "bad",
    [["x"], [1.0], [True], "12", [None], {"a": 1}],
    ids=["str-in-list", "float", "bool", "str", "null", "dict"],
)
async def test_malformed_token_ids_are_a_wire_error_m18(bad: Any) -> None:
    """Mutant M18 (only ``isinstance(ids, list)`` checked). Reviewer 1's ``p_abort.py`` shapes on the final chunk:
    criterion 5 wants a list of ints, so each is ``LlmWireError`` and nothing of that chunk is yielded. ``[True]`` is
    the sharp one: ``True == 1`` is ``<eos>``, a stop id."""
    base = S.stream_lines([[*S.thought("t"), *S.prose("Ok.")], [*S.call(), TOOL_RESPONSE]])

    def corrupt(k: int, chunk: dict[str, Any]) -> None:
        if k == 1:
            chunk["choices"][0]["token_ids"] = bad

    got, exc, _ = await _served_by(_rewrite(base, corrupt))
    assert isinstance(exc, S.wire_error()), f"criterion 5: expected LlmWireError, got {exc!r}"
    assert "call:" not in "".join(c.text for c in got)
    assert S.tool_commits(got) == []


async def test_ids_after_the_stop_in_a_later_chunk_are_ignored_m23b() -> None:
    """Mutant M23b (decoding continues after the stop). The stop ends chunk 1; chunk 2 carries more ids and the
    ``finish_reason``. Nothing of chunk 2 is this response's (criterion 5)."""
    first = [*S.thought("t"), *S.prose("Ok."), TURN_CLOSE]
    second = [*S.prose(" leaked"), *S.call(), TOOL_RESPONSE]
    lines = S.stream_lines([first, second])
    assert json.loads(lines[1][len("data: ") :])["choices"][0]["finish_reason"] is None
    got, exc, _ = await _served_by(lines)
    assert exc is None
    assert S.texts(got, "answer") == "Ok."
    assert "leaked" not in "".join(c.text for c in got)
    assert S.tool_commits(got) == []


async def test_a_length_cut_flushes_the_tool_text_tail_m26() -> None:
    """Mutant M26 (no flush when the response ends on ``finish_reason: length``). No stop id: the tool text ends in
    ``<`` and must still be the router's, ``<|tool_call>call:click_at{x:1<`` (hand; checked against ``route``)."""
    seq = [*S.thought("t"), *S.prose("Ok."), TOOL_CALL, *S.prose("call:click_at{x:1<")]
    r = router.route(S.decode(seq))
    assert (
        router.TAGS[r.tag] + r.draft_call_text + r.draft_close == "<|tool_call>call:click_at{x:1<"
    )
    for chunks in [[seq], _cuts(seq)[len(seq) // 2], [[t] for t in seq]]:
        lines = S.stream_lines(chunks)
        assert json.loads(lines[-2][len("data: ") :])["choices"][0]["finish_reason"] == "length"
        got, exc, _ = await _served_by(lines)
        assert exc is None
        assert S.texts(got, "answer") == "Ok."
        (tool,) = S.tool_commits(got)
        assert tool.text == "<|tool_call>call:click_at{x:1<", chunks
        assert S.stop_id(tool) is None


# =========================================================================== ruling E15: the held-run cap
#
# Reviewer 1's re-review probes (``t0088-r1b-…/probe/``: ``p_time3.py``, ``p_cap.py``).


def test_host_work_is_linear_when_a_marker_prefix_is_followed_by_a_long_run_of_pads_e15() -> None:
    """Ruling E15 (pins the run cap, kills N8): ``<|tool`` stays a marker prefix however many ``<pad>`` follow it,
    so the held text would grow with every chunk and be re-scanned each time unless the removed run is capped.
    One-id chunks: ``thought("t") + [<, |, tool] + <pad>... + <turn|>``. Measures, bars and regime as the E11 row
    (T-0129: work at 1024/2048, a guarded run at 4096, CPU at 4096 and 1024); they replace a ``perf_counter`` median
    of 5 under 50 ms, red in 154 of 360 trials under load."""
    _premise_4096("e15", seq_e15)

    def no_check(commits: list[Any], n: int) -> None:
        pass  # the answer of a capped held run is pinned by the E15 cap rows below, not here

    _id_work("ruling E15", "e15", no_check)
    _guarded_first_run("ruling E15", "e15")
    _id_cpu("ruling E15", "e15", with_1024=True)


def _cap_case(n: int) -> tuple[list[int], list[int]]:
    """Reviewer 1's ``p_cap.py``: ``Hi <a2ui>{"a":"x<`` + ``<|"|>`` x n + ``/a2`` | ``b"} Done.`` then the stop."""
    first = [
        *S.thought("t"),
        *S.prose("Hi "),
        *S.ids("<", "a", "2", "ui", ">"),
        *S.prose('{"a":"x<'),
        *([S.STRING_DELIM] * n),
        *S.prose("/a2"),
    ]
    second = [*S.prose('b"} Done.'), TURN_CLOSE]
    q = '<|"|>'
    assert S.decode(first) == '<|channel>thought\nt<channel|>Hi <a2ui>{"a":"x<' + q * n + "/a2"
    assert S.decode(second[:-1]) == 'b"} Done.'
    return first, second


def _classified(got: list[Any]) -> tuple[str, list[tuple[str, Any]], tuple[str, Any] | None]:
    """What a consumer acts on: the spoken text, the fences that ended, and the last ``ui`` commit."""
    ui = [c for c in got if c.segment == "ui"]
    ended = [(c.text, c.fence_close) for c in ui if c.fence_close != "open"]
    return S.texts(got, "answer"), ended, ((ui[-1].text, ui[-1].fence_close) if ui else None)


@pytest.mark.parametrize("n", [12, 13], ids=["12-literals-uncapped", "13-literals-capped"])
async def test_the_run_cap_keeps_quote_parity_at_a_chunk_boundary_e15(n: int) -> None:
    """Ruling E15: a capped run that held an odd number of ``"`` must collapse to ``<|"|>``, not ``<pad>``, or the
    fence's inside-a-string state flips and `` Done.`` changes channel. Oracle: the same text in one chunk (no hold,
    no cap), so the boundary must not change what is spoken or shown. 12 literals (60 characters) stay under the
    64-character cap and are the green twin; 13 (65) are capped."""
    first, second = _cap_case(n)
    one, _ = await S.run_chunks([[*first, *second]])
    split, _ = await S.run_chunks([first, second])
    assert _classified(split) == _classified(one)


# =========================================================================== final recheck: the two _short_run branches
#
# Reviewer 1's ``t0088-r1c-…/probe/p_short.py``. Both runs are longer than the 64-character cap, so each is held
# in its collapsed form across the chunk boundary.

THINK = S.LITERAL_ID["<|think|>"]


async def test_a_capped_run_keeps_its_think_marker_so_the_rest_is_thought_s3() -> None:
    """Kills S3 (``return quote or "<pad>"``: the capped run loses ``<|think|>``). Chunk 1 ends in
    ``<|t`` + ``<|think|>`` + ``<pad>`` x 12 (69 characters, capped); chunk 2 is ``ool secret plan.``. The canvas
    that carries ``<|think|>`` is thought (M9), so nothing after ``Ok `` is spoken (hand)."""
    first = [*S.thought("t"), *S.prose("Ok <|t"), THINK, *([PAD] * 12)]
    second = [*S.prose("ool secret plan."), TURN_CLOSE]
    got, _ = await S.run_chunks([first, second])
    for c in got:
        if c.segment == "answer":
            assert "secret" not in c.text, c
    assert S.texts(got, "answer") == "Ok "
    assert S.tool_commits(got) == []


async def test_a_capped_run_still_breaks_a_spliced_tag_s10() -> None:
    """Kills S10 (``return think + quote``: a capped run of plain specials collapses to nothing). Chunk 1 ends in
    ``<|tool_c`` + ``<pad>`` x 13; chunk 2 is ``all>call:x{}``. Collapsing the run to "" would splice
    ``<|tool_call>`` and route it; the run must still separate the halves, so there is no ``tool`` commit and the
    spliced literal is removed from the answer (Q2): ``Ok call:x{}`` (hand; ``route`` agrees there is no tag)."""
    first = [*S.thought("t"), *S.prose("Ok <|tool_c"), *([PAD] * 13)]
    second = [*S.prose("all>call:x{}"), TURN_CLOSE]
    d = S.decode([*first, *second[:-1]])
    assert router.route(d).tag is None
    assert drop_special_literals(router.route(d).spoken_prefix) == "Ok call:x{}"
    got, _ = await S.run_chunks([first, second])
    assert S.tool_commits(got) == []
    assert S.texts(got, "answer") == "Ok " + "call:x{}"
