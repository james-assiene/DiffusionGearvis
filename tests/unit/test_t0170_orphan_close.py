"""T-0170: an orphan ``<channel|>`` ends an implicit thought (docs/tasks/T-0170.md, criteria 1-4, 7, 8, 10-12).

Every stream is built from ids by the wire fake (``tests/fakes/vllm_sse.py``, unchanged) and read by the real
``VllmClient`` over ``httpx.MockTransport`` with T-0088a's subset ``HostDetokenizer``, as ``test_llm_id_mode.py`` does
(``tests/unit/_llm_id_support``). Letters the subset lacks are spelled with its byte-fallback pieces, so
``<channel|>`` can be written in plain pieces (``<`` + ``channel`` + ``|>``) as well as as id 101.

**The oracle.** ``_B372ThoughtSplit`` and ``_B372IdReader`` below are verbatim copies of ``_ThoughtSplit`` and
``_IdReader`` at main b372edaa (``local_astra/conductor/llm_client.py`` lines 889-931 and 999-1061, sha256
``d552d298…`` of the file), renamed; ``b372_commits`` is the id-mode body of ``VllmClient.generate`` at b372edaa
(the HTTP and abort plumbing removed, every other line kept). The level-2 pieces they call (``_AnswerSplit``,
``split_canvas``, ``Piece``, ``FenceState``, ``drop_special_literals``, ``router``) are imported from the code under
test: the task leaves them unchanged (Implementation notes: "the hold lives in ``_ThoughtSplit``, before
``_AnswerSplit``"). ``test_frozen_oracle_is_b372edaa`` proves the two classes are the b372edaa text.

**The criterion-2 reference** (``reference_commits``), written from the criterion and not from any implementation:
decode D up to the first stop; if the first of ``<|channel>``, ``<channel|>`` and the ``router.TAGS`` literals in D
is ``<channel|>``, D is read as ``<|channel>`` + D by b372edaa (which opens a thought at the start, drops a leading
``thought\\n`` header, ends the thought at that very ``<channel|>`` and reads the rest with no thought open: exactly
criterion 2's three clauses); otherwise D is read by b372edaa as is ("the output equals today's"). The reference
keeps the row's chunk partition (the prepended ``<|channel>`` joins the first chunk), so the UI fence's per-canvas
bound is spent identically on both sides.

What criterion 4(a) compares: the concatenated ``thought`` text, the concatenated ``answer`` text, the ``ui`` commits
(text, ending, continued) and the ``tool`` commits (text, stop id). Canvas indexes and per-piece chunking are not
compared: the hold legitimately moves them.

Readings where the task is silent (listed in the test-author report):

- ``Commit.implicit_thought`` is read off the model fields; its absence is a named ``AssertionError`` (rows stay
  red on their own assertion before T-0170, never on a ``ValidationError``);
- implicit and explicit thought text never share one ``thought`` commit (criterion 1: the flag is set on every
  thought commit "whose text came from an implicit thought", so a commit is wholly one or the other);
- "before the first channel marker or tag has been fed" is measured by the chunk the wire had sent when the commit
  was yielded (``Wire.on_chunk``); the decision chunk is the first chunk after which the decode so far contains a
  marker, or the final chunk when D has none.

Domains (criterion 4): prose, fragments that spell markers, every special except the stops and ``<|think|>`` (M9
makes a whole content piece thought per call, which is partition-dependent and not this task's), with or without an
orphan close (as id 101 or spelled), an explicit block, a call tag (special or spelled) or a UI fence. The UI fence
is only placed after a channel marker (right after the orphan close or right after an explicit block): a fence
written before any marker is held whole by the hold and then spends the per-canvas bound differently from a reading
canvas by canvas, which the criterion does not rule on (reported).

Regime: CPU over fakes, no GPU, no network; Hypothesis default seed; n = 1 per example.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from local_astra.conductor import llm_client
from local_astra.conductor.llm_client import (
    _CHANNEL_CLOSE_PREFIXES,
    _CHANNEL_OPEN_PREFIXES,
    SSE_DONE,
    THOUGHT_HEADER,
    FenceState,
    LlmWireError,
    Piece,
    _AnswerSplit,
    _delta_fields,
    _first_choice,
    _sse_payload,
    _tail,
)
from local_astra.contracts.llm import Commit, Message, TextPart
from local_astra.toolcall import router
from local_astra.toolcall.detok import HostDetokenizer, UnknownTokenId
from local_astra.toolcall.special_tokens import SPECIAL_IDS, STOP_IDS, drop_special_literals
from tests.fakes import vllm_sse as V
from tests.unit import _llm_id_support as S
from tests.unit._llm_id_support import (
    CHANNEL_CLOSE,
    CHANNEL_OPEN,
    TOOL_CALL,
    TOOL_CALL_CLOSE,
    TOOL_RESPONSE,
    TURN_CLOSE,
)

REPO = Path(__file__).resolve().parents[2]
#: T-0170 branched from main at b372edaa (task file). A literal, never a moving ref (tests.md).
BASE_SHA = "b372edaa"

CALL_TEXT = "call:click_at{x:1,y:2}"
CALL = f"<|tool_call>{CALL_TEXT}<tool_call|>"
#: criterion 2's window markers, as text in D
MARKERS: tuple[str, ...] = (router.CHANNEL_OPEN, router.CHANNEL_CLOSE, *router.TAGS.values())


# =========================================================================== the frozen b372edaa oracle
# Copied verbatim from local_astra/conductor/llm_client.py at b372edaa (lines 889-931), class renamed.


class _B372ThoughtSplit:
    """Level 1: D in, thought texts and answer-channel text out (thoughts removed as ``router`` removes them)."""

    def __init__(self) -> None:
        self.in_thought = False
        #: the current thought's leading ``thought\n`` header is not decided yet
        self._header = False
        self._held = ""

    def feed(self, text: str, final: bool) -> tuple[list[str], str]:
        thoughts: list[str] = []
        answer: list[str] = []
        text, self._held = self._held + text, ""
        while text:
            if not self.in_thought:
                i = text.find(router.CHANNEL_OPEN)
                if i < 0:
                    keep = 0 if final else _tail(text, _CHANNEL_OPEN_PREFIXES)
                    answer.append(text[: len(text) - keep])
                    self._held = text[len(text) - keep :]
                    break
                answer.append(text[:i])
                text = text[i + len(router.CHANNEL_OPEN) :]
                self.in_thought, self._header = True, True
                continue
            if self._header:
                if text.startswith(THOUGHT_HEADER):
                    text = text[len(THOUGHT_HEADER) :]
                elif THOUGHT_HEADER.startswith(text) and not final:
                    self._held = text
                    break
                self._header = False
                continue
            j = text.find(router.CHANNEL_CLOSE)
            if j < 0:
                keep = 0 if final else _tail(text, _CHANNEL_CLOSE_PREFIXES)
                thoughts.append(text[: len(text) - keep])
                self._held = text[len(text) - keep :]
                break
            thoughts.append(text[:j])
            text = text[j + len(router.CHANNEL_CLOSE) :]
            self.in_thought = False
        return [t for t in thoughts if t], "".join(answer)


# Copied verbatim from local_astra/conductor/llm_client.py at b372edaa (lines 999-1061), class renamed and its
# ``_ThoughtSplit()`` pointed at the frozen copy above.


class _B372IdReader:
    """One tools-on response read by its ids. A local of the ``generate`` frame (T-0052 D1)."""

    def __init__(self, detokenizer: HostDetokenizer, tool_prefix: str | None) -> None:
        self._stream = detokenizer.stream()
        self._thoughts = _B372ThoughtSplit()
        self._answer = _AnswerSplit(tool_prefix)
        self._decoding = True
        #: the first id of ``STOP_IDS`` in the stream (never ``stop_reason``); None for a length cut
        self.stop_id: int | None = None
        #: a chunk with a non-null ``finish_reason`` has been read
        self.finished = False
        #: index of the last canvas that carried ids (E6: the ``tool`` commit's ``canvas_idx``)
        self.last_canvas = 0

    @property
    def fence(self) -> FenceState:
        return self._answer.fence

    def feed(self, payload: object, canvas_idx: int) -> tuple[list[Piece], bool]:
        """One chunk: its pieces, and whether the canvas ends inside a thought. Raises ``LlmWireError`` before
        anything of the chunk is returned."""
        choice = _first_choice(payload)
        reasoning, content = _delta_fields(payload)
        finish = None if choice is None else choice.get("finish_reason")
        ids = None if choice is None else choice.get("token_ids")
        if ids is None:
            if reasoning or content or finish is not None:
                raise LlmWireError("a chunk with text or a finish_reason carries no token_ids")
            return [], self._thoughts.in_thought  # the role chunk
        if not isinstance(ids, list) or not all(type(t) is int for t in ids):
            raise LlmWireError(f"token_ids is not a list of ints: {ids!r:.120}")
        self.last_canvas = canvas_idx
        text = ""
        final = False
        if self._decoding:
            stop = next((k for k, t in enumerate(ids) if t in STOP_IDS), None)
            try:
                text = self._stream.feed(ids if stop is None else ids[:stop])
            except UnknownTokenId as exc:
                raise LlmWireError(str(exc)) from exc
            if stop is not None:
                self.stop_id = ids[stop]
            if stop is not None or finish is not None:
                # ids after the stop are not this response's
                text += self._stream.finish()
                self._decoding, final = False, True
        if finish is not None:
            self.finished = True
        thoughts, answer = self._thoughts.feed(text, final)
        pieces = [Piece("thought", t) for t in thoughts]
        pieces += self._answer.feed(answer, final)
        return pieces, self._thoughts.in_thought

    def tool_text(self) -> str | None:
        """The ``tool`` commit's text: from the tag to the stop, thoughts removed, as ``router.route`` reads it."""
        raw = self._answer.tool_raw()
        if raw is None:
            return None
        found = router.route(raw)
        if found.tag is None:
            return raw
        return router.TAGS[found.tag] + found.draft_call_text + found.draft_close


def b372_commits(
    lines: Sequence[str], *, tool_prefix: str | None = None, request_id: str = "r1"
) -> list[Commit]:
    """The commits b372edaa's ``VllmClient.generate`` yields for ``lines`` in id mode: its loop body, verbatim but for
    the HTTP context, ``record.is_aborted()`` (never true here) and ``yield`` (append)."""
    reader = _B372IdReader(S.detok(), tool_prefix)
    fence = FenceState()
    published = 0
    canvas_idx = 0
    out: list[Commit] = []
    for line in lines:
        payload = _sse_payload(line)
        if payload is None:
            continue
        if payload == SSE_DONE:
            break
        pieces, thought_open = reader.feed(json.loads(payload), canvas_idx)
        fence = reader.fence
        if not pieces:
            continue
        emitted = False
        for piece in pieces:
            if not piece.text:
                continue
            if piece.segment == "ui":
                if piece.fence == "open" and len(piece.text) <= published:
                    continue  # the carry grew by nothing: no news to publish
                published = len(piece.text) if piece.fence == "open" else 0
            out.append(
                Commit(
                    request_id=request_id,
                    canvas_idx=canvas_idx,
                    segment=piece.segment,
                    text=piece.text,
                    thought_open=thought_open,
                    fence_close=piece.fence,
                    fence_continued=piece.continued,
                )
            )
            emitted = True
        if emitted:
            canvas_idx += 1
    if not reader.finished:
        raise LlmWireError(f"{request_id}: the stream ended without a finish_reason")
    shown = drop_special_literals(fence.body)
    if fence.open and len(shown) > published:
        out.append(
            Commit(
                request_id=request_id,
                canvas_idx=canvas_idx,
                segment="ui",
                text=shown,
                fence_close="open",
                fence_continued=fence.canvases > 1,
            )
        )
    tool = reader.tool_text()
    if tool is not None:
        out.append(
            Commit(
                request_id=request_id,
                canvas_idx=reader.last_canvas,
                segment="tool",
                text=tool,
                stop_id=reader.stop_id,
            )
        )
    return out


# =========================================================================== helpers


def implicit(c: Any) -> bool:
    """``c.implicit_thought``; a ``Commit`` without the field is a named failure (criterion 1)."""
    if "implicit_thought" not in type(c).model_fields:
        raise AssertionError("T-0170 criterion 1: Commit has no field `implicit_thought` yet")
    return bool(c.implicit_thought)


def implicit_or_false(c: Any) -> bool:
    """For rows that hold before T-0170: the field reads False while it does not exist."""
    return bool(getattr(c, "implicit_thought", False))


def full(c: Any) -> tuple[Any, ...]:
    """Criterion 8's commit identity."""
    return (
        c.segment,
        c.text,
        c.thought_open,
        c.canvas_idx,
        c.fence_close,
        c.fence_continued,
        c.stop_id,
        implicit_or_false(c),
    )


def spelled_close() -> list[int]:
    """``<channel|>`` in plain pieces: ``<``, the letters of ``channel`` (byte fallback where the subset lacks them),
    ``|>``."""
    return [*S.ids("<"), *S.prose("channel"), *S.ids("|>")]


def spelled_open() -> list[int]:
    return [*S.ids("<", "|"), *S.prose("channel"), *S.ids(">")]


def decoded(seq: Sequence[int]) -> str:
    """D: the decode up to the first stop id."""
    ids, _stop = V.until_stop(list(seq))
    return S.decode(ids)


def first_marker(d: str) -> tuple[int, str] | None:
    found = [(d.find(m), m) for m in MARKERS if d.find(m) >= 0]
    return min(found) if found else None


def orphan_first(d: str) -> bool:
    fm = first_marker(d)
    return fm is not None and fm[1] == router.CHANNEL_CLOSE


def strip_header(text: str) -> str:
    return text[len(THOUGHT_HEADER) :] if text.startswith(THOUGHT_HEADER) else text


def reference_commits(
    chunks: Sequence[Sequence[int]], *, tool_prefix: str | None = None
) -> list[Commit]:
    """Criterion 2 as a reading of D by b372edaa (module docstring)."""
    d = decoded([t for c in chunks for t in c])
    if tool_prefix is None and orphan_first(d):
        chunks = [[CHANNEL_OPEN, *chunks[0]], *chunks[1:]]
    return b372_commits(S.stream_lines(chunks), tool_prefix=tool_prefix)


def summary(commits: Sequence[Any]) -> dict[str, Any]:
    return {
        "thought": S.texts(commits, "thought"),
        "answer": S.texts(commits, "answer"),
        "ui": [(c.text, c.fence_close, c.fence_continued) for c in commits if c.segment == "ui"],
        "tool": [(c.text, c.stop_id) for c in commits if c.segment == "tool"],
    }


def decision_chunk(chunks: Sequence[Sequence[int]]) -> int:
    """The first chunk after which D so far holds a window marker; the final chunk when D has none (criterion 3)."""
    stream = S.detok().stream()
    text = ""
    for k, chunk in enumerate(chunks):
        stop = next((i for i, t in enumerate(chunk) if t in STOP_IDS), None)
        text += stream.feed(list(chunk if stop is None else chunk[:stop]))
        if stop is not None:
            text += stream.finish()
        if any(m in text for m in MARKERS) or stop is not None:
            return k
    return len(chunks) - 1


async def run_tracked(
    chunks: Sequence[Sequence[int]], prompt: Any = None, *, abort_before: int | None = None
) -> list[tuple[Any, int]]:
    """Serve ``chunks`` through the real client; each commit with the index of the chunk the wire had sent when it
    was yielded. ``abort_before=k``: ``abort`` is called just before chunk ``k`` is sent."""
    lines = S.stream_lines(chunks)
    sent = [0]
    holder: list[Any] = []

    async def on_chunk(k: int) -> None:
        sent[0] = k
        if abort_before is not None and k == abort_before + 1:
            await holder[0].abort("r1")

    wire = V.Wire(streams=[lines], on_chunk=on_chunk)
    cl = S.client(wire.transport())
    holder.append(cl)
    out: list[tuple[Any, int]] = []
    try:
        async for c in cl.generate(prompt if prompt is not None else S.tools_prompt(), "r1"):
            out.append((c, sent[0] - 1))  # line 0 is the role chunk
    finally:
        await cl.aclose()
    return out


def tracked(chunks: Sequence[Sequence[int]], prompt: Any = None) -> list[tuple[Any, int]]:
    return asyncio.run(run_tracked(chunks, prompt))


def cuts(seq: Sequence[int]) -> list[list[list[int]]]:
    """The one-chunk stream and every two-chunk partition of ``seq``."""
    return [[list(seq)], *([list(seq[:k]), list(seq[k:])] for k in range(1, len(seq)))]


def per_id(seq: Sequence[int]) -> list[list[int]]:
    return [[t] for t in seq]


def assert_no_early_release(
    got: Sequence[tuple[Any, int]], chunks: Sequence[Sequence[int]]
) -> None:
    k = decision_chunk(chunks)
    early = [(c.segment, c.text, at) for c, at in got if c.segment in ("answer", "ui") and at < k]
    assert not early, (
        f"T-0170 criterion 3: no answer/ui piece before the first channel marker or tag (chunk {k} of "
        f"{len(chunks)}); yielded early: {early}"
    )


# =========================================================================== criterion 2: the rule, by hand


async def _orphan_rows(seq: list[int], thought: str, answer: str) -> None:
    for chunks in [*cuts(seq), per_id(seq)]:
        got, _ = await S.run_chunks(chunks)
        assert S.texts(got, "answer") == answer, (
            f"T-0170 criterion 2: the text after the orphan close is the answer; chunks {chunks}"
        )
        assert S.texts(got, "thought") == thought, (
            f"T-0170 criterion 2: the text before the orphan close is thought; chunks {chunks}"
        )
        thoughts = [c for c in got if c.segment == "thought"]
        assert all(implicit(c) for c in thoughts), (
            f"T-0170 criterion 1: every thought commit of an implicit thought is marked; "
            f"{[(c.text, implicit(c)) for c in thoughts]}"
        )
        assert not any(implicit(c) for c in got if c.segment != "thought")
        assert S.tool_commits(got) == []


async def test_orphan_close_first_makes_prefix_thought() -> None:
    """``A B<channel|>C D``: thought "A B" marked ``implicit_thought``, answer "C D" (criterion 2), at every two-chunk
    cut and one id per chunk; and the same with the close spelled in plain pieces."""
    for close in ([CHANNEL_CLOSE], spelled_close()):
        seq = [*S.prose("A B"), *close, *S.prose("C D"), TURN_CLOSE]
        assert decoded(seq) == "A B<channel|>C D"
        await _orphan_rows(seq, "A B", "C D")


async def test_orphan_close_strips_leading_header() -> None:
    """``thought\\nX<channel|>Y``: thought "X" (the header is removed as for an explicit thought), answer "Y"."""
    seq = [*S.ids("thought", "\n"), *S.prose("X"), CHANNEL_CLOSE, *S.prose("Y"), TURN_CLOSE]
    assert decoded(seq) == "thought\nX<channel|>Y"
    await _orphan_rows(seq, "X", "Y")


async def test_orphan_close_after_a_bare_header_is_an_empty_thought() -> None:
    """``thought\\n<channel|>Hi.``: the header alone before the close is an empty implicit thought, so nothing of it
    is answer (today the answer is ``thought\\nHi.``: the header is voiced)."""
    seq = [*S.ids("thought", "\n"), CHANNEL_CLOSE, *S.prose("Hi."), TURN_CLOSE]
    for chunks in cuts(seq):
        got, _ = await S.run_chunks(chunks)
        assert S.texts(got, "answer") == "Hi.", chunks
        assert "thought" not in S.texts(got, "answer"), chunks
        assert all(implicit(c) for c in got if c.segment == "thought"), chunks


async def test_open_first_unchanged() -> None:
    """``<|channel>thought\\nT<channel|>A``: as b372edaa, commit for commit (criteria 2 and 8)."""
    seq = [*S.thought("T"), *S.prose("A."), TURN_CLOSE]
    for chunks in [*cuts(seq), per_id(seq)]:
        got, _ = await S.run_chunks(chunks)
        assert (S.texts(got, "thought"), S.texts(got, "answer")) == ("T", "A."), chunks
        assert [full(c) for c in got] == [full(c) for c in b372_commits(S.stream_lines(chunks))], (
            chunks
        )
        assert not any(implicit_or_false(c) for c in got)


async def test_tag_first_unchanged() -> None:
    """A call tag before any channel marker: the window ends at the tag and the output is b372edaa's, including
    a ``<channel|>`` after the tag (inside the tool text) and answer text before the tag."""
    for lead in ("", "I will."):
        seq = [
            *S.prose(lead),
            *S.call(),
            *S.prose(" ok"),
            CHANNEL_CLOSE,
            *S.prose("x"),
            TOOL_RESPONSE,
        ]
        for chunks in [*cuts(seq)[::3], per_id(seq)]:
            got, _ = await S.run_chunks(chunks)
            want = b372_commits(S.stream_lines(chunks))
            assert summary(got) == summary(want), chunks
            assert S.texts(got, "answer") == lead, chunks
            assert S.texts(got, "thought") == "", chunks
            assert not any(implicit_or_false(c) for c in got)


async def test_tag_inside_a_fence_before_any_marker_ends_the_window() -> None:
    """Implementation notes: a tag inside a UI fence before any channel marker also ends the window, so a later
    ``<channel|>`` is a stray close, read as b372edaa reads it (deleted from the answer, E8): nothing is thought."""
    seq = [
        *S.ids("<", "a", "2", "ui", ">"),
        *S.prose('{"a":"'),
        TOOL_CALL,
        *S.prose('"}'),
        *S.ids("</", "a", "2", "ui", ">"),
        *S.prose("Then"),
        CHANNEL_CLOSE,
        *S.prose("B."),
        TURN_CLOSE,
    ]
    got, _ = await S.run_chunks([seq])
    assert S.texts(got, "thought") == ""
    assert S.texts(got, "answer") == "ThenB."
    assert [(c.text, c.fence_close) for c in got if c.segment == "ui"] == [('{"a":""}', "written")]
    assert summary(got) == summary(b372_commits(S.stream_lines([seq])))


async def test_no_marker_unchanged() -> None:
    """No channel marker and no tag: the answer is D, as b372edaa (criterion 2, "nothing comes first")."""
    seq = [*S.prose("Hello there."), TURN_CLOSE]
    for chunks in [*cuts(seq)[::2], per_id(seq)]:
        got, _ = await S.run_chunks(chunks)
        assert S.texts(got, "answer") == "Hello there.", chunks
        assert summary(got) == summary(b372_commits(S.stream_lines(chunks))), chunks
        assert [c.segment for c in got] == ["answer"] * len(got)


async def test_second_stray_close_after_block_unchanged() -> None:
    """``<|channel>thought\\nT<channel|>A<channel|>B``: answer "AB", as today (criterion 2: later stray closes keep
    today's behaviour)."""
    seq = [*S.thought("T"), *S.prose("A"), CHANNEL_CLOSE, *S.prose("B"), TURN_CLOSE]
    for chunks in [*cuts(seq), per_id(seq)]:
        got, _ = await S.run_chunks(chunks)
        assert S.texts(got, "answer") == "AB", chunks
        assert S.texts(got, "thought") == "T", chunks
        assert [full(c) for c in got] == [full(c) for c in b372_commits(S.stream_lines(chunks))], (
            chunks
        )


async def test_second_stray_close_after_an_orphan_is_a_stray_close() -> None:
    """``A<channel|>B<channel|>C``: only the first close ends the implicit thought; the second is read as today with no
    thought open (deleted from the answer, E8): thought "A", answer "BC"."""
    seq = [*S.prose("A"), CHANNEL_CLOSE, *S.prose("B"), CHANNEL_CLOSE, *S.prose("C"), TURN_CLOSE]
    for chunks in [*cuts(seq), per_id(seq)]:
        got, _ = await S.run_chunks(chunks)
        assert (S.texts(got, "thought"), S.texts(got, "answer")) == ("A", "BC"), chunks


# =========================================================================== criterion 3: the hold


async def test_held_text_released_at_end() -> None:
    """No marker and no tag: nothing is yielded before the final chunk, and the answer then has the same bytes as
    b372edaa's single-chunk reading. Both ends: a stop id and a length cut (``finish_reason: length``)."""
    for tail_ids in ([TURN_CLOSE], []):
        seq = [*S.prose("Hello there, "), *S.prose("how are you?"), *tail_ids]
        chunks = V.partition(seq, 4)
        got = await run_tracked(chunks)
        last = len(chunks) - 1
        early = [(c.segment, c.text, at) for c, at in got if at < last]
        assert not early, (
            f"T-0170 criterion 3: text that opened no thought is held until the end of the response; "
            f"yielded before chunk {last}: {early}"
        )
        want = S.texts(b372_commits(S.stream_lines([seq])), "answer")
        assert want == "Hello there, how are you?"
        assert S.texts([c for c, _ in got], "answer") == want


async def test_held_text_is_never_released_on_abort() -> None:
    """An abort before the deciding chunk: no answer (nor anything else) is ever yielded (criterion 3)."""
    chunks = [
        S.prose("I think "),
        S.prose("the user "),
        [CHANNEL_CLOSE, *S.prose("Hi."), TURN_CLOSE],
    ]
    for at in (1, 2):
        got = await run_tracked(chunks, abort_before=at)
        assert [c for c, _ in got if c.segment == "answer"] == [], (
            f"T-0170 criterion 3: held text is never released on abort (abort before chunk {at}); got "
            f"{[(c.segment, c.text) for c, _ in got]}"
        )


async def test_held_text_released_at_the_first_open_or_tag() -> None:
    """The held text is released unchanged at the first ``<|channel>`` or call tag, as today's split produces it."""
    for marker in (S.thought("T"), S.call()):
        seq = [
            *S.prose("Sure. "),
            *S.prose("One moment."),
            *marker,
            *S.prose(" Ok."),
            TOOL_RESPONSE,
        ]
        chunks = [
            S.prose("Sure. "),
            S.prose("One moment."),
            [*marker, *S.prose(" Ok."), TOOL_RESPONSE],
        ]
        got = await run_tracked(chunks)
        assert_no_early_release(got, chunks)
        assert summary([c for c, _ in got]) == summary(b372_commits(S.stream_lines([seq])))
        assert S.texts([c for c, _ in got], "answer").startswith("Sure. One moment.")


async def test_close_spelled_across_chunks() -> None:
    """``A B<chan`` | ``nel|>C D``: the close spelled in plain pieces across two chunks is the marker (E7/E11 holding
    in the window); nothing is answer before it."""
    close = spelled_close()
    seq = [*S.prose("A B"), *close, *S.prose("C D"), TURN_CLOSE]
    lead = len(S.prose("A B"))
    for k in range(1, len(close)):
        chunks = [seq[: lead + k], seq[lead + k :]]
        got = await run_tracked(chunks)
        assert_no_early_release(got, chunks)
        commits = [c for c, _ in got]
        assert (S.texts(commits, "thought"), S.texts(commits, "answer")) == ("A B", "C D"), chunks
        assert all(implicit(c) for c in commits if c.segment == "thought")
    chunks = per_id(seq)
    got = await run_tracked(chunks)
    assert_no_early_release(got, chunks)
    assert S.texts([c for c, _ in got], "answer") == "C D"


@st.composite
def _late_orphan(draw: st.DrawFn) -> tuple[list[list[int]], str, str]:
    """Answer-looking text over several chunks, then the orphan close in the last chunk (the adversarial shape)."""
    words = draw(
        st.lists(
            st.sampled_from(["▁I", "▁the", "▁a", "▁b", ".", "A", "B", "é", "Hi", "Here"]),
            min_size=2,
            max_size=12,
        )
    )
    head = S.ids(*words)
    cuts_ = sorted(draw(st.sets(st.integers(1, len(head) - 1), max_size=len(head) - 1)))
    bounds = [0, *cuts_, len(head)]
    chunks = [head[a:b] for a, b in itertools.pairwise(bounds)]
    close = draw(st.sampled_from([[CHANNEL_CLOSE], spelled_close()]))
    tail = draw(st.sampled_from(["Hi.", " Sure.", "B"]))
    stop = draw(st.sampled_from(sorted(STOP_IDS)))
    chunks.append([*close, *S.prose(tail), stop])
    return chunks, S.decode(head), tail


@settings(
    max_examples=100,
    deadline=None,
    report_multiple_bugs=False,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(_late_orphan())
def test_no_answer_before_decision_any_partition(case: tuple[list[list[int]], str, str]) -> None:
    """Criterion 3 (adversarial): the orphan close is in the last chunk; no ``answer`` or ``ui`` piece is yielded in
    any earlier chunk, for any partition of the text before it; then the text before it is the (implicit) thought
    and the text after it the answer."""
    chunks, head, tail = case
    got = tracked(chunks)
    assert_no_early_release(got, chunks)
    commits = [c for c, _ in got]
    assert S.texts(commits, "answer") == tail
    assert S.texts(commits, "thought") == head


# =========================================================================== criterion 4: the property

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
#: plain pieces markers are spelled from (``ui`` with ``gen`` spells ``<genui>``; no ``a``/``2``: fences are
#: placed deliberately, never spelled by chance)
FRAGMENTS = ["<", "|", "|>", ">", "tool", "cc", "gen", "ui", "▁<"]
#: every special but the stops and ``<|think|>`` (module docstring)
SPECIALS = [
    lit for i, lit in sorted(SPECIAL_IDS.items()) if i not in STOP_IDS and lit != "<|think|>"
]
TAG_SPELLINGS: list[list[str]] = [
    ["<|tool_call>"],
    ["<", "|", "tool", "_", "call", ">"],
    ["<", "cc", ">"],
    ["<", "gen", "ui", ">"],
]
FENCE = ["<", "a", "2", "ui", ">", '{"', "a", '":', "1", "}", "</", "a", "2", "ui", ">"]


def _pieces(draw: st.DrawFn, max_size: int, *, prose_only: bool = False) -> list[int]:
    pool = PROSE if prose_only else [*PROSE, *FRAGMENTS, *SPECIALS]
    return S.ids(*draw(st.lists(st.sampled_from(pool), max_size=max_size)))


@st.composite
def _d_ids(draw: st.DrawFn) -> list[int]:
    """One D (as ids, ending in a stop): criterion 4's domain (module docstring)."""
    lead = S.ids("thought", "\n") if draw(st.integers(0, 3)) == 0 else []
    p = _pieces(draw, 8, prose_only=draw(st.booleans()))
    close_kind = draw(st.sampled_from(["id", "spelled", "none"]))
    close = {"id": [CHANNEL_CLOSE], "spelled": spelled_close(), "none": []}[close_kind]
    r = _pieces(draw, 10)
    extra = draw(st.sampled_from(["none", "block", "tag", "fence"]))
    if extra == "block":
        at = draw(st.integers(0, len(r)))
        block = [*S.thought("t"), *(S.ids(*FENCE) if draw(st.booleans()) else [])]
        r[at:at] = block
    elif extra == "tag":
        at = draw(st.integers(0, len(r)))
        tag = S.ids(*draw(st.sampled_from(TAG_SPELLINGS)))
        r[at:at] = [
            *tag,
            *S.ids(*draw(st.lists(st.sampled_from(PROSE), max_size=4))),
            TOOL_CALL_CLOSE,
        ]
    elif extra == "fence" and close:
        r[0:0] = S.ids(*FENCE)
    stop = draw(st.sampled_from(sorted(STOP_IDS)))
    return [*lead, *p, *close, *r, stop]


@st.composite
def _d_case(draw: st.DrawFn) -> tuple[list[int], list[list[int]]]:
    seq = draw(_d_ids())
    if len(seq) == 1:
        return seq, [seq]
    cuts_ = sorted(draw(st.sets(st.integers(1, len(seq) - 1), max_size=len(seq) - 1)))
    bounds = [0, *cuts_, len(seq)]
    return seq, [seq[a:b] for a, b in itertools.pairwise(bounds)]


def _check_against_reference(chunks: list[list[int]]) -> None:
    d = decoded([t for c in chunks for t in c])
    got = tracked(chunks)
    commits = [c for c, _ in got]
    want = reference_commits(chunks)
    assert summary(commits) == summary(want), (
        f"T-0170 criterion 4(a): the split equals criterion 2's reference on D={d!r} "
        f"(orphan first: {orphan_first(d)})"
    )
    # criterion 1: exactly the implicit thought's text is marked
    if orphan_first(d):
        fm = first_marker(d)
        assert fm is not None
        marked = "".join(c.text for c in commits if c.segment == "thought" and implicit(c))
        assert marked == strip_header(d[: fm[0]]), (
            f"T-0170 criterion 1: the marked thought text is D before the orphan close; D={d!r}"
        )
    else:
        assert not any(implicit(c) for c in commits), f"no implicit thought in D={d!r}"
    assert not any(implicit(c) for c in commits if c.segment != "thought")
    # criterion 4(b)
    assert_no_early_release(got, chunks)


@settings(
    max_examples=200,
    deadline=None,
    report_multiple_bugs=False,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(_d_case())
def test_split_matches_reference_any_partition(case: tuple[list[int], list[list[int]]]) -> None:
    """Criterion 4: for any D of the domain and any partition, (a) thought / answer / ui / tool outputs equal
    criterion 2's reference and (b) no answer or ui commit before the first channel marker or tag has been fed."""
    _seq, chunks = case
    _check_against_reference(chunks)


@settings(
    max_examples=100,
    deadline=None,
    report_multiple_bugs=False,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(_d_ids())
def test_split_matches_reference_one_id_per_chunk(seq: list[int]) -> None:
    """Criterion 4 with one id per chunk (the partition that exercises every partial-marker hold)."""
    _check_against_reference(per_id(seq))


# =========================================================================== criterion 8: the normal case


@st.composite
def _normal_case(draw: st.DrawFn) -> list[list[int]]:
    """D whose first channel marker is ``<|channel>`` with no text before it, then anything (``<|think|>`` and fences
    included: both sides read the same partition)."""
    opener = draw(st.sampled_from([[CHANNEL_OPEN], spelled_open()]))
    body = S.ids(
        *draw(
            st.lists(
                st.sampled_from([*PROSE, *FRAGMENTS, *SPECIALS, "<|think|>", "thought", "\n"]),
                max_size=10,
            )
        )
    )
    seq = [*opener, *body, *draw(_d_ids())]
    if draw(st.booleans()):
        return per_id(seq)
    cuts_ = sorted(draw(st.sets(st.integers(1, len(seq) - 1), max_size=len(seq) - 1)))
    bounds = [0, *cuts_, len(seq)]
    return [seq[a:b] for a, b in itertools.pairwise(bounds)]


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_normal_case())
def test_normal_commits_identical_to_frozen_oracle(chunks: list[list[int]]) -> None:
    """Criterion 8: when D opens with ``<|channel>``, every yielded ``Commit`` (segment, text, ``thought_open``,
    ``canvas_idx``, ``fence_close``, ``fence_continued``, ``stop_id``; ``implicit_thought`` false) is b372edaa's, for
    any partition. Holds before T-0170 (active row)."""
    got = S.run(chunks)
    want = b372_commits(S.stream_lines(chunks))
    assert [full(c) for c in got] == [full(c) for c in want], decoded(
        [t for c in chunks for t in c]
    )


# =========================================================================== criterion 10: the continued call


async def test_continued_call_exempt() -> None:
    """Criterion 10: a continuation whose assistant text ends in the tag starts in the tool state and never enters
    implicit mode. ``call:click_at{x:1,<channel|>y:2}<tool_call|>``: no thought, no answer, and the ``tool`` commit
    is b372edaa's (``router.route`` of the tag plus the output: the parity definition, T-0085)."""
    prompt = S.continuation_prompt("I will click it.<|tool_call>")
    out = [
        *S.prose("call:click_at{x:1,"),
        CHANNEL_CLOSE,
        *S.prose("y:2}"),
        TOOL_CALL_CLOSE,
        TOOL_RESPONSE,
    ]
    raw = "<|tool_call>call:click_at{x:1,<channel|>y:2}<tool_call|>"
    r = router.route(raw)
    assert r.tag == "native"
    want_tool = router.TAGS["native"] + r.draft_call_text + r.draft_close
    assert "call:click_at{x:1," in want_tool, "premise: the text before the close is call text"
    for chunks in [*cuts(out), per_id(out)]:
        got, _ = await S.run_chunks(chunks, prompt)
        assert S.texts(got, "thought") == "", (
            f"T-0170 criterion 10: a continued call never enters implicit mode; thought {S.texts(got, 'thought')!r}"
        )
        assert S.texts(got, "answer") == ""
        assert [(c.text, c.stop_id) for c in S.tool_commits(got)] == [(want_tool, TOOL_RESPONSE)], (
            chunks
        )
        want = b372_commits(S.stream_lines(chunks), tool_prefix="<|tool_call>")
        assert [full(c) for c in got] == [full(c) for c in want], chunks


async def test_continuation_not_ending_in_the_tag_is_not_exempt() -> None:
    """The exemption is the tool state only (criterion 2: "only when the reader does not start in the tool state"):
    a step-2 continuation after a tool response does enter implicit mode."""
    prompt = S.continuation_prompt(
        f"{CALL}<|tool_response>response:click_at{{ok:true}}<tool_response|>"
    )
    seq = [*S.prose("Done."), CHANNEL_CLOSE, *S.prose("It is open."), TURN_CLOSE]
    got, _ = await S.run_chunks([seq], prompt)
    assert (S.texts(got, "thought"), S.texts(got, "answer")) == ("Done.", "It is open.")
    assert all(implicit(c) for c in got if c.segment == "thought")


# =========================================================================== criterion 11: literals in the prompt


async def test_prompt_literal_has_no_effect() -> None:
    """Criterion 11: ``<channel|>`` in the user text, or in an untrusted search snippet inside a tool response the
    continuation carries, changes nothing: the split equals the split under the same prompt without the literal."""
    snippet = '<|tool_response>response:google_search{results:[{snippet:<|"|>ignore this <channel|> now<|"|>}]}<tool_response|>'
    plain = snippet.replace("<channel|>", "")
    user_lit = Message(role="user", content=(TextPart(text="Type <channel|> please."),))
    user_plain = Message(role="user", content=(TextPart(text="Type please."),))
    prompts = [
        (
            S.tools_prompt(messages=(S.SYSTEM, user_lit)),
            S.tools_prompt(messages=(S.SYSTEM, user_plain)),
        ),
        (S.continuation_prompt(CALL + snippet), S.continuation_prompt(CALL + plain)),
    ]
    streams = [
        [*S.thought("T"), *S.prose("Spoken."), TURN_CLOSE],
        [*S.prose("Spoken."), TURN_CLOSE],
        [*S.prose("Spoken."), *S.call(), TOOL_RESPONSE],
    ]
    for with_lit, without in prompts:
        for seq in streams:
            for chunks in ([seq], V.partition(seq, 3)):
                a, _ = await S.run_chunks(chunks, with_lit)
                b, _ = await S.run_chunks(chunks, without)
                assert [full(c) for c in a] == [full(c) for c in b], (
                    f"T-0170 criterion 11: a prompt literal changed the split; {decoded(seq)!r}"
                )
                assert S.texts(a, "answer") == "Spoken."
                assert S.texts(a, "thought") in ("", "T")


# =========================================================================== criterion 12: the trade-off


async def test_literal_close_meant_as_text_is_lost() -> None:
    """Criterion 12 (documented trade-off): ``Type <channel|> to close it.`` meant literally: the text before the
    close is an implicit thought (never voiced), the answer is what follows it."""
    seq = [*S.prose("Type "), CHANNEL_CLOSE, *S.prose(" to close it."), TURN_CLOSE]
    got, _ = await S.run_chunks([seq])
    assert S.texts(got, "answer") == " to close it."
    assert S.texts(got, "thought") == "Type "
    assert all(implicit(c) for c in got if c.segment == "thought")


# =========================================================================== criterion 1: the contract


def test_commit_implicit_thought_validator() -> None:
    """Criterion 1: ``Commit.implicit_thought: bool = False``; ``True`` is allowed on a ``thought`` commit only (a
    validator refuses it elsewhere, following the ``stop_id`` precedent)."""
    assert "implicit_thought" in Commit.model_fields, (
        "T-0170 criterion 1: Commit has no field `implicit_thought`"
    )
    assert Commit.model_fields["implicit_thought"].default is False
    base = {"request_id": "r", "canvas_idx": 0, "text": "x"}
    ok = Commit(segment="thought", implicit_thought=True, **base)
    assert ok.implicit_thought is True
    for seg in ("thought", "answer", "ui", "tool"):
        assert Commit(segment=seg, **base).implicit_thought is False
        assert Commit(segment=seg, implicit_thought=False, **base).implicit_thought is False
    for seg in ("answer", "ui", "tool"):
        try:
            Commit(segment=seg, implicit_thought=True, **base)
        except ValidationError as exc:
            assert "implicit_thought" in str(exc) or "thought" in str(exc), str(exc)
        else:
            raise AssertionError(
                f"T-0170 criterion 1: implicit_thought=True is refused on a {seg} commit"
            )
    dumped = ok.model_dump(mode="json")
    assert dumped["implicit_thought"] is True and Commit.model_validate(dumped) == ok


# =========================================================================== criterion 7: tools off


def test_text_path_unchanged() -> None:
    """Criterion 7: tools off, ``split_canvas`` keeps b372edaa's pieces for content holding ``<channel|>`` (an orphan
    close is answer text there: vLLM's reasoning parser decides, H-T170-1(b)). Expected by hand from b372edaa's
    ``split_canvas``: no thought marker, no fence, nothing to hold or strip."""
    pieces, state = llm_client.split_canvas(None, "The user asks.<channel|>Hi.", FenceState())
    assert [(p.segment, p.text) for p in pieces] == [("answer", "The user asks.<channel|>Hi.")]
    assert state == FenceState()
    pieces, _ = llm_client.split_canvas("Plan.", "<channel|>Hi.", FenceState())
    assert [(p.segment, p.text) for p in pieces] == [
        ("thought", "Plan."),
        ("answer", "<channel|>Hi."),
    ]


async def test_text_path_client_unchanged() -> None:
    """Criterion 7 through the client (``reader is None``): two content chunks, the first with no marker, are
    yielded as they come, the second with its ``<channel|>`` kept (b372edaa's text path)."""
    sse = (
        "".join(
            "data: " + json.dumps({"choices": [{"delta": {"content": c}}]}) + "\n\n"
            for c in ("The user asks.", "<channel|>Hi.")
        )
        + "data: [DONE]\n\n"
    )
    cl = llm_client.VllmClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)
        )
    )
    got = await S.collect(cl, S.make_prompt())
    await cl.aclose()
    assert [(c.segment, c.text, c.canvas_idx) for c in got] == [
        ("answer", "The user asks.", 0),
        ("answer", "<channel|>Hi.", 1),
    ]
    assert not any(implicit_or_false(c) for c in got)


# =========================================================================== the oracle's provenance


def _b372_source() -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{BASE_SHA}:local_astra/conductor/llm_client.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def _class_source(text: str, name: str) -> str:
    m = re.search(rf"^class {name}\b.*?(?=^\S|\Z)", text, re.S | re.M)
    assert m is not None, name
    return m.group(0).rstrip()


def test_frozen_oracle_is_b372edaa() -> None:
    """The two oracle classes are b372edaa's text, but for the renames (skipped when the commit is absent)."""
    src = _b372_source()
    if src is None:
        pytest.skip(
            f"{BASE_SHA} is not in this clone (a squashed release); the copy cannot be re-checked"
        )
    mine = Path(__file__).read_text()
    for theirs, ours in (("_ThoughtSplit", "_B372ThoughtSplit"), ("_IdReader", "_B372IdReader")):
        expected = (
            _class_source(src, theirs)
            .replace(f"class {theirs}", f"class {ours}")
            .replace("_ThoughtSplit()", "_B372ThoughtSplit()")
        )
        assert _class_source(mine, ours) == expected, (
            f"the frozen {ours} is not b372edaa's {theirs}"
        )
    body = inspect.getsource(b372_commits)
    assert "pieces, thought_open = reader.feed(json.loads(payload), canvas_idx)" in body


# =========================================================================== reviews A1 / B-F1: the cross-chunk tail
# The window carries the last characters of what it has searched into the next chunk's search. If it carries fewer
# than ``len(longest marker) - 1``, a marker cut late inside it is never found: an orphan close cut after 8 of its
# 10 characters voices the thought, and a ``<|tool_call>`` cut after 11 characters is swallowed by a later orphan
# close. The client rows above cannot see this with the subset detokeniser (its byte-fallback lag means no fed chunk
# carries more than ~4 marker characters before a boundary), so these rows feed the two levels directly.


def split_direct(chunks: Sequence[str]) -> tuple[str, str, str, str | None]:
    """``(thought, implicit thought, answer, tool raw)`` from ``_ThoughtSplit(implicit=True)`` then ``_AnswerSplit``
    fed ``chunks`` (the last one final), as ``_IdReader`` chains them."""
    thoughts = llm_client._ThoughtSplit(implicit=True)
    answers = llm_client._AnswerSplit(None)
    thought, implicit_text, answer = "", "", ""
    for k, chunk in enumerate(chunks):
        final = k == len(chunks) - 1
        pieces, channel = thoughts.feed(chunk, final)
        for p in pieces:
            thought += p.text
            if getattr(p, "implicit_thought", False):
                implicit_text += p.text
        for p in answers.feed(channel, final):
            if p.segment == "answer":
                answer += p.text
    return thought, implicit_text, answer, answers.tool_raw()


#: (marker, text after it, expected (thought, implicit thought, answer, tool raw after the prefix)), by hand
_TAIL_CASES: list[tuple[str, str, tuple[str, str, str, str | None]]] = [
    (router.CHANNEL_CLOSE, "C D", ("{p}", "{p}", "C D", None)),
    (router.CHANNEL_OPEN, "thought\nT<channel|>C D", ("T", "", "{p}C D", None)),
    *(
        (tag, "call:x{}<channel|>B", ("", "", "{p}", tag + "call:x{}<channel|>B"))
        for tag in router.TAGS.values()
    ),
]


@pytest.mark.parametrize("prefix", ["A B", "The user asks."])
@pytest.mark.parametrize(
    ("marker", "after", "want"), _TAIL_CASES, ids=[m for m, _, _ in _TAIL_CASES]
)
def test_window_marker_cut_at_every_interior_point(
    prefix: str, marker: str, after: str, want: tuple[str, str, str, str | None]
) -> None:
    """Every window marker cut at every interior point between two chunks, whatever the number of marker characters
    before the cut (``<channel|>`` up to 9, ``<|tool_call>`` up to 11, then a later orphan close): the split equals
    the hand-written single-reading result."""
    text = prefix + marker + after
    expected = tuple(w.format(p=prefix) if isinstance(w, str) and "{p}" in w else w for w in want)
    for k in range(1, len(marker)):
        cut = len(prefix) + k
        got = split_direct([text[:cut], text[cut:]])
        assert got == expected, (
            f"T-0170 criterion 2/3: {marker!r} cut after {k} of its characters "
            f"({text[:cut]!r} | {text[cut:]!r}): got {got}, want {expected}"
        )


def test_window_marker_cut_over_three_chunks() -> None:
    """``A B<chann`` | ``el|`` | ``>C D``: a marker spread over three chunks is still found."""
    assert split_direct(["A B<chann", "el|", ">C D"]) == ("A B", "A B", "C D", None)
    assert split_direct(["I<|tool", "_call", ">call:x{}<channel|>B"]) == (
        "",
        "",
        "I",
        "<|tool_call>call:x{}<channel|>B",
    )


async def test_spelled_tag_then_orphan_close_per_id() -> None:
    """``I`` + ``<|tool_call>`` in plain pieces + ``call:x{}<tool_call|><channel|>B``, one id per chunk and cut where
    the final ``>`` starts the second chunk: the tag ends the window, so the answer is ``I`` and the later close stays
    in the tool text; with too short a carried tail the call is swallowed into an implicit thought."""
    tag = S.ids("<", "|", "tool", "_", "call", ">")
    lead = S.prose("I")
    seq = [
        *lead,
        *tag,
        *S.prose("call:x{}"),
        TOOL_CALL_CLOSE,
        CHANNEL_CLOSE,
        *S.prose("B"),
        TURN_CLOSE,
    ]
    gt = len(lead) + len(tag) - 1
    for chunks in (per_id(seq), [seq[:gt], seq[gt:]]):
        got, _ = await S.run_chunks(chunks)
        assert S.texts(got, "thought") == "", chunks
        assert S.texts(got, "answer") == "I", chunks
        assert [c.text for c in S.tool_commits(got)] == [
            "<|tool_call>call:x{}<tool_call|><channel|>B"
        ], chunks
