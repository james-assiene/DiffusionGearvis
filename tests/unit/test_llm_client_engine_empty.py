"""T-0171 criteria 1-4: the client recognises the engine's empty "retryable" response, and nothing else.

The wire is ``tests/fakes/vllm_sse.py``: ``engine_empty`` for the drift miss (HTTP 200, the role chunk, ``[DONE]``;
its docstring cites the vLLM 0.29.0 source lines), ``build_lines`` for every other shape. The real ``VllmClient``
reads it over ``httpx.MockTransport`` in id mode (a tools-on prompt carrying one ``ImagePart``, the shape of the
request the T-0125 smoke lost), decoding with the T-0088a vocabulary subset.

The recognition rule under test (task file, "Implementation notes"): id mode, not aborted, ``[DONE]`` seen, no chunk
carried a ``finish_reason``, and **no chunk carried a ``token_ids`` key** -> ``LlmEngineEmpty``, a public subclass of
``LlmWireError`` whose message names the attempt id and says ``retryable``. Every other ending is what it is today.

Which rows were pending (activated at the start of ``/implement T-0171``): the two that need ``LlmEngineEmpty`` (criterion 1). The negative rows (criteria 2-4) hold on
the base already and are active, so they guard the implementation from the first commit; each was checked to be
discriminating against the mutant it names (the test-author's report).

Regime: CPU, httpx ``MockTransport``, no engine; n = 1 per row, deterministic.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from local_astra.conductor import llm_client
from local_astra.contracts.llm import ImagePart, Message, TextPart
from tests.fakes import vllm_sse as V
from tests.fakes.llm import ENGINE_EMPTY_IMAGE
from tests.unit import _llm_id_support as S

#: the attempt id of the generate the T-0125 smoke lost (session 20261027-112530, seq 14), as the engine saw it
ATTEMPT = "7c4a47eefb16-t1-gen-2"
#: what ``generate`` raises today for a stream that ends without a finish_reason (llm_client.py, T-0088 criterion 5)
TRUNCATED = f"{ATTEMPT}: the stream ended without a finish_reason"


def engine_empty_cls() -> type[BaseException]:
    """``llm_client.LlmEngineEmpty``, or a named assertion while it does not exist (the pending convention)."""
    cls = getattr(llm_client, "LlmEngineEmpty", None)
    assert cls is not None, (
        "T-0171 criterion 1: llm_client.LlmEngineEmpty (a public subclass of LlmWireError) does not exist yet"
    )
    assert isinstance(cls, type) and issubclass(cls, llm_client.LlmWireError), (
        f"T-0171 criterion 1: LlmEngineEmpty must subclass LlmWireError; its MRO is {getattr(cls, '__mro__', cls)}"
    )
    return cls


def is_engine_empty(exc: BaseException | None) -> bool:
    """True when ``exc`` is an ``LlmEngineEmpty``; False (not an error) while the class does not exist, so a negative
    row holds on the base and keeps holding once it does."""
    cls = getattr(llm_client, "LlmEngineEmpty", None)
    return exc is not None and isinstance(cls, type) and isinstance(exc, cls)


def image_prompt(*, tools: bool = True) -> Any:
    """The user turn carries a screen image; tools on (id mode) unless ``tools=False``."""
    user = Message(
        role="user",
        content=(
            ImagePart(url=ENGINE_EMPTY_IMAGE, phash="ab" * 8),
            TextPart(text="what is on this screen"),
        ),
    )
    messages = (S.SYSTEM, user)
    return S.tools_prompt(messages=messages) if tools else S.make_prompt(messages=messages)


async def serve(
    lines: list[str], *, tools: bool = True, request_id: str = ATTEMPT
) -> tuple[list[Any], BaseException | None, V.Wire]:
    """One generate over ``lines``: the commits it yielded, what it raised (``None``: nothing) and the wire."""
    wire = V.Wire(streams=[lines])
    cl = S.client(wire.transport())
    got: list[Any] = []
    exc: BaseException | None = None
    try:
        async for c in cl.generate(image_prompt(tools=tools), request_id):
            got.append(c)
    except Exception as e:  # the row inspects what was raised
        exc = e
    finally:
        await cl.aclose()
    assert len(wire.bodies) == 1, f"premise: one request was sent; got {len(wire.bodies)}"
    sent = json.dumps(wire.bodies[0])
    assert ENGINE_EMPTY_IMAGE in sent, "premise: the request carries the image"
    assert ("return_token_ids" in wire.bodies[0]) is tools, "premise: id mode iff tools on"
    return got, exc, wire


def ids_chunks() -> list[list[int]]:
    return [[*S.thought("t"), *S.prose("Hi.")], [*S.prose(" More.")]]


# =========================================================================== criterion 1


async def test_role_chunk_then_done_is_engine_empty() -> None:
    """Criterion 1: HTTP 200, the role chunk, ``[DONE]`` (the drift miss as the chat server streams it) raises
    ``LlmEngineEmpty`` with zero commits, and the message names the attempt and says ``retryable``."""
    lines = V.engine_empty()
    assert len(lines) == 2 and lines[-1] == V.DONE_LINE, "premise: the role chunk and [DONE]"
    role = json.loads(lines[0][len("data: ") :])["choices"][0]
    assert "token_ids" not in role and role["finish_reason"] is None, "premise: the role chunk"
    got, exc, _ = await serve(lines)
    cls = engine_empty_cls()
    assert got == [], f"criterion 1: zero commits; got {got}"
    assert isinstance(exc, cls), f"criterion 1: expected LlmEngineEmpty, got {exc!r}"
    assert isinstance(exc, llm_client.LlmWireError), "criterion 1: still an LlmWireError"
    assert ATTEMPT in str(exc), f"criterion 1: the message names the attempt id; got {str(exc)!r}"
    assert "retryable" in str(exc), f"criterion 1: the message says 'retryable'; got {str(exc)!r}"
    assert not cls.__name__.startswith("_") and cls.__module__ == llm_client.__name__, (
        "criterion 1: a public class of llm_client"
    )


async def test_bare_done_is_engine_empty() -> None:
    """Criterion 1, second half: a stream that is only ``data: [DONE]`` (no role chunk) raises the same."""
    lines = V.engine_empty(with_role=False)
    assert lines == [V.DONE_LINE], "premise: only [DONE]"
    got, exc, _ = await serve(lines)
    cls = engine_empty_cls()
    assert got == []
    assert isinstance(exc, cls), f"criterion 1: expected LlmEngineEmpty, got {exc!r}"
    assert ATTEMPT in str(exc) and "retryable" in str(exc), str(exc)


# =========================================================================== criterion 2 (negative)


@pytest.mark.parametrize(
    "variant",
    ["one-chunk", "two-chunks", "empty-token-ids", "ids-then-role-shaped-chunk"],
)
async def test_ids_then_done_without_finish_is_plain_wire_error(variant: str) -> None:
    """Criterion 2 (adversarial; the reviewers' mutation target): at least one chunk carried a ``token_ids`` key, then
    ``[DONE]``, and no ``finish_reason`` -> a plain ``LlmWireError`` (``type(exc) is LlmWireError``) with today's
    message. A recogniser that drops the "no ``token_ids`` chunk" condition reads every one of these as engine-empty.

    ``empty-token-ids``: one chunk whose ``token_ids`` is ``[]`` -- the key was carried, so by the rule it is not the
    drift signature (a recogniser keyed on "no ids decoded" instead of "no key seen" fails here).
    ``ids-then-role-shaped-chunk``: an ids chunk, then a chunk shaped like the role chunk (no ``token_ids`` key), so
    the *last* chunk alone looks empty."""
    if variant == "one-chunk":
        lines = S.stream_lines(ids_chunks()[:1], truncate=True)
    elif variant == "two-chunks":
        lines = S.stream_lines(ids_chunks(), truncate=True)
    elif variant == "empty-token-ids":
        lines = S.stream_lines([[]], truncate=True)
        chunk = json.loads(lines[1][len("data: ") :])["choices"][0]
        assert chunk["token_ids"] == [] and chunk["finish_reason"] is None, chunk
    else:
        lines = S.stream_lines(ids_chunks()[:1], truncate=True)
        lines = [*lines[:-1], lines[0], V.DONE_LINE]
    assert lines[-1] == V.DONE_LINE, "premise: [DONE] was sent"
    chunks = [json.loads(ln[len("data: ") :]) for ln in lines if ln != V.DONE_LINE]
    assert any("token_ids" in c["choices"][0] for c in chunks), "premise: a token_ids chunk"
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks), "premise: no finish"
    _, exc, _ = await serve(lines)
    assert type(exc) is llm_client.LlmWireError, (
        f"criterion 2: a truncation after ids is a plain LlmWireError, not {type(exc).__name__}: {exc!r}"
    )
    assert str(exc) == TRUNCATED, f"criterion 2: today's message; got {str(exc)!r}"
    assert not is_engine_empty(exc)


@pytest.mark.parametrize("shape", ["zero-chunks", "role-only", "ids-chunk"])
async def test_connection_end_without_done_is_plain_wire_error(shape: str) -> None:
    """Criterion 2: the connection ends with no ``[DONE]`` -- with zero chunks, with only the role chunk, or after an
    ids chunk -> a plain ``LlmWireError`` with today's message. "``[DONE]`` seen" is part of the signature: a server
    that died is not a cache miss."""
    if shape == "zero-chunks":
        lines: list[str] = []
    elif shape == "role-only":
        lines = V.engine_empty()[:-1]
    else:
        lines = S.stream_lines(ids_chunks()[:1], truncate=True, done=False)
    assert V.DONE_LINE not in lines, "premise: no [DONE]"
    _, exc, _ = await serve(lines)
    assert type(exc) is llm_client.LlmWireError, (
        f"criterion 2: a stream that just ends is a plain LlmWireError, not {type(exc).__name__}: {exc!r}"
    )
    assert str(exc) == TRUNCATED, f"criterion 2: today's message; got {str(exc)!r}"


async def test_null_token_ids_is_plain_wire_error() -> None:
    """Criterion 2 / the recognition rule (review A-F2): a chunk carrying the ``token_ids`` *key* counts as ids seen,
    whatever its value. The role chunk with ``"token_ids": null`` added, then ``[DONE]`` -> a plain ``LlmWireError``
    with today's message, never ``LlmEngineEmpty`` (a recogniser keyed on ``token_ids is not None`` fails here)."""
    role = json.loads(V.engine_empty()[0][len("data: ") :])
    role["choices"][0]["token_ids"] = None
    lines = ["data: " + V.compact(role), V.DONE_LINE]
    _, exc, _ = await serve(lines)
    assert type(exc) is llm_client.LlmWireError, (
        f"a null token_ids key is ids seen: plain LlmWireError, not {type(exc).__name__}: {exc!r}"
    )
    assert str(exc) == TRUNCATED, f"today's message; got {str(exc)!r}"
    assert not is_engine_empty(exc)


# =========================================================================== criterion 3 (negative)

#: an OpenAI-style in-stream error event, as vLLM's streaming error response carries one before ``[DONE]``
ERROR_EVENT = "data: " + V.compact(
    {
        "error": {
            "object": "error",
            "message": "EngineCore encountered an issue",
            "type": "InternalServerError",
            "param": None,
            "code": 500,
        }
    }
)


@pytest.mark.parametrize("with_role", [True, False])
async def test_error_event_is_not_engine_empty(with_role: bool) -> None:
    """Criterion 3: a ``{"error": {...}}`` payload before ``[DONE]`` is an error the engine *said*, not the silent
    drift signature -- it must not raise ``LlmEngineEmpty`` (the error event, like the role chunk, has no
    ``token_ids`` key, so the bare rule of the implementation notes would misread it). It still raises: nothing
    about it is a response."""
    role = V.engine_empty()[:1] if with_role else []
    lines = [*role, ERROR_EVENT, V.DONE_LINE]
    got, exc, _ = await serve(lines)
    assert got == []
    assert exc is not None and isinstance(exc, llm_client.LlmWireError), (
        f"criterion 3: still an error the client raises; got {exc!r}"
    )
    assert not is_engine_empty(exc), (
        f"criterion 3: an in-stream error event is not the empty retryable response; got {exc!r}"
    )


# =========================================================================== the abort exemption


@pytest.mark.parametrize("when", ["before-the-request", "in-flight"])
async def test_aborted_empty_stream_raises_nothing(when: str) -> None:
    """ "Not aborted" is part of the rule: an abort (barge-in) of a request whose answer is the empty response raises
    nothing and yields nothing -- the abort is the end the consumer asked for. ``before-the-request``: the abort
    lands between ``generate(...)`` and the first ``__anext__`` (T-0005 M5); ``in-flight``: after the request is sent
    and before the response opens (the wire holds it on ``gate``)."""
    gate = asyncio.Event() if when == "in-flight" else None
    wire = V.Wire(streams=[V.engine_empty()], gate=gate)
    cl = S.client(wire.transport())
    got: list[Any] = []

    async def consume() -> None:
        async for c in cl.generate(image_prompt(), ATTEMPT):
            got.append(c)

    try:
        if gate is None:
            await cl.abort(ATTEMPT)
            await consume()
            assert wire.bodies == [], "premise: an abort before the first __anext__ sends nothing"
        else:
            task = asyncio.create_task(consume())
            await asyncio.wait_for(wire.arrived.wait(), 5)
            await cl.abort(ATTEMPT)
            gate.set()
            await asyncio.wait_for(task, 5)  # raises here if the client raised
            assert len(wire.bodies) == 1, "premise: the request was in flight"
    finally:
        await cl.aclose()
    assert got == []


# =========================================================================== criterion 4 (negative)


@pytest.mark.parametrize("with_role", [True, False])
async def test_tools_off_empty_stream_unchanged(with_role: bool) -> None:
    """Criterion 4: tools off the stream is read by its served text, and an empty one ends quietly with zero commits,
    exactly as today -- never ``LlmEngineEmpty`` (tools-off recognition is a follow-up, not this task)."""
    got, exc, wire = await serve(V.engine_empty(with_role=with_role), tools=False)
    assert "tools" not in wire.bodies[0], "premise: tools off"
    assert exc is None, f"criterion 4: a tools-off empty stream raises nothing today; got {exc!r}"
    assert got == []


async def test_tools_off_plain_stream_still_reads() -> None:
    """Criterion 4, the positive half on the same rig: a tools-off text stream reads as on main."""
    sse = 'data: {"choices":[{"delta":{"content":"Four."}}]}\n\ndata: [DONE]\n\n'
    cl = S.client(
        httpx.MockTransport(
            lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)
        )
    )
    try:
        got = await S.collect(cl, image_prompt(tools=False))
    finally:
        await cl.aclose()
    assert [(c.segment, c.text) for c in got] == [("answer", "Four.")]
