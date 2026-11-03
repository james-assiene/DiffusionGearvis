"""The tools-on chat-completions wire, as vLLM 0.29.0 serves it (T-0088; T-0087 arm (b), reading B).

This is the wire fake: it changes in the same PR as the client's id mode (CLAUDE.md). It builds an SSE body in
the **recorded** arm-(b) shape (``stream: true, skip_special_tokens: false, return_token_ids: true`` with
``tools`` and ``tool_choice: "none"``; ``docs/cd/toolcall-live-probe.md``, ``tests/fixtures/llm/t0087_streams.jsonl``)
from an id sequence and a partition of it into canvases:

- **role chunk:** top-level ``id, object, created, model, choices, prompt_token_ids, prompt_text``; one choice
  ``{index: 0, delta: {role: "assistant", content: ""}, logprobs: null, finish_reason: null}``. It has **no**
  ``token_ids`` key;
- **one chunk per canvas:** top-level ``id, object, created, model, choices[, system_fingerprint]``; one choice
  ``{index, delta, logprobs: null, finish_reason, [stop_reason,] token_ids}``. ``system_fingerprint`` is present
  iff the chunk's ``finish_reason`` is non-null, i.e. on the last chunk only (T-0095: recorded on a live
  multi-canvas stream, ``tests/fixtures/llm/t0095_row2_live.jsonl``; non-final chunks have no such key). ``delta`` holds the served
  ``content`` and ``reasoning`` the engine derives (in that key order), and a key is **absent** when its text in
  this chunk is empty;
- **last chunk:** ``finish_reason`` ``"stop"`` with ``stop_reason`` = the first stop id in the ids, or
  ``"length"`` with ``stop_reason: null`` when there is none;
- then ``data: [DONE]``.

**The served fields** follow ``split_served`` / T-0087's ``regions()`` and the probe's "rule that fits"
(48/48 rows): let D be the host decode of the ids up to, and excluding, the first id in ``STOP_IDS``.

- ``reasoning`` is the thought region: the text between ``<|channel>thought\\n`` and the next ``<channel|>`` (to
  the end of D while it is unclosed). No header: no reasoning.
- ``content`` is the answer region (the text after that ``<channel|>``, or all of D when there is no header), cut
  at its first ``SPECIAL_TOKENS`` literal.

Per canvas, each field carries the part of the whole-response field that the canvases so far have decoded (the
cumulative text after this canvas, minus the cumulative text before it). One recorded response is one canvas, so
for every recorded stream this is exactly the served text (``test_fakes_conform.py`` pins that key for key). The
multi-canvas shape — this text split, the fingerprint on the final chunk only, and no ``stop_reason`` on non-final
chunks — matches the one recorded multi-canvas tools-on stream (T-0088 gpu row 2 at the T-0093 launch, commit
``efadf45``: five content chunks), pinned on CPU by ``tests/contract/test_vllm_sse_live_shape.py``. Gpu row 2 compares
the live engine with this fake on every run; a difference means the fake changes.

**Knobs** (each is a way the wire can be wrong that the client must refuse, or a way it can be slow):

- ``drop_token_ids``: canvas indices whose chunk carries no ``token_ids`` key;
- ``truncate``: the last canvas chunk carries ``finish_reason: null`` and no ``stop_reason`` (a stream that ends
  without saying why), so **no** chunk carries ``system_fingerprint``: inferred, not recorded — a truncated stream
  never sends the finishing chunk, which is the only one the engine puts it on; ``done=False`` also drops ``[DONE]``;
- ``foreign_ids``: ids appended to a canvas's ``token_ids`` only (never decoded for the served fields), e.g. an id
  the vocabulary does not have;
- ``delay_s`` (transport): a pause before each canvas chunk;
- ``status`` (transport): a non-200 answer with an OpenAI-style error body;
- ``gate`` (transport): an ``asyncio.Event`` the transport waits for before it answers, so a request is in flight
  with no response yet (an abort lands there without closing a body that is being read).

**The engine's empty "retryable" response** (T-0171; ``engine_empty``). In vLLM v0.29.0, as read in the pinned image
``vllm/vllm-openai:v0.29.0``, a P0/P1 multimodal-cache drift miss is answered by the engine core with
``finish_reason=ERROR`` and no tokens (``vllm/v1/engine/core.py:1767-1771``; ``_handle_mm_cache_miss``, lines
1866-1898, which logs "Multi-modal cache miss ... P0/P1 cache drift; returning a retryable response"). The frontend
drops the hash from its own cache (``vllm/v1/engine/async_llm.py:786-794``), so the next identical request carries
the image data again. The chat streaming server, however, skips an output with zero new tokens *before* it reaches
its error check (``vllm/entrypoints/openai/chat_completion/serving.py:644-650``, ahead of line 752), so nothing about
the error reaches the wire. What the client sees is therefore: HTTP 200, the role chunk (sent on the first
iteration, before any output is looked at; the same chunk as above, ``prompt_token_ids`` included), then
``data: [DONE]``. No content chunk, no ``token_ids`` key, no ``finish_reason``, no ``{"error": ...}`` event.
``engine_empty(with_role=False)`` is the same response without the role chunk (only ``[DONE]``), which the client
must read the same way (T-0171 criterion 1). Inferred from the source and the T-0125 smoke log (one occurrence,
``logs/vllm.log:59326`` and session seq 545 of the b372edaa phase-2 run); not yet recorded on the wire: the GPU repro
is H-T171-1.

The transport serves ``/chat/completions`` only; any other path is a named assertion (``tests.md``: a double that
quietly answers a URL nobody meant to call is how a client change escapes its tests).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from local_astra.toolcall.special_tokens import SPECIAL_TOKENS, STOP_IDS

MODEL = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
#: what the recorded streams carry (``vllm-<version>-<build>``)
FINGERPRINT = "vllm-0.29.0-b591a092"
THOUGHT_HEADER = "<|channel>thought\n"
CHANNEL_CLOSE = "<channel|>"
DONE_LINE = "data: [DONE]"


class _Stream(Protocol):
    def feed(self, ids: Iterable[int]) -> str: ...

    def finish(self) -> str: ...


class Detok(Protocol):
    """``local_astra.toolcall.detok.HostDetokenizer`` (only ``stream`` is used)."""

    def stream(self) -> _Stream: ...


def compact(obj: Any) -> str:
    """The engine's serialisation (round-trips every recorded line byte for byte)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def until_stop(ids: Sequence[int]) -> tuple[list[int], int | None]:
    """The ids before the first stop id, and that stop id (``None`` when there is none)."""
    for i, t in enumerate(ids):
        if t in STOP_IDS:
            return list(ids[:i]), t
    return list(ids), None


def served_regions(text: str) -> tuple[tuple[int, int] | None, tuple[int, int]]:
    """``(thought span or None, content span)`` of the decode ``text`` as character offsets (rules above)."""
    h = text.find(THOUGHT_HEADER)
    if h < 0:
        thought = None
        answer_start = 0
    else:
        t0 = h + len(THOUGHT_HEADER)
        j = text.find(CHANNEL_CLOSE, t0)
        if j < 0:
            return (t0, len(text)), (len(text), len(text))
        thought = (t0, j)
        answer_start = j + len(CHANNEL_CLOSE)
    cut = len(text)
    for lit in SPECIAL_TOKENS:
        k = text.find(lit, answer_start)
        if 0 <= k < cut:
            cut = k
    return thought, (answer_start, cut)


def _clip(span: tuple[int, int] | None, lo: int, hi: int) -> tuple[int, int]:
    if span is None:
        return (0, 0)
    a, b = max(span[0], lo), min(span[1], hi)
    return (a, b) if a < b else (0, 0)


def partition(ids: Sequence[int], canvas: int) -> list[list[int]]:
    """``ids`` cut into consecutive canvases of ``canvas`` ids (the last one may be shorter)."""
    assert canvas >= 1
    return [list(ids[i : i + canvas]) for i in range(0, len(ids), canvas)] or [[]]


@dataclass
class StreamSpec:
    """Everything one fake response is built from."""

    chunks: Sequence[Sequence[int]]
    prompt_ids: Sequence[int] = (2, 105)
    request_id: str = "chatcmpl-fake"
    created: int = 1791761880
    drop_token_ids: frozenset[int] = frozenset()
    truncate: bool = False
    done: bool = True
    foreign_ids: Mapping[int, Sequence[int]] = field(default_factory=dict)


def _head(request_id: str, created: int) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL,
    }


def _role_chunk(head: Mapping[str, Any], prompt_ids: Sequence[int]) -> dict[str, Any]:
    """The first chunk of every response: the role, ``prompt_token_ids`` and no ``token_ids`` key."""
    return {
        **head,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
        "prompt_token_ids": list(prompt_ids),
        "prompt_text": None,
    }


def engine_empty(
    *,
    with_role: bool = True,
    prompt_ids: Sequence[int] = (2, 105),
    request_id: str = "chatcmpl-fake",
    created: int = 1791761880,
) -> list[str]:
    """The ``data:`` lines of the engine's empty "retryable" response (module docstring, T-0171): the role chunk
    (``with_role``) and ``[DONE]``, nothing else."""
    role = [_role_chunk(_head(request_id, created), prompt_ids)] if with_role else []
    return sse_lines(role, done=True)


def build_chunks(spec: StreamSpec, detok: Detok) -> list[dict[str, Any]]:
    """The parsed chunk dicts of one response, role chunk first (``[DONE]`` is not a chunk)."""
    all_ids = [t for c in spec.chunks for t in c]
    decodable, stop = until_stop(all_ids)
    # cumulative decoded length after each canvas: only the ids before the stop are decoded
    stream = detok.stream()
    remaining = len(decodable)
    ends: list[int] = []
    text_parts: list[str] = []
    for c in spec.chunks:
        take = min(len(c), remaining)
        remaining -= take
        text_parts.append(stream.feed(list(c[:take])))
        ends.append(sum(len(p) for p in text_parts))
    tail = stream.finish()
    text = "".join(text_parts) + tail
    ends[-1] += len(tail)
    thought, content = served_regions(text)

    head = _head(spec.request_id, spec.created)
    out: list[dict[str, Any]] = [_role_chunk(head, spec.prompt_ids)]
    lo = 0
    last = len(spec.chunks) - 1
    for k, c in enumerate(spec.chunks):
        hi = ends[k]
        delta: dict[str, str] = {}
        ca, cb = _clip(content, lo, hi)
        if cb > ca:
            delta["content"] = text[ca:cb]
        ra, rb = _clip(thought, lo, hi)
        if rb > ra:
            delta["reasoning"] = text[ra:rb]
        lo = hi
        choice: dict[str, Any] = {
            "index": 0,
            "delta": delta,
            "logprobs": None,
            "finish_reason": None,
        }
        if k == last and not spec.truncate:
            choice["finish_reason"] = "stop" if stop is not None else "length"
            choice["stop_reason"] = stop
        if k not in spec.drop_token_ids:
            choice["token_ids"] = [*c, *spec.foreign_ids.get(k, ())]
        chunk: dict[str, Any] = {**head, "choices": [choice]}
        if choice["finish_reason"] is not None:
            chunk["system_fingerprint"] = FINGERPRINT
        out.append(chunk)
    return out


def sse_lines(chunks: Sequence[Mapping[str, Any]], *, done: bool = True) -> list[str]:
    """The ``data:`` lines of a stream (what the fixture stores)."""
    lines = ["data: " + compact(c) for c in chunks]
    return [*lines, DONE_LINE] if done else lines


def sse_text(lines: Sequence[str]) -> str:
    return "".join(f"{ln}\n\n" for ln in lines)


def build_lines(spec: StreamSpec, detok: Detok) -> list[str]:
    return sse_lines(build_chunks(spec, detok), done=spec.done)


@dataclass
class Wire:
    """A ``MockTransport`` serving prepared streams, one per chat-completions request, and what it was sent."""

    streams: list[Sequence[str]]
    status: int = 200
    delay_s: float = 0.0
    bodies: list[dict[str, Any]] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    #: called with the chunk index before each chunk is sent (tests use it to abort mid-stream)
    on_chunk: Callable[[int], Any] | None = None
    #: T-0171: when set, every request is recorded and then held until the event is set (a request in flight with
    #: no response yet); ``arrived`` is set as each held request comes in
    gate: asyncio.Event | None = None
    arrived: asyncio.Event = field(default_factory=asyncio.Event)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        assert request.url.path.endswith("/chat/completions"), (
            f"the tools-on wire fake serves chat completions only; got {request.method} {request.url}"
        )
        self.bodies.append(json.loads(await request.aread()))
        if self.gate is not None:
            self.arrived.set()
            await self.gate.wait()
        if self.status != 200:
            return httpx.Response(
                self.status,
                json={
                    "object": "error",
                    "message": "fake failure",
                    "type": "InternalServerError",
                    "code": self.status,
                },
            )
        assert self.streams, "the wire fake has no stream left to serve"
        lines = list(self.streams.pop(0))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=self._body(lines)
        )

    async def _body(self, lines: list[str]) -> AsyncIterator[bytes]:
        for k, ln in enumerate(lines):
            if self.delay_s and ln != DONE_LINE:
                await asyncio.sleep(self.delay_s)
            if self.on_chunk is not None:
                res = self.on_chunk(k)
                if asyncio.iscoroutine(res):
                    await res
            yield f"{ln}\n\n".encode()
