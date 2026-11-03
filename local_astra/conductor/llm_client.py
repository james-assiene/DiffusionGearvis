"""vLLM chat-completions SSE client for DiffusionGemma (docs/02 §2.4, docs/10 §9). Task T-0005.

What the tests pin:

- request body carries only whitelisted sampling fields (``LlmOptions``) and plugin args under
  ``vllm_xargs``; ``response_format``/``temperature``/``seed`` are never sent (400 on diffusion);
- SSE deltas are grouped into *commits* per canvas: the ``gemma4`` reasoning parser's
  ``reasoning`` delta → ``thought``, ``content`` delta → ``answer``; a configurable UI fence
  (``<a2ui>…</a2ui>`` in the answer channel) → ``ui``;
- ``thought_open`` is set when the canvas ends inside the reasoning channel;
- ``abort`` closes the stream and is idempotent; a second call is a no-op;
- ``prefill`` sends ``max_tokens=1``-style warm request with the prefix only and keeps the
  request id so it can be aborted at end-of-turn (Q3: prefix cache survives).

T-0088 (tools on, ``Prompt.tools`` set): the request is T-0087's arm (b) and the stream is read by its
``token_ids`` (*id mode*): decoded on the host (``toolcall/detok.py``) up to the first stop id, split into
``thought``/``answer``/``ui`` by the same fence and M9 logic, and everything from the first call tag to the stop
comes back as one ``tool`` commit at the end, with the stop id. Served ``content``/``reasoning`` are only checked
for presence. With tools off, every body and every commit is ``main``'s.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

import httpx

from local_astra.contracts.cd import CdRequest
from local_astra.contracts.llm import (
    VLLM_XARGS_KEY,
    Commit,
    FenceClose,
    ImagePart,
    LlmOptions,
    Message,
    Part,
    PrefillHandle,
    Prompt,
    Segment,
    TextPart,
)
from local_astra.toolcall import router
from local_astra.toolcall.detok import HostDetokenizer, UnknownTokenId
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS, STOP_IDS, drop_special_literals

#: fence around the A2UI channel inside the answer text (docs/12 §1.1, docs/02 §2.2)
UI_OPEN, UI_CLOSE = "<a2ui>", "</a2ui>"

#: Delta keys that may carry the reasoning channel. **Both** are accepted on purpose (T-0005
#: amendment B1): nobody has captured a real streaming delta from ``--reasoning-parser gemma4``
#: on this box, so picking one would be a coin flip. When a real chunk is captured, record the
#: key in ``.claude/rules/gpu-services.md`` and narrow this tuple in the same change.
REASONING_DELTA_KEYS: tuple[str, ...] = ("reasoning", "reasoning_content")

#: Literal thinking-channel markers. If one of these shows up in the *content* channel the
#: reasoning parser is not doing its job (wrong ``--reasoning-parser``, a template that emits the
#: token as text, a third delta spelling we do not know about), and the thought would otherwise
#: go straight to TTS. CLAUDE.md's rule is absolute — never voice thinking — so content carrying
#: any of these is treated as thought wholesale (M9). The Gemma spellings are the ones this stack
#: uses; the bare pair is included because it costs one substring test and the failure direction
#: matters: misrouting an answer to ``thought`` loses a sentence, misrouting a thought to
#: ``answer`` speaks it aloud.
THOUGHT_MARKERS: tuple[str, ...] = ("<|think|>", "<|/think|>", "<think>", "</think>")

#: OpenAI stream sentinel.
SSE_DONE = "[DONE]"

CHAT_COMPLETIONS_PATH = "/chat/completions"

#: How many pre-start aborts are remembered (M5). One per turn is the realistic rate; the
#: bound only exists so an abort for an id that never starts cannot grow without limit.
MAX_EARLY_ABORTS = 64

#: How many *further* canvases one open fence may carry over before the client gives up on it
#: (T-0052 D3). The measured surface of 2026-09-16 was 1159 characters over two canvases and
#: T-0044 ruling A7's minimal surface 1060, so the opening canvas plus four more — ~3400
#: characters at that turn's ~846 characters per canvas — is about three times the observed need.
#: A bigger bound lets a fence the model never closed eat the rest of the answer (amendment M7's
#: failure, and the reason the carry is bounded at all); a smaller one cuts a legitimate dashboard.
UI_FENCE_MAX_CANVASES = 4

#: …and the same bound in characters, because one canvas is not a fixed number of them: 8192 is
#: ~7× the measured 1159-character surface and comfortably more than ``UI_FENCE_MAX_CANVASES``
#: canvases of ordinary size, so it only fires on a runaway. Both are ambiguity Q2 in the task:
#: raising either needs the ``canvases_spanned`` distribution from criterion 16 and a task entry.
UI_FENCE_MAX_CHARS = 8192

TOKENIZE_PATH = "/tokenize"


class LlmWireError(RuntimeError):
    """A tools-on stream the client cannot read by its ids (T-0088 criterion 5): a chunk that should carry
    ``token_ids`` and does not, an id outside the vocabulary, or a stream that ends without a ``finish_reason``.
    Never degraded to reading ``content``."""


class LlmEngineEmpty(LlmWireError):
    """The engine's empty "retryable" response (T-0171), read in id mode only: HTTP 200, at most the role chunk, then
    ``data: [DONE]``, with no chunk carrying a ``token_ids`` key, no ``finish_reason`` and no ``{"error": ...}`` event.

    The drift miss is why the retry exists: vLLM v0.29.0 answers a P0/P1 multimodal-cache drift miss with
    ``finish_reason=ERROR`` and no tokens (``vllm/v1/engine/core.py:1767-1771``, ``_handle_mm_cache_miss`` at
    1866-1898) and drops the hash from the frontend cache (``vllm/v1/engine/async_llm.py:786-794``), so the identical
    request, resent, carries the image again. The chat streaming server skips a zero-token output before its error
    check (``vllm/entrypoints/openai/chat_completion/serving.py:644-650``, ahead of line 752), so the error never
    reaches the wire.

    It is not the only trigger. The same wire signature comes from **any** zero-token engine finish in v0.29.0: the
    drift-miss ERROR above; a preprocessing ERROR (``_send_error_outputs_to_client``, core.py ~1917-1924 and
    ~1988-1991); and an engine-side ABORT (``_send_abort_outputs``, core.py ~1983-1986). The client cannot tell them
    apart, so each gets the one resend. The session resends once; every other truncated stream stays a plain
    ``LlmWireError``."""


@dataclass(frozen=True)
class VllmEndpoint:
    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
    request_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not (
            self.base_url.startswith("http://127.0.0.1")
            or self.base_url.startswith("http://localhost")
        ):
            raise ValueError("services bind to loopback only (docs/02 §9)")


@dataclass
class _Active:
    """One in-flight request: the live response (once open) and whether abort was asked for."""

    response: httpx.Response | None = None
    aborted: bool = False

    def is_aborted(self) -> bool:
        """Read through a method on purpose: ``abort()`` runs in another task, so the flag must
        be re-read at every check point and never narrowed to a constant."""
        return self.aborted


def _part_body(part: Part) -> dict[str, Any]:
    """One OpenAI content part. ``phash`` is ours and never goes on the wire."""
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        if not part.url.startswith("data:"):
            # Every image must arrive as inline pixels. ``prompt_builder`` no longer produces a
            # reference URL (a history frame it cannot resolve degrades to its caption line,
            # amendment B9), so reaching this line means some other producer is handing us a
            # reference vLLM would have to fetch. Fail here, loudly, rather than on the server:
            # a silently unresolved image is a model that has gone blind mid-turn.
            raise ValueError(f"image part is not an inline data URL: {part.url[:32]!r}")
        return {"type": "image_url", "image_url": {"url": part.url}}
    # VideoPart: clips (S7) land with T-0009, which measures the wire format vLLM wants.
    raise NotImplementedError("video parts are not sent yet (T-0009)")


def _message_body(message: Message) -> dict[str, Any]:
    return {"role": message.role, "content": [_part_body(p) for p in message.content]}


def _sse_payload(line: str) -> str | None:
    """The payload of one SSE line, or None for comments, blanks and non-data fields."""
    if not line or line.startswith(":") or not line.startswith("data:"):
        return None
    return line[len("data:") :].strip()


def _delta_fields(payload: object) -> tuple[str | None, str | None]:
    """``(reasoning, content)`` of the first choice's delta; None for absent or non-string."""
    if not isinstance(payload, dict):
        return None, None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, None
    first = choices[0]
    delta = first.get("delta") if isinstance(first, dict) else None
    if not isinstance(delta, dict):
        return None, None
    reasoning: str | None = None
    for key in REASONING_DELTA_KEYS:
        value = delta.get(key)
        if isinstance(value, str) and value:
            reasoning = value
            break
    content = delta.get("content")
    return reasoning, content if isinstance(content, str) else None


class VllmClient:
    """``transport`` is an ``httpx.AsyncBaseTransport`` (tests inject ``httpx.MockTransport``)."""

    def __init__(
        self,
        endpoint: VllmEndpoint | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        detokenizer: HostDetokenizer | None = None,
        visual_tokens: int | None = None,
    ) -> None:
        self.endpoint = endpoint or VllmEndpoint()
        self.transport = transport
        #: T-0131 A2: the per-request visual-token budget (``mm_processor_kwargs.max_soft_tokens``) on every body
        #: ``request_body`` builds, so a prefill and its generate always agree (the multimodal hash includes the
        #: processor kwargs). ``None`` sends nothing: the engine default (280 for this checkpoint).
        self.visual_tokens = visual_tokens
        #: the host decoder id mode reads with (T-0088); loaded by the caller, never by the client, so no file is
        #: read inside a turn. Tools-off needs none.
        self.detokenizer = detokenizer
        self._client: httpx.AsyncClient | None = None
        self._active: dict[str, _Active] = {}
        #: ids aborted before their request was registered (M5). ``generate`` is an async
        #: generator, so its body — and ``_begin`` with it — first runs on the first
        #: ``__anext__``; an ``abort`` in that window used to find nothing and return silently,
        #: and the turn was then spoken anyway. That window is exactly when barge-in happens, so
        #: the abort leaves a tombstone the request picks up when it starts. Bounded: an abort
        #: for an id that is never issued must not leak.
        self._aborted_early: deque[str] = deque(maxlen=MAX_EARLY_ABORTS)

    # -- body -------------------------------------------------------------------------------

    def request_body(
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        xargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Pure: the chat-completions body. Only whitelisted sampling fields; plugin args and the
        request id under ``vllm_xargs``; thinking via ``chat_template_kwargs``; always streaming.

        A fresh, unshared dict every call — the caller may mutate what it gets back.
        """
        merged: dict[str, Any] = dict(xargs or {})
        merged["astra_request_id"] = request_id
        messages = [_message_body(m) for m in prompt.messages]
        if prompt.continue_final_message:
            # q9: the continued assistant message goes out as a plain string (its one TextPart)
            last = prompt.messages[-1]
            messages[-1] = {"role": last.role, "content": last.text()}
        body: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            "stream": True,
            "chat_template_kwargs": dict(prompt.chat_template_kwargs()),
            VLLM_XARGS_KEY: LlmOptions.extra_body(merged)[VLLM_XARGS_KEY],
        }
        if options is not None:
            body.update(options.to_request_fields())
        if prompt.tools is not None:
            # T-0087 arm (b), exactly: the native tools, no engine-side parsing, and the ids on every chunk.
            # Output-format fields, not sampling ones; they follow from ``Prompt.tools`` alone.
            body["tools"] = copy.deepcopy(list(prompt.tools))
            body["tool_choice"] = "none"
            body["skip_special_tokens"] = False
            body["return_token_ids"] = True
        if prompt.continue_final_message:
            body["continue_final_message"] = True
            body["add_generation_prompt"] = False
        if self.visual_tokens is not None:
            # T-0131 A2: a top-level key, never under ``vllm_xargs`` or ``LlmOptions``; absent when unset.
            body["mm_processor_kwargs"] = {"max_soft_tokens": self.visual_tokens}
        return body

    # -- streaming --------------------------------------------------------------------------

    async def prefill(self, prefix: Prompt, request_id: str) -> PrefillHandle:
        """Warm the prefix cache (docs/01 D9). One streaming request, ``max_tokens=1``, kept
        addressable by ``request_id`` so ``abort`` can free the sequence at end-of-turn.

        ``prefix_tokens`` comes from ``usage.prompt_tokens`` when the server reports it;
        **0 means "not reported"**, never "an empty prefix" (amendments B5/B12: it is never
        counted with a local tokenizer — a second tokenizer is a second source of truth, and
        ``PrefillHandle.prefix_tokens`` is a non-optional ``int``). vLLM only reports usage on a
        stream when asked via ``stream_options``, which the accepted key set does not allow, so 0
        is the normal case today; nothing reads the field yet. Whether the diffusion path accepts
        ``stream_options`` is a GPU question queued for the next session.
        """
        body = self.request_body(prefix, request_id, LlmOptions(max_tokens=1))
        record = self._begin(request_id)
        prefix_tokens = 0
        try:
            if record.is_aborted():
                return PrefillHandle(request_id=request_id, prefix_tokens=0)
            async with self._http().stream("POST", CHAT_COMPLETIONS_PATH, json=body) as response:
                record.response = response
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if record.is_aborted():
                        break
                    payload = _sse_payload(line)
                    if payload is None or payload == SSE_DONE:
                        if payload == SSE_DONE:
                            break
                        continue
                    reported = _usage_prompt_tokens(json.loads(payload))
                    if reported is not None:
                        prefix_tokens = reported
        except (httpx.HTTPError, httpx.StreamError):
            # Aborting the anticipatory prefill at end-of-turn is the design (docs/01 D9), not an
            # error: ``abort`` closes the response under this reader. Same rule as ``generate`` —
            # a failure that is *not* an abort still propagates.
            if not record.is_aborted():
                raise
        finally:
            self._end(request_id)
        return PrefillHandle(request_id=request_id, prefix_tokens=prefix_tokens)

    async def generate(
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        xargs: Mapping[str, Any] | None = None,
        *,
        cd: CdRequest | None = None,
    ) -> AsyncIterator[Commit]:
        """Stream commits: one per segment, ``canvas_idx`` per committed canvas (= one SSE chunk,
        amendment B8). ``thought_open`` is set on every commit of a canvas that ends inside the
        reasoning channel; such text never reaches TTS (docs/11 §2.1).

        The UI fence carries across the canvases of **this** request and no further: the state is a
        local of this frame (T-0052 D1/criterion 3), so a retry after a 400/413 is a new call and
        starts clean. Its endings are ``FenceClose`` values on the commit.

        With ``prompt.tools`` the stream is read by its ids (T-0088 id mode, :class:`_IdReader`; the reader is a
        local of this frame too) and a ``tool`` commit, if any, is the last one yielded. ``cd`` joins ``xargs``
        under ``vllm_xargs`` with its JSON types; a key given by both is a ``ValueError`` before any request.
        """
        reader = self._id_reader(prompt)
        body = self.request_body(prompt, request_id, options, _with_cd(xargs, cd))
        record = self._begin(request_id)
        fence = FenceState()
        #: characters of the current carry already published, so the end-of-response flush adds a
        #: commit only when it has something the consumer has not seen (criterion 7 / A12).
        published = 0
        canvas_idx = 0
        #: T-0171: the stream said ``[DONE]``, and (id mode) carried an ``{"error": ...}`` event
        done = False
        error_event = False
        try:
            if record.is_aborted():
                # Aborted between ``generate(...)`` and the first ``__anext__`` (M5): no request
                # is sent and no commit is yielded, which is what barge-in asked for.
                return
            try:
                async with self._http().stream(
                    "POST", CHAT_COMPLETIONS_PATH, json=body
                ) as response:
                    record.response = response
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if record.is_aborted():
                            break
                        payload = _sse_payload(line)
                        if payload is None:
                            continue
                        if payload == SSE_DONE:
                            done = True
                            break
                        if reader is None:
                            reasoning, content = _delta_fields(json.loads(payload))
                            pieces, fence = split_canvas(reasoning, content, fence)
                            if not pieces:
                                continue
                            # the canvas ends inside the thought when its last piece is thought
                            thought_open = pieces[-1].segment == "thought"
                        else:
                            chunk = json.loads(payload)
                            if isinstance(chunk, dict) and "error" in chunk:
                                error_event = True
                            pieces, thought_open = reader.feed(chunk, canvas_idx)
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
                            yield Commit(
                                request_id=request_id,
                                canvas_idx=canvas_idx,
                                segment=piece.segment,
                                text=piece.text,
                                thought_open=thought_open,
                                fence_close=piece.fence,
                                fence_continued=piece.continued,
                                implicit_thought=piece.implicit_thought,
                            )
                            emitted = True
                            if record.is_aborted():
                                break
                        if record.is_aborted():
                            break
                        if emitted:
                            canvas_idx += 1
            except (httpx.HTTPError, httpx.StreamError):
                # abort() closes the response under a reader; that is the intended end of the
                # stream, not a failure. Anything else propagates.
                if not record.is_aborted():
                    raise
            if reader is not None and not record.is_aborted() and not reader.finished:
                if done and not reader.ids_seen and not error_event:
                    # T-0171: the engine's empty "retryable" response (see ``LlmEngineEmpty``)
                    raise LlmEngineEmpty(
                        f"{request_id}: the engine returned an empty retryable response "
                        "(no token_ids and no finish_reason before [DONE])"
                    )
                # criterion 5: a stream that ends without saying why is not a response; no tool commit
                raise LlmWireError(f"{request_id}: the stream ended without a finish_reason")
            # id mode publishes ui text with its special literals removed (ruling E8)
            shown = fence.body if reader is None else drop_special_literals(fence.body)
            if fence.open and len(shown) > published:
                # The response ended — [DONE], a stream that stopped, or an abort — with the fence
                # still open. Publish what there is to the *shown* channel, marked unclosed, and
                # let the gate rule on it (D2). Re-reading it as prose is the 2026-09-16 bug;
                # dropping it loses a surface the gate could salvage and records nothing.
                yield Commit(
                    request_id=request_id,
                    canvas_idx=canvas_idx,
                    segment="ui",
                    text=shown,
                    fence_close="open",
                    fence_continued=fence.canvases > 1,
                )
            if reader is not None and not record.is_aborted():
                tool = reader.tool_text()
                if tool is not None:
                    # criterion 7: at most one, last, after the final chunk; the speech gate refuses it
                    yield Commit(
                        request_id=request_id,
                        canvas_idx=reader.last_canvas,
                        segment="tool",
                        text=tool,
                        stop_id=reader.stop_id,
                    )
        finally:
            self._end(request_id)

    async def tokenize(self, text: str) -> list[int]:
        """The engine's ids for ``text`` (T-0088 criterion 4; H9): ``/tokenize`` at the server root (the base URL
        without ``/v1``; T-0087 q10), ``add_special_tokens: false``. Never tokenised locally. Raises on a non-200
        status and on a body whose ``tokens`` is not a list of ints."""
        body = {"model": self.endpoint.model, "prompt": text, "add_special_tokens": False}
        response = await self._http().post(self._root_url(TOKENIZE_PATH), json=body)
        response.raise_for_status()
        payload = response.json()
        tokens = payload.get("tokens") if isinstance(payload, dict) else None
        if not isinstance(tokens, list) or not all(type(t) is int for t in tokens):
            raise LlmWireError(f"/tokenize answered without a list of int tokens: {payload!r:.200}")
        return list(tokens)

    async def abort(self, request_id: str) -> None:
        """Idempotent. Closing the SSE stream *is* the abort: vLLM frees the sequence when the
        client disconnects (contract test: ``num_requests_running`` drops within 200 ms)."""
        record = self._active.get(request_id)
        if record is None:
            # Either the request has not started yet — ``generate`` is an async generator, so its
            # body runs on the first ``__anext__``, and barge-in lands in exactly that window —
            # or it has already finished. Leaving a tombstone covers the first case; in the
            # second it is a bounded, harmless entry that no request will ever claim (M5).
            self._aborted_early.append(request_id)
            return
        record.aborted = True
        if record.response is not None:
            await record.response.aclose()

    async def aclose(self) -> None:
        """Close the shared connection pool (the Conductor owns the client's lifetime)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- internals --------------------------------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.endpoint.base_url,
                transport=self.transport,
                timeout=self.endpoint.request_timeout_s,
            )
        return self._client

    def _root_url(self, path: str) -> str:
        """``path`` at the server root: the base URL with a trailing ``/v1`` removed (loopback, as validated)."""
        url = httpx.URL(self.endpoint.base_url)
        root = url.path.rstrip("/").removesuffix("/v1")
        return str(url.copy_with(path=root + path))

    def _id_reader(self, prompt: Prompt) -> _IdReader | None:
        """The id-mode reader for a tools-on prompt; ``None`` for tools off. Refuses before any request."""
        if prompt.tools is None:
            return None
        if self.detokenizer is None:
            raise ValueError(
                "a tools-on generate needs VllmClient(detokenizer=...) (T-0088 id mode)"
            )
        tag = router.TAGS["native"]
        # criterion 7 / E4: a continuation whose assistant text ends in the call tag starts in the tool state
        continues_call = prompt.continue_final_message and prompt.messages[-1].text().endswith(tag)
        return _IdReader(self.detokenizer, tag if continues_call else None)

    def _begin(self, request_id: str) -> _Active:
        record = _Active(aborted=request_id in self._aborted_early)
        while request_id in self._aborted_early:  # consume the tombstone (M5)
            self._aborted_early.remove(request_id)
        self._active[request_id] = record
        return record

    def _end(self, request_id: str) -> None:
        self._active.pop(request_id, None)


def _with_cd(xargs: Mapping[str, Any] | None, cd: CdRequest | None) -> Mapping[str, Any] | None:
    """``xargs`` plus ``cd.to_xargs()`` (JSON types kept); a key given by both is refused (T-0088 criterion 3)."""
    if cd is None:
        return xargs
    typed = cd.to_xargs()
    clash = sorted(set(xargs or {}) & set(typed))
    if clash:
        raise ValueError(f"vllm_xargs keys given both in xargs and by cd: {clash}")
    return {**(xargs or {}), **typed}


def _usage_prompt_tokens(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    tokens = usage.get("prompt_tokens")
    return tokens if isinstance(tokens, int) else None


# -- the UI fence, across the canvases of one request (T-0052) ------------------------------------
#
# T-0005 amendments B7/M7 said a fence could not straddle two canvases and the state was thrown
# away between chunks. On 2026-09-16 the model wrote a flight-booking surface that did not fit in
# one canvas; the 750-character tail was classified ``answer`` and Pocket TTS read A2UI JSON aloud
# for 807 characters. B7/M7 are withdrawn (T-0052 D1) — but threading the flag *alone* would have
# made that same turn silent, because the model never wrote ``</a2ui>`` and the carry would have
# swallowed the 152 characters of prose that followed the JSON (ruling A1). So the carry comes with
# the three endings of D3: the written marker, the structural end of the JSON, and a bound.


@dataclass(frozen=True)
class FenceState:
    """The UI fence as carried through one ``generate`` frame. Never stored on the client (D1).

    ``body`` is the whole fenced text so far — what a ``ui`` commit hands the gate, because a slice
    that begins in the middle of a JSON object is not a document and the gate is right to reject it
    (ruling A10). ``held`` is a partial fence marker a canvas ended inside: with a carry it must be
    kept for the next canvas rather than dropped, or the client either voices ``ui>`` or loses the
    opening of a real surface.
    """

    open: bool = False
    body: str = ""
    #: canvases this carry has survived the end of; the opening canvas counts as one
    canvases: int = 0
    held: str = ""


@dataclass(frozen=True)
class Piece:
    """One segment of one canvas. ``fence``/``continued`` are set on ``ui`` pieces only; ``implicit_thought`` on
    ``thought`` pieces of an implicit thought only (T-0170, id mode)."""

    segment: Segment
    text: str
    fence: FenceClose | None = None
    continued: bool = False
    implicit_thought: bool = False


def split_canvas(
    reasoning_delta: str | None,
    content_delta: str | None,
    state: FenceState,
    *,
    hold: bool = True,
) -> tuple[list[Piece], FenceState]:
    """Pure: one SSE delta pair plus the carried fence in, pieces plus the new fence out.

    The reasoning channel is always ``thought`` — a ``<a2ui>`` written while still thinking is part
    of the thought and must never be voiced or rendered, and it must not open the *content*
    channel's fence. Content carrying a literal thinking marker is thought in full (M9).

    In the content channel the fence ends in exactly one of four ways, and the ending travels on
    the piece so a reader can tell a surface the model finished from one whose end the client
    guessed (criterion 12):

    ``written``     the model wrote ``</a2ui>``;
    ``inferred``    the JSON ended structurally (bracket depth back to zero outside a string, the
                    trailing run of closers consumed, the remainder not resuming with ``{``);
    ``abandoned``   ``UI_FENCE_MAX_CANVASES``/``UI_FENCE_MAX_CHARS`` reached, or a second
                    ``<a2ui>`` arrived while the carry was open (D3);
    ``open``        the canvas ended inside the fence. The text is still published to the *shown*
                    channel — never re-read as prose (D2).

    ``hold=False`` (id mode, T-0088): no partial fence marker is held here. The id reader holds every partial
    marker itself before it calls this, and gives a fragment that never became one back as text (ruling E7).
    """
    pieces: list[Piece] = []
    if reasoning_delta:
        pieces.append(Piece("thought", reasoning_delta))
    text = content_delta or ""
    if text and any(marker in text for marker in THOUGHT_MARKERS):
        pieces.append(Piece("thought", text))
        return pieces, state
    if not text:
        # nothing in the content channel: the fence neither grows nor spends its budget
        return pieces, state

    open_ui, body, canvases, held = state.open, state.body, state.canvases, state.held
    rest = text
    if held:
        merged = held + text
        # Inside the fence the fragment is just text, so it is always kept. Outside it, a fragment
        # that never became a marker is dropped exactly as ``_strip_markup`` dropped it before the
        # carry existed (L10): the alternative is reading a half-open tag aloud.
        rest = merged if open_ui or _resumes_marker(merged) else text
        held = ""

    while rest:
        if not open_ui:
            head, fence, rest = rest.partition(UI_OPEN)
            if not fence and hold:
                head, held = _hold_partial(head)
            head = _strip_markup(head)
            if head:
                pieces.append(Piece("answer", head))
            if fence:
                open_ui, body, canvases = True, "", 0
            continue
        cut, ending = _fence_cut(body, rest)
        if ending is None:
            addition, held = _hold_partial(rest) if hold else (rest, "")
            body, rest = body + addition, ""
            continue
        body += rest[:cut]
        consumed = {"written": len(UI_CLOSE), "abandoned": len(UI_OPEN), "inferred": 0}[ending]
        rest = rest[cut + consumed :]
        pieces.append(Piece("ui", body, ending, canvases > 0))
        # a second ``<a2ui>`` ends the carry as abandoned and opens a new fence (D3)
        open_ui, body, canvases = ending == "abandoned", "", 0

    if open_ui:
        canvases += 1
        spent = canvases > UI_FENCE_MAX_CANVASES or len(body) > UI_FENCE_MAX_CHARS
        pieces.append(Piece("ui", body, "abandoned" if spent else "open", canvases > 1))
        if spent:
            # ".claude/rules/conductor.md": a subsystem that turns itself off publishes that it
            # did. The answer channel is live again from the next canvas.
            open_ui, body, canvases = False, "", 0
    return pieces, FenceState(open=open_ui, body=body, canvases=canvases, held=held)


def split_segments(
    reasoning_delta: str | None, content_delta: str | None, in_ui: bool
) -> list[tuple[Segment, str, bool]]:
    """Pure helper: (segment, text, in_ui_after) pieces from one SSE delta pair. T-0005.

    A thin adapter over :func:`split_canvas` so there is exactly one implementation of the split
    (two would drift). It starts from an *empty* carry, which is all a caller holding a bool can
    say: the fence markers still belong to the ``ui`` piece they delimit, and the flag still reads
    "is the UI fence still open after this piece". What a bool cannot express is the carried text,
    so ``generate`` threads :class:`FenceState` instead.
    """
    pieces, _ = split_canvas(reasoning_delta, content_delta, FenceState(open=in_ui))
    return [
        (piece.segment, piece.text, in_ui if piece.segment == "thought" else piece.fence == "open")
        for piece in pieces
    ]


def _fence_cut(body: str, rest: str) -> tuple[int, FenceClose | None]:
    """Where the fence ends inside ``rest`` (already carrying ``body``), and how — or ``None``.

    The earliest ending wins; at the same index a close the model *wrote* beats an inferred one,
    and both beat a reopening fence. A written close also wins when the structural end precedes it
    by nothing but whitespace: the model did finish the surface, and criterion 12 is about
    reporting that honestly (the two cuts carry the same JSON either way).
    """
    i_close = rest.find(UI_CLOSE)
    i_open = rest.find(UI_OPEN)
    end = _structural_end(body + rest)
    i_struct = None if end is None else max(0, end - len(body))
    written_first = i_close >= 0 and i_struct is not None and i_struct <= i_close
    if written_first and not rest[i_struct:i_close].strip():
        return i_close, "written"
    best: tuple[int, FenceClose] | None = None
    candidates: tuple[tuple[int | None, FenceClose], ...] = (
        (i_close if i_close >= 0 else None, "written"),
        (i_struct, "inferred"),
        (i_open if i_open >= 0 else None, "abandoned"),
    )
    for index, ending in candidates:
        if index is not None and (best is None or index < best[0]):
            best = (index, ending)
    return best if best is not None else (0, None)


def _structural_end(text: str) -> int | None:
    r"""End of the concatenated JSON in ``text``, or ``None`` while it cannot be told yet.

    A bracket/quote scanner, deliberately **not** a parser: the payload this exists for is
    malformed (``{"Text","text":…``, a run of ``}}}}``) and a parser would refuse the whole thing.
    The rules, in the order they bite:

    - characters inside a string, and any character after a ``\``, are text;
    - depth clamps at zero, so a surplus closer cannot drive the scanner backwards;
    - at depth zero the trailing run of closers belongs to the JSON — otherwise a stray ``}`` leads
      the spoken remainder and TTS says "close brace";
    - A2UI surfaces are *concatenated* top-level objects, so depth zero is an ending only when the
      remainder does not resume with ``{``;
    - an empty remainder is not an ending yet: the next canvas decides. Cutting here would end a
      surface between two of its own messages.
    """
    depth = 0
    opened = False
    in_string = False
    escaped = False
    index = 0
    size = len(text)
    while index < size:
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
            opened = True
        elif char in "}]":
            depth = max(0, depth - 1)
            if opened and depth == 0:
                closers = index + 1
                while closers < size and text[closers] in "}]":
                    closers += 1
                after = closers
                while after < size and text[after].isspace():
                    after += 1
                if after >= size:
                    return None
                if text[after] == "{":
                    index = closers
                    continue
                return closers
        index += 1
    return None


def _resumes_marker(text: str) -> bool:
    """Does ``text`` start with a fence marker, or is it still a prefix of one?"""
    return any(text.startswith(m) or m.startswith(text) for m in (UI_CLOSE, UI_OPEN))


def _hold_partial(text: str) -> tuple[str, str]:
    """Split a trailing *partial* fence marker off the end of a canvas, to hold for the next one.

    Before the carry, ``_strip_markup`` dropped it (L10): a canvas truncated mid-fence ends in
    ``<a2u`` and reading a half-open tag aloud is worse than losing a trailing ``<``. With a carry
    the fragment must be held instead, or the opening of a real surface is lost — the fence then
    never opens and the whole surface is spoken. It is still dropped if it never becomes a marker
    (see :func:`split_canvas`), so a response that ends while holding one behaves as it did.
    """
    for marker in (UI_CLOSE, UI_OPEN):  # longest first: "</a2ui>" before "<a2ui>"
        for cut in range(len(marker) - 1, 0, -1):
            if text.endswith(marker[:cut]):
                return text[:-cut], marker[:cut]
    return text, ""


def _strip_markup(text: str) -> str:
    """Remove fence markup from a piece of *answer* text before it can be voiced (L10).

    A stray ``</a2ui>``: a close with no opening fence is not a fence, it is markup the model
    emitted, and TTS would read "less than slash a two u i". A complete ``<a2ui>`` cannot be here
    (the splitter partitions answer text on it) and a *partial* marker at the end has already been
    taken by :func:`_hold_partial`. Only this; anything else in the answer channel is the answer.
    """
    return text.replace(UI_CLOSE, "")


# -- id mode: a tools-on stream read by its token ids (T-0088) -----------------------------------------------
#
# T-0087 reading B: with ``tool_choice: "none"`` the served ``content`` is the answer cut at its first special
# piece and the call is in no field, so the client decodes each chunk's ``token_ids`` itself (T-0088a's host
# detokeniser) up to the first stop id and splits the decode D in two levels, both incremental (criterion 9):
#
# 1. :class:`_ThoughtSplit` takes thoughts out of D the way ``router`` does (``<|channel>`` to the next
#    ``<channel|>``; an unclosed one runs to the end). What is left is the answer channel A, router's scanned text.
#    One departure (T-0170): unless the reader starts in the tool state, text at the start of the response is held
#    until the first ``<|channel>``, ``<channel|>`` or call tag (or the end); when that first marker is an orphan
#    ``<channel|>``, the text before it is an implicit thought, never answer (``router.answer_channel`` keeps it).
# 2. :class:`_AnswerSplit` finds the first call tag in A as ``router.route`` finds it, before any special literal
#    is removed (a removal can never make a tag), except inside an open UI fence. Text before the tag goes through
#    :func:`split_canvas` (the one fence and M9 implementation) and then ``drop_special_literals``; everything from
#    the tag on is the ``tool`` text, finished by ``router.route`` itself at the end of the response.
#
# A canvas can end inside the spelling of a marker (a special is one id, but its text can also arrive in plain
# pieces), so each level holds a trailing fragment for the next canvas and gives it back as text when it never
# becomes a marker (ruling E7). The answer level holds exactly the longest suffix that, once special literals are
# removed, is a proper prefix of a recognised marker (ruling E11; removal can splice a new literal, which is why
# it is read after removal), and releases everything before it, in time linear in the text. The deletion of
# literals that begin with ``<`` and contain no other ``>`` than their last is confluent, so the order of removal
# does not matter. A fragment still held when the response ends is dropped (L10).

#: Markers a canvas may end inside (criterion 6): the call tags, the channel and UI markers, every special
#: literal. Each begins with ``<`` and has no ``>`` but its last character.
_MARKERS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *router.TAGS.values(),
            router.CHANNEL_OPEN,
            router.CHANNEL_CLOSE,
            UI_OPEN,
            UI_CLOSE,
            *SPECIAL_TOKENS,
        )
    )
)
_MARKER_PREFIXES = frozenset(m[:k] for m in _MARKERS for k in range(1, len(m)))
_CHANNEL_OPEN_PREFIXES = frozenset(
    router.CHANNEL_OPEN[:k] for k in range(1, len(router.CHANNEL_OPEN))
)
_CHANNEL_CLOSE_PREFIXES = frozenset(
    router.CHANNEL_CLOSE[:k] for k in range(1, len(router.CHANNEL_CLOSE))
)
_LONGEST_MARKER = max(len(m) for m in _MARKERS)
#: the header the engine writes after ``<|channel>``; not thought text (T-0087 ``regions()``)
THOUGHT_HEADER = "thought\n"
_TAG_LITERALS: tuple[str, ...] = tuple(router.TAGS.values())
#: T-0170: the markers that end the implicit-thought window (criterion 2)
_WINDOW_MARKERS: tuple[str, ...] = (router.CHANNEL_OPEN, router.CHANNEL_CLOSE, *_TAG_LITERALS)
_LONGEST_WINDOW_MARKER = max(len(m) for m in _WINDOW_MARKERS)


def _tail(text: str, prefixes: frozenset[str]) -> int:
    """Length of the longest suffix of ``text`` in ``prefixes`` (0 for none)."""
    for k in range(min(_LONGEST_MARKER, len(text)), 0, -1):
        if text[-k:] in prefixes:
            return k
    return 0


_SPECIAL_SET = frozenset(SPECIAL_TOKENS)
_SPECIAL_LENGTHS: tuple[int, ...] = tuple(sorted({len(s) for s in SPECIAL_TOKENS}))
#: a run of removed special literals longer than this inside a held fragment is kept as one literal, so the
#: fragment, and the work of re-reading it with the next canvas, stays bounded (ruling E11)
_MAX_HELD_RUN = 64


def _short_run(run: str) -> str:
    """A removed run as held: verbatim when short, else special literals with the same effect on the split: removed
    from the text, breaking any spelling across them, keeping M9's ``<|think|>``, and (ruling E15) keeping the
    parity of ``"`` so ``_structural_end``'s inside-a-string state is unchanged. A run with both gets both."""
    if len(run) <= _MAX_HELD_RUN:
        return run
    think = "<|think|>" if any(m in run for m in THOUGHT_MARKERS) else ""
    quote = '<|"|>' if run.count('"') % 2 else ""
    return think + quote or "<pad>"


def _hold(text: str) -> tuple[int, str]:
    """``(cut, held)``: ``text[:cut]`` may be emitted now; ``held`` (``text[cut:]``, long removed runs shortened) is
    kept for the next canvas. Ruling E11: the held part is the longest suffix of ``drop_special_literals(text)``
    that is a proper prefix of a recognised marker, and nothing before it.

    The removal is done in one pass with a stack: every special ends in ``>`` and has no other ``>``, so a literal
    can only be completed when a ``>`` is pushed, and only at the top. Each kept character remembers where it came
    from in ``text``, which is where the cut goes. Linear in ``text``. A raw trailing marker prefix (a tag spelled
    in plain pieces) has no ``>``, so it is never removed and is always inside the held suffix."""
    size = len(text)
    if "<" not in text:
        return size, ""  # no marker can begin here
    if ">" not in text:
        keep = _tail(text, _MARKER_PREFIXES)  # nothing is removed: the text is its own removal
        return size - keep, text[size - keep :]
    chars: list[str] = []
    origin: list[int] = []
    start = 0
    while start < size:
        end = text.find(">", start)
        stop = size if end < 0 else end + 1
        chars.extend(text[start:stop])
        origin.extend(range(start, stop))
        if end >= 0:
            for length in _SPECIAL_LENGTHS:
                if length <= len(chars) and "".join(chars[-length:]) in _SPECIAL_SET:
                    del chars[-length:]
                    del origin[-length:]
                    break
        start = stop
    keep = _tail("".join(chars[-_LONGEST_MARKER:]), _MARKER_PREFIXES)
    if not keep:
        return size, ""
    first = len(chars) - keep
    cut = origin[first]
    parts: list[str] = []
    prev = cut
    for index in origin[first:]:
        parts.append(_short_run(text[prev:index]))
        parts.append(text[index])
        prev = index + 1
    parts.append(_short_run(text[prev:]))
    return cut, "".join(parts)


class _ThoughtSplit:
    """Level 1: D in, thought pieces and answer-channel text out (thoughts removed as ``router`` removes them).

    T-0170, the implicit-thought window (``implicit=True``: id mode, a reader not started in the tool state). The
    model sometimes skips ``<|channel>thought\\n`` but still writes the closing ``<channel|>``. So, from the start of
    the response until the first window marker (``<|channel>``, ``<channel|>`` or a ``router.TAGS`` literal, written
    as one special or spelled in plain pieces) or the end of the response, D is held whole and nothing is released:

    - the first marker is ``<channel|>`` at j: ``D[:j]`` (a leading ``thought\\n`` header removed) is an implicit
      thought, yielded as ``thought`` pieces marked ``implicit_thought``; the marker is consumed and the rest is read
      as below with no thought open, so a later stray close keeps today's handling;
    - the first marker is ``<|channel>`` or a call tag, or the response ends with none: the held text is released
      unchanged and read exactly as before T-0170.

    A call tag ends the window wherever it stands, **also inside a UI fence** opened before any channel marker: the
    window is decided on the raw text, before level 2 knows about fences, so a ``<channel|>`` after such a tag is a
    stray close, read as before.

    **Departure from ``router.answer_channel``** (deliberate, unchanged router): for a D whose first marker is an
    orphan ``<channel|>``, ``answer_channel`` keeps ``D[:j]`` in the answer channel; this split makes it thought, so it
    is never voiced nor stored in history. Every other D reads as ``router`` reads it. Trade-off (criterion 12): a
    ``<channel|>`` the model meant literally as its first marker costs the text before it.

    The window search is linear: each chunk is searched together with only the last ``len(longest marker) - 1``
    held characters (no complete marker lies wholly in text already searched), and the held text is joined once,
    at the decision."""

    def __init__(self, implicit: bool = False) -> None:
        self.in_thought = False
        #: the current thought's leading ``thought\n`` header is not decided yet
        self._header = False
        self._held = ""
        #: T-0170: the text of the open implicit-thought window, by chunk; None once decided (or never opened)
        self._window: list[str] | None = [] if implicit else None
        #: the window's last characters, too few to hold a whole marker, searched again with the next chunk
        self._window_tail = ""

    def feed(self, text: str, final: bool) -> tuple[list[Piece], str]:
        implicit = ""
        if self._window is not None:
            self._window.append(text)
            found = self._window_marker(text)
            if found is None and not final:
                return [], ""
            whole = "".join(self._window)
            self._window, self._window_tail = None, ""
            if found is not None and found[1] == router.CHANNEL_CLOSE:
                j = len(whole) - found[0]
                implicit = whole[:j]
                if implicit.startswith(THOUGHT_HEADER):
                    implicit = implicit[len(THOUGHT_HEADER) :]
                text = whole[j + len(router.CHANNEL_CLOSE) :]
            else:
                text = whole
        thoughts, answer = self._split(text, final)
        pieces = [Piece("thought", implicit, implicit_thought=True)] if implicit else []
        return pieces + [Piece("thought", t) for t in thoughts], answer

    def _window_marker(self, text: str) -> tuple[int, str] | None:
        """The first window marker in the held tail plus ``text``: ``(how many characters before the end of the
        window's text it starts, the marker)``, or None (and the tail moves on)."""
        scan = self._window_tail + text
        found = [(i, m) for m in _WINDOW_MARKERS if (i := scan.find(m)) >= 0]
        if not found:
            keep = min(len(scan), _LONGEST_WINDOW_MARKER - 1)
            self._window_tail = scan[len(scan) - keep :]
            return None
        i, marker = min(found)
        return len(scan) - i, marker

    def _split(self, text: str, final: bool) -> tuple[list[str], str]:
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


class _AnswerSplit:
    """Level 2: answer-channel text in, ``answer``/``ui`` pieces out, and the raw ``tool`` text from the tag on."""

    def __init__(self, tool_prefix: str | None) -> None:
        self.fence = FenceState()
        self._pending = ""
        self._tool: list[str] | None = None if tool_prefix is None else [tool_prefix]

    def tool_raw(self) -> str | None:
        return None if self._tool is None else "".join(self._tool)

    def feed(self, text: str, final: bool) -> list[Piece]:
        if self._tool is not None:
            self._tool.append(text)
            return []
        pending = self._pending + text
        i = self._tag_at(pending)
        if i is not None:
            self._pending = ""
            self._tool = [pending[i:]]
            return self._split(pending[:i])
        cut, held = _hold(pending)
        # a fragment still held when the response ends is dropped (L10)
        self._pending = "" if final else held
        return self._split(pending[:cut])

    def _tag_at(self, text: str) -> int | None:
        """Where the first call tag outside an open UI fence starts in ``text`` (``route``'s search: on the text
        as written, before any special literal is removed)."""
        start = 0
        while True:
            found = [i for i in (text.find(t, start) for t in _TAG_LITERALS) if i >= 0]
            if not found:
                return None
            i = min(found)
            if not self._fence_open_after(text[:i]):
                return i
            start = i + 1  # a tag inside an open UI fence is UI text

    def _fence_open_after(self, head: str) -> bool:
        if not self.fence.open and UI_OPEN not in head:
            return False
        return split_canvas(None, head, self.fence, hold=False)[1].open

    def _split(self, text: str) -> list[Piece]:
        if not text:
            return []
        pieces, self.fence = split_canvas(None, text, self.fence, hold=False)
        # human Q2 / ruling E8: answer and ui text lose every special literal, after the tag search
        return [
            p if p.segment == "thought" else replace(p, text=drop_special_literals(p.text))
            for p in pieces
        ]


def _first_choice(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    first: dict[str, Any] = choices[0]
    return first


class _IdReader:
    """One tools-on response read by its ids. A local of the ``generate`` frame (T-0052 D1)."""

    def __init__(self, detokenizer: HostDetokenizer, tool_prefix: str | None) -> None:
        self._stream = detokenizer.stream()
        # T-0170 criterion 10: a reader started in the tool state (a continued call) never opens the
        # implicit-thought window; its text is call arguments, not a possible thought
        self._thoughts = _ThoughtSplit(implicit=tool_prefix is None)
        self._answer = _AnswerSplit(tool_prefix)
        self._decoding = True
        #: the first id of ``STOP_IDS`` in the stream (never ``stop_reason``); None for a length cut
        self.stop_id: int | None = None
        #: a chunk with a non-null ``finish_reason`` has been read
        self.finished = False
        #: T-0171: a chunk carrying a ``token_ids`` key has been read (whatever its value)
        self.ids_seen = False
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
        if choice is not None and "token_ids" in choice:
            self.ids_seen = True
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
        pieces, answer = self._thoughts.feed(text, final)
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
