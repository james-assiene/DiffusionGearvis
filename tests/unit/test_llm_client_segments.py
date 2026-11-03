"""LLM client: segment separation and request shape (docs/10 §9, docs/11 §2.1). Task T-0005."""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from local_astra.conductor.llm_client import (
    MAX_EARLY_ABORTS,
    THOUGHT_MARKERS,
    UI_CLOSE,
    UI_OPEN,
    VllmClient,
    VllmEndpoint,
    split_segments,
)
from local_astra.conductor.prompt_builder import PromptBuilder
from local_astra.contracts.llm import (
    SAMPLING_WHITELIST,
    THINKING_TEMPLATE_KWARG,
    ImagePart,
    LlmOptions,
    Message,
    PrefillHandle,
    Prompt,
    TextPart,
)
from local_astra.contracts.perception import AssistFlags, Frame, Source
from local_astra.contracts.turn import HistoryTurn

#: served model id = the HF repo id (measured 2026-09-13, logs/vllm-plugin-on.log line 8:
#: ``'model': 'nvidia/diffusiongemma-26B-A4B-it-NVFP4'``).
SERVED_MODEL = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"

#: Everything a chat-completions body may contain besides the sampling whitelist (criterion 1).
NON_SAMPLING_KEYS = frozenset({"model", "messages", "stream", "chat_template_kwargs", "vllm_xargs"})


def _sse(*chunks: dict) -> str:
    """One SSE chunk per committed canvas (docs/02 §2.4), terminated by the OpenAI sentinel."""
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def _delta(**fields: str) -> dict:
    return {"choices": [{"delta": dict(fields)}]}


def _prompt(*texts: str) -> Prompt:
    msgs = tuple(
        Message(role="user" if i % 2 == 0 else "assistant", content=(TextPart(text=t),))
        for i, t in enumerate(texts)
    )
    return Prompt(messages=msgs, prefix_len=len(msgs))


def _client(handler, **kw) -> VllmClient:
    return VllmClient(transport=httpx.MockTransport(handler), **kw)


def _json_handler(sse: str, seen: list[dict] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.read()))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)

    return handler


# --- active ------------------------------------------------------------------------------------------


def test_endpoint_must_be_loopback():
    with pytest.raises(ValueError):
        VllmEndpoint(base_url="http://10.0.0.5:8001/v1")
    assert VllmClient().endpoint.base_url.startswith("http://127.0.0.1")


def test_ui_fence_tokens_are_distinct_and_closed():
    assert UI_OPEN == "<a2ui>" and UI_CLOSE == "</a2ui>"  # docs/12 §1.1, docs/02 §2.2


# --- T-0005 (activated for /implement; formerly pending) ------------------------------------------


def test_reasoning_delta_is_thought_and_content_delta_is_answer():
    assert split_segments("let me think", None, False) == [("thought", "let me think", False)]
    assert split_segments(None, "Four.", False) == [("answer", "Four.", False)]


def test_ui_fence_splits_content_into_answer_and_ui_and_carries_state():
    pieces = split_segments(None, 'Here you go. <a2ui>{"version":', False)
    assert pieces == [("answer", "Here you go. ", False), ("ui", '{"version":', True)]
    pieces = split_segments(None, '"v0.9.1"}</a2ui> Anything else?', True)
    assert pieces == [("ui", '"v0.9.1"}', False), ("answer", " Anything else?", False)]


def test_a_delta_with_both_channels_yields_thought_then_answer():
    assert split_segments("done thinking", "Four.", False) == [
        ("thought", "done thinking", False),
        ("answer", "Four.", False),
    ]


def test_empty_deltas_yield_nothing():
    assert split_segments(None, None, False) == []
    assert split_segments("", "", True) == []


def test_request_body_never_contains_forbidden_fields_and_carries_thinking_and_request_id():
    """Pure body builder: whitelist only, plugin args under vllm_xargs, thinking via the template."""
    from local_astra.contracts.llm import (
        THINKING_TEMPLATE_KWARG,
        LlmOptions,
        Message,
        Prompt,
        TextPart,
    )

    prompt = Prompt(
        messages=(Message(role="user", content=(TextPart(text="hi"),)),),
        prefix_len=1,
        thinking=True,
    )
    body = VllmClient().request_body(prompt, "r1", LlmOptions(top_k=40))
    assert body["stream"] is True and body["top_k"] == 40 and body["model"]
    assert not ({"temperature", "seed", "response_format", "logit_bias", "min_p"} & set(body))
    assert body["vllm_xargs"]["astra_request_id"] == "r1"
    assert body["chat_template_kwargs"] == {THINKING_TEMPLATE_KWARG: True}
    assert body["messages"][0]["role"] == "user"


async def test_generate_streams_through_an_injected_transport():
    """The transport is injected (httpx.MockTransport); no private attribute is monkeypatched."""
    import httpx

    from local_astra.contracts.llm import Message, Prompt, TextPart

    seen: dict = {}
    sse = (
        'data: {"choices":[{"delta":{"reasoning":"hmm"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"Four."}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)

    client = VllmClient(transport=httpx.MockTransport(handler))
    prompt = Prompt(messages=(Message(role="user", content=(TextPart(text="hi"),)),), prefix_len=1)
    commits = [c async for c in client.generate(prompt, "r1")]
    assert [(c.segment, c.text) for c in commits] == [("thought", "hmm"), ("answer", "Four.")]
    assert b'"stream": true' in seen["body"] or b'"stream":true' in seen["body"]


# --- T-0005 extensions written 2026-09-13 (criteria 1-3; see the test-author report) -------------


def test_request_body_contains_nothing_outside_the_whitelist_and_the_served_repo_id():
    """Criterion 1 the other way round: a *closed* key set, not a list of things to avoid.

    Only ``top_k``/``top_p``/``max_tokens``/``stop`` are accepted per request on the diffusion
    path; everything else (``temperature``, ``min_p``, ``seed``, ``min_tokens``, ``logit_bias``,
    ``bad_words``, ``allowed_token_ids``, ``response_format``) is a hard 400 — measured
    2026-09-13, PROGRESS.md "X3 part 2". So the body may only hold the whitelist plus the five
    structural keys.
    """
    client = VllmClient()
    bare = client.request_body(_prompt("hi"), "r1", None)
    assert set(bare) <= NON_SAMPLING_KEYS | SAMPLING_WHITELIST
    assert set(bare) & SAMPLING_WHITELIST == set(), "no sampling field is sent unless asked for"
    assert bare["model"] == SERVED_MODEL

    full = client.request_body(
        _prompt("hi"), "r1", LlmOptions(top_k=40, top_p=0.9, max_tokens=64, stop=("</a2ui>",))
    )
    assert set(full) == NON_SAMPLING_KEYS | SAMPLING_WHITELIST
    assert full["stop"] == ["</a2ui>"] and full["max_tokens"] == 64 and full["top_p"] == 0.9


def test_request_body_is_pure_and_returns_a_fresh_dict_each_call():
    """Criterion 1 says *pure*: two calls agree, and mutating the result cannot poison the next."""
    client = VllmClient()
    prompt = _prompt("hi")
    first = client.request_body(prompt, "r1", LlmOptions(top_k=40))
    second = client.request_body(prompt, "r1", LlmOptions(top_k=40))
    assert first == second
    first["model"] = "clobbered"
    first["vllm_xargs"]["astra_request_id"] = "clobbered"
    third = client.request_body(prompt, "r1", LlmOptions(top_k=40))
    assert third == second
    assert prompt.messages[0].content[0].text == "hi", "the prompt is not mutated"


def test_serialised_prefix_is_a_byte_prefix_of_the_serialised_full_request():
    """D9 (criterion 4) survives serialisation: prefill and final must hit the same cache blocks.

    Built by hand here, not with PromptBuilder, so this fails on a client that re-orders or
    re-encodes message parts even when the builder is correct.
    """
    client = VllmClient()
    frame = ImagePart(url="data:image/jpeg;base64,AAAA", phash="a" * 16)
    prefix = Prompt(
        messages=(
            Message(role="system", content=(TextPart(text="sys"),)),
            Message(role="user", content=(TextPart(text="look:"), frame)),
        ),
        prefix_len=2,
    )
    full = Prompt(
        messages=(
            prefix.messages[0],
            Message(
                role="user",
                content=(*prefix.messages[1].content, TextPart(text="what is the total")),
            ),
        ),
        prefix_len=2,
    )
    pre_body = client.request_body(prefix, "p1", None)
    full_body = client.request_body(full, "r1", None)
    assert full_body["messages"][0] == pre_body["messages"][0]
    pre_last, full_last = pre_body["messages"][-1], full_body["messages"][-1]
    assert full_last["role"] == pre_last["role"]
    assert full_last["content"][: len(pre_last["content"])] == pre_last["content"]
    assert len(full_body["messages"]) == len(pre_body["messages"])


def test_thinking_false_still_sends_the_kwarg_explicitly():
    """The template kwarg is session-level; omitting it when False makes the prefix differ
    between a thinking and a non-thinking session in a way the server cannot see (docs/01 D11)."""
    body = VllmClient().request_body(
        _prompt("hi").model_copy(update={"thinking": False}), "r1", None
    )
    assert body["chat_template_kwargs"] == {THINKING_TEMPLATE_KWARG: False}


def test_repeated_ui_fences_in_one_delta_split_in_order_and_leave_the_state_open():
    """Criterion 2, hand-computed: a,<open>X<close>,b,<open>Y -> 4 pieces, ui still open."""
    pieces = split_segments(None, f"a{UI_OPEN}X{UI_CLOSE}b{UI_OPEN}Y", False)
    assert pieces == [
        ("answer", "a", False),
        ("ui", "X", False),
        ("answer", "b", False),
        ("ui", "Y", True),
    ]


def test_a_fence_that_closes_with_nothing_after_it_yields_only_the_ui_piece():
    assert split_segments(None, f'{{"version":"v0.9.1"}}{UI_CLOSE}', True) == [
        ("ui", '{"version":"v0.9.1"}', False)
    ]


def test_reasoning_is_never_reported_as_answer_even_when_it_contains_a_fence():
    """The UI fence lives in the *answer* channel (criterion 2). A ``<a2ui>`` the model writes
    while still thinking must stay thought — voicing it, or rendering it, would leak the thought."""
    assert split_segments(f"maybe {UI_OPEN} here", None, False) == [
        ("thought", f"maybe {UI_OPEN} here", False)
    ]


async def test_commits_carry_request_id_canvas_index_and_thought_open_per_canvas():
    """Criterion 2: one commit per segment, ``canvas_idx`` per SSE chunk (= one committed canvas,
    docs/02 §2.4); a canvas that ends inside reasoning sets ``thought_open`` on *its* commit."""
    sse = _sse(
        _delta(reasoning="step one"),  # canvas 0: ends inside the thought
        _delta(reasoning=" done.", content="Four."),  # canvas 1: thought closed, answer started
    )
    commits = [c async for c in _client(_json_handler(sse)).generate(_prompt("hi"), "r7", None)]
    assert [(c.canvas_idx, c.segment, c.text) for c in commits] == [
        (0, "thought", "step one"),
        (1, "thought", " done."),
        (1, "answer", "Four."),
    ]
    assert [c.thought_open for c in commits] == [True, False, False]
    assert {c.request_id for c in commits} == {"r7"}


async def test_stream_ignores_keepalive_comments_and_empty_deltas():
    """SSE comment lines and role-only first chunks are part of the OpenAI wire format
    (docs/02 §5); a client that crashes on them, or emits an empty commit, is broken."""
    sse = (
        ": keep-alive\n\n"
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        "\n"
        'data: {"choices":[{"delta":{"content":"Four."}}]}\n\n'
        "data: [DONE]\n\n"
    )
    commits = [c async for c in _client(_json_handler(sse)).generate(_prompt("hi"), "r1", None)]
    assert [(c.segment, c.text) for c in commits] == [("answer", "Four.")]


async def test_reasoning_delta_is_recognised_under_both_vllm_spellings():
    """UNMEASURED WIRE DETAIL — see the test-author report.

    The delta key for the reasoning channel has never been captured on this box: the acceptance
    test above uses ``reasoning``; ``scripts/gpu/x3_prefix_cache.sh`` reads ``reasoning_content``
    from the non-streaming message and got ``''`` (thinking was off, so it proves nothing).
    Exactly one of the two is what vLLM v0.29.0 emits with ``--reasoning-parser gemma4``. Until a
    ``gpu`` contract test settles it, the client must accept both — that assertion cannot be the
    wrong one, whereas picking one would be a coin flip baked into the implementation.
    """
    for key in ("reasoning", "reasoning_content"):
        sse = _sse({"choices": [{"delta": {key: "hmm"}}]})
        commits = [c async for c in _client(_json_handler(sse)).generate(_prompt("hi"), "r1", None)]
        assert [(c.segment, c.text) for c in commits] == [("thought", "hmm")], key


async def test_abort_stops_an_in_flight_stream_closes_it_and_is_idempotent():
    """Criterion 3. The stream has six canvases; abort after the first must end the iteration,
    close the HTTP response, and a second abort must not raise."""

    class RecordingStream(httpx.AsyncByteStream):
        def __init__(self, payloads: list[bytes]) -> None:
            self.payloads = payloads
            self.sent = 0
            self.closed = False

        async def __aiter__(self):
            for chunk in self.payloads:
                self.sent += 1
                yield chunk

        async def aclose(self) -> None:
            self.closed = True

    payloads = [
        f'data: {{"choices":[{{"delta":{{"content":"c{i}"}}}}]}}\n\n'.encode() for i in range(6)
    ]
    stream = RecordingStream([*payloads, b"data: [DONE]\n\n"])
    client = _client(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream
        )
    )
    seen = []
    async for commit in client.generate(_prompt("hi"), "r1", None):
        seen.append(commit.text)
        if len(seen) == 1:
            await client.abort("r1")
    assert seen == ["c0"], f"abort did not stop the stream: {seen}"
    assert stream.closed, "abort() must close the SSE stream (criterion 3)"
    await client.abort("r1")  # idempotent: a second abort is a no-op
    await client.abort("never-started")


async def test_prefill_sends_the_prefix_only_and_returns_a_handle_for_that_request_id():
    """Criterion 4 / docs/02 §5: prefill warms exactly the prefix and stays abortable by id."""
    seen: list[dict] = []
    prefix = _prompt("sys", "warm me")
    handle = await _client(_json_handler(_sse(), seen)).prefill(prefix, "p9")
    assert isinstance(handle, PrefillHandle) and handle.request_id == "p9"
    assert len(seen) == 1, "one HTTP request per prefill"
    body = seen[0]
    assert [m["role"] for m in body["messages"]] == [m.role for m in prefix.messages]
    assert body["vllm_xargs"]["astra_request_id"] == "p9", "abort() must be able to target it"
    assert set(body) <= NON_SAMPLING_KEYS | SAMPLING_WHITELIST


# --- T-0005 round 2: the two modules must compose (amendment B9, review finding H1) -------------
#
# H1 reached review because *nothing ran the prompt builder and the client together*: the builder
# produced an ``ImagePart`` for a history frame and ``_part_body`` rejected every non-``data:``
# URL, so an ordinary second turn raised ValueError before a byte left the process. Both modules
# were green in isolation. These tests cross the seam.

#: the current screen's JPEG; distinct bytes from the history frame so the two data URLs differ.
CURRENT_JPEG = b"\xff\xd8\xff\xe0current-screen"
HISTORY_JPEG = b"\xff\xd8\xff\xe0history-screen"

HISTORY_CAPTION = "a spreadsheet of monthly totals"


def _data_url(jpeg: bytes) -> str:
    """Independent oracle for the wire form of a local frame (``ImagePart`` contract)."""
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def _two_turn(resolve=None) -> tuple[Prompt, Prompt]:
    """A perfectly ordinary second turn: one past turn that had a frame, plus the current one.

    The history turn is *young* (one turn old, under ``image_age_turns``), which is exactly the
    case where the builder wants to send pixels it does not hold.
    """
    pb = PromptBuilder("You are Astra.")
    history = (
        HistoryTurn(
            user_text="what am I looking at",
            assistant_spoken="a spreadsheet.",
            frame_phash="c" * 16,
            frame_caption=HISTORY_CAPTION,
        ),
    )
    frame = Frame(
        source=Source.SCREEN,
        ts_ms=2000,
        width=1280,
        height=800,
        jpeg=CURRENT_JPEG,
        phash="d" * 16,
    )
    prefix = pb.prefix(history, frame, AssistFlags(), resolve=resolve)
    full = pb.full(prefix, "and what is the total?", None, [])
    return prefix, full


def _image_urls(body: dict) -> list[str]:
    return [
        part["image_url"]["url"]
        for message in body["messages"]
        for part in message["content"]
        if part["type"] == "image_url"
    ]


def test_a_two_turn_prompt_built_without_a_resolver_serialises():
    """B9's default composes: builder output goes straight into the client and out as JSON.

    Without a resolver the young history frame degrades to its caption line, so the only image on
    the wire is the current screen. Both the prefill prefix and the final request must serialise:
    the prefill sends the prefix verbatim, so a part the client rejects kills the turn twice.
    """
    client = VllmClient()
    prefix, full = _two_turn()
    pre_body = client.request_body(prefix, "p1", None)
    body = client.request_body(full, "r1", None)
    json.dumps(pre_body)  # a body that will not encode never reaches vLLM either
    wire = json.dumps(body)
    assert _image_urls(body) == [_data_url(CURRENT_JPEG)], "only the current frame is pixels"
    assert HISTORY_CAPTION in wire, "B9: the history frame degraded to its caption, not to nothing"
    assert "astra-frame" not in wire


async def test_a_two_turn_prompt_reaches_the_wire_through_generate():
    """The same composition through the real code path, transport injected: the request is built,
    sent and answered. ``generate`` builds the body itself, so a rejected part raises here even
    if a caller never touches ``request_body``."""
    seen: list[dict] = []
    _, full = _two_turn()
    client = _client(_json_handler(_sse(_delta(content="Ninety.")), seen))
    commits = [c async for c in client.generate(full, "r1", None)]
    assert [(c.segment, c.text) for c in commits] == [("answer", "Ninety.")]
    assert len(seen) == 1
    assert all(url.startswith("data:") for url in _image_urls(seen[0]))
    assert HISTORY_CAPTION in json.dumps(seen[0])


def test_a_two_turn_prompt_built_with_a_resolver_carries_both_frames_as_pixels():
    """The other half of B9: with the resolver T-0007/T-0009 will inject, the history frame keeps
    its pixels and both images travel as inline data URLs, in history-then-current order."""
    client = VllmClient()
    _, full = _two_turn(resolve=lambda phash: HISTORY_JPEG)
    body = client.request_body(full, "r1", None)
    assert _image_urls(body) == [_data_url(HISTORY_JPEG), _data_url(CURRENT_JPEG)]


def test_a_resolver_returning_none_never_puts_a_reference_url_on_the_wire():
    """Adversarial twin of the composition test (B9).

    A resolver is allowed to fail — the ring buffer evicts frames — and the builder then holds a
    phash and no bytes. The two acceptable outcomes are "degrade to the caption" and "refuse to
    build the request". The unacceptable one is a bare ``astra-frame:``/``file:``/``http:``
    reference in the body: vLLM would either 400 or, worse, fetch it, and the reviewer's H1 is
    exactly this shape.
    """
    client = VllmClient()
    _, full = _two_turn(resolve=lambda phash: None)
    try:
        body = client.request_body(full, "r1", None)
    except ValueError:
        return  # refusing is the other sanctioned outcome; nothing reached the wire
    for url in _image_urls(body):
        assert url.startswith("data:"), f"reference URL on the wire: {url!r}"
    assert HISTORY_CAPTION in json.dumps(body), "an unresolved frame degrades to its caption"


def test_the_client_refuses_an_image_part_that_is_not_inline_pixels():
    """Whatever the producer, an image the server would have to *fetch* must not be sent.

    This is the guard that made H1 visible instead of silent, and it has to survive B9: the
    builder no longer emits such a part, so nothing else in the suite covers the check, and a
    future producer (memory thumbnails, T-0009 clips) will meet it. A silently forwarded
    reference is a model that has gone blind mid-turn, which is the failure B9 is about.
    """
    prompt = Prompt(
        messages=(
            Message(
                role="user",
                content=(
                    TextPart(text="look:"),
                    ImagePart(url="astra-frame:" + "c" * 16, phash="c" * 16),
                ),
            ),
        ),
        prefix_len=1,
    )
    with pytest.raises(ValueError):
        # Refusing is the only correct outcome *here*: unlike the builder, the client has no
        # caption to fall back to, so dropping the part quietly would send a prompt about a
        # screen with no picture of it (B9: "raising is the right instinct").
        VllmClient().request_body(prompt, "r1", None)


def test_builder_output_keeps_the_d9_prefix_relation_through_the_client():
    """Criterion 4 across the seam: the serialised prefill body must be a prefix of the final
    one *as the client encodes it*, for a prompt the builder produced (the existing test builds
    the prompt by hand, so it cannot catch a builder that breaks the relation)."""
    client = VllmClient()
    prefix, full = _two_turn()
    assert prefix.shares_prefix_with(full)
    pre_body = client.request_body(prefix, "p1", None)
    full_body = client.request_body(full, "r1", None)
    assert len(full_body["messages"]) == len(pre_body["messages"])
    assert full_body["messages"][:-1] == pre_body["messages"][:-1]
    pre_last, full_last = pre_body["messages"][-1], full_body["messages"][-1]
    assert full_last["role"] == pre_last["role"]
    assert full_last["content"][: len(pre_last["content"])] == pre_last["content"]


# --- T-0005 round 2, B13: the three gaps the verifier found after the merge ----------------------
#
# The implementation of all three is merged and was believed correct "by code reading". These
# tests exist so that reading is not the standard: each one goes red when its guard is removed
# (mutation evidence in the test-author report), which is the only thing that makes the guard a
# behaviour rather than a comment.


class _ParkingStream(httpx.AsyncByteStream):
    """A stream that hands over one chunk, parks, and then fails the way a closed socket does.

    ``abort()`` closes the SSE response while a reader is parked inside ``aiter_lines``. On a real
    connection that reader wakes with an ``httpx`` error — ``ReadError`` from the transport,
    ``StreamClosed`` from httpx itself. ``MockTransport`` never owns a socket, so nothing in the
    suite can produce that wake-up unless a stream models it; H2 (a prefill abort raising into the
    caller) was invisible for exactly that reason.
    """

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.parked = asyncio.Event()  # set once the reader has asked for a second chunk
        self.released = asyncio.Event()  # set by aclose(), i.e. by abort()
        self.closed = False
        self.raised = False

    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"c0"}}]}\n\n'
        self.parked.set()
        await self.released.wait()
        if self.error is not None:
            self.raised = True
            raise self.error
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        self.closed = True
        self.released.set()


def _stream_client(stream: httpx.AsyncByteStream) -> VllmClient:
    return _client(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream
        )
    )


#: The two ways httpx reports "the response you were reading is gone" (0.28): a transport-level
#: network error and httpx's own stream-state error. ``VllmClient`` catches ``HTTPError`` and
#: ``StreamError``, which are their two disjoint base classes — one except clause covering only
#: one of them would leave the other to reach the Conductor.
CLOSE_ERRORS = [httpx.ReadError("connection closed by abort"), httpx.StreamClosed()]


# --- M5: the barge-in race — abort between issuing the request and the first token ---------------


async def test_abort_before_the_generator_is_first_pumped_yields_nothing() -> None:
    """M5. ``generate`` is an async generator: nothing in its body runs until the first
    ``__anext__``. Barge-in lands in precisely that window — the Conductor issues the request on
    end-of-turn and the user starts talking again before the first canvas comes back — and an
    abort that finds no registered request used to return silently, after which the generator
    started anyway and the turn was spoken. Zero commits, and no raise.
    """
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_delta(content="Four."), _delta(content=" Anything else?")),
        )

    client = _client(handler)
    stream = client.generate(_prompt("what is two plus two"), "r1", None)  # not pumped yet
    await client.abort("r1")

    commits = [c async for c in stream]  # pumped to exhaustion; must not raise

    assert commits == [], f"an abort before the first token was dropped: {commits}"
    assert sent == [], (
        "a request aborted before it started must not be sent at all: it would take one of the "
        "three sequence slots (--max-num-seqs 3) for a turn nobody is listening to"
    )


async def test_a_pre_start_abort_is_consumed_and_does_not_kill_the_next_request() -> None:
    """The tombstone M5 leaves behind must be spent by the request it was meant for.

    A tombstone that survives its request turns ``abort`` into a permanent block on that id, and
    one that leaks across ids silences an unrelated turn. Both are silent failures: the user just
    gets no answer. Two independent checks, no mutation of the client in between.
    """
    sse = _sse(_delta(content="Four."))
    client = _client(_json_handler(sse))

    aborted = client.generate(_prompt("hi"), "r1", None)
    await client.abort("r1")
    assert [c async for c in aborted] == []

    other = [c async for c in client.generate(_prompt("hi"), "r2", None)]
    assert [(c.segment, c.text) for c in other] == [("answer", "Four.")], (
        "the tombstone for r1 must not touch r2"
    )
    reused = [c async for c in client.generate(_prompt("hi"), "r1", None)]
    assert [(c.segment, c.text) for c in reused] == [("answer", "Four.")], (
        "the tombstone was not consumed by the request it aborted; r1 is now permanently dead"
    )


async def test_aborts_for_requests_that_never_start_stay_bounded() -> None:
    """Adversarial twin of M5: the fix must not become a leak.

    ``abort`` for an unknown id remembers it, and the Conductor aborts liberally (every
    end-of-turn aborts the anticipatory prefill, whether or not it ever ran). A session is
    long-lived, so "remember every id forever" would be an unbounded dict in the one process that
    must stay resident. Reaching into the private store on purpose: from outside the API this
    growth is invisible until the box swaps.
    """
    client = VllmClient()
    for i in range(5 * MAX_EARLY_ABORTS):
        await client.abort(f"ghost-{i}")
    assert len(client._aborted_early) <= MAX_EARLY_ABORTS


async def test_a_prefill_aborted_before_it_starts_sends_nothing_and_still_returns_a_handle() -> (
    None
):
    """The same race on the prefill side. The Conductor fires the anticipatory prefill and aborts
    it at end-of-turn; if the turn ends before the coroutine is first scheduled the abort lands in
    the pre-start window. ``abort`` does not suspend (no record, no await), so the task below is
    guaranteed not to have run when it returns — the race is deterministic here, not a sleep.
    """
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=_sse())

    client = _client(handler)
    task = asyncio.create_task(client.prefill(_prompt("sys", "warm me"), "p1"))
    await client.abort("p1")
    handle = await asyncio.wait_for(task, timeout=2)

    assert sent == [], "an aborted prefill must not warm anything"
    assert handle.request_id == "p1"
    # 0 means "the server did not report usage" (amendments B5/B12), which is also what an
    # aborted prefill knows: it never saw a usage block.
    assert handle.prefix_tokens == 0


# --- H2: aborting a prefill must not raise into the caller --------------------------------------


@pytest.mark.parametrize("error", CLOSE_ERRORS, ids=lambda e: type(e).__name__)
async def test_aborting_a_prefill_returns_instead_of_raising(error: BaseException) -> None:
    """H2. The Conductor aborts the anticipatory prefill at *every* end-of-turn (docs/01 D9,
    .claude/rules/conductor.md), so this is the happy path, not an error path: the awaited
    ``prefill()`` must come back normally while the reader is parked mid-stream and ``abort``
    closes the response under it.

    ``stream.raised`` is asserted so the test cannot pass by never reaching the failure at all —
    the exception really is thrown into ``prefill`` and really is swallowed there.
    """
    stream = _ParkingStream(error)
    client = _stream_client(stream)
    task = asyncio.create_task(client.prefill(_prompt("sys", "warm me"), "p1"))
    await asyncio.wait_for(stream.parked.wait(), timeout=2)

    await client.abort("p1")
    handle = await asyncio.wait_for(task, timeout=2)  # must not raise

    assert handle.request_id == "p1"
    assert stream.closed, "abort() must close the SSE stream (criterion 3)"
    assert stream.raised, "the parked reader never saw the close; the test proves nothing"


@pytest.mark.parametrize("error", CLOSE_ERRORS, ids=lambda e: type(e).__name__)
async def test_aborting_a_generate_mid_stream_returns_instead_of_raising(
    error: BaseException,
) -> None:
    """The same wake-up on the generate side. The existing abort test closes the stream between
    chunks, where nothing is parked and no exception is thrown; this one aborts while the reader
    waits, which is what actually happens at 30 canvases a turn. Commits already yielded stand;
    the iteration ends quietly.
    """
    stream = _ParkingStream(error)
    client = _stream_client(stream)
    seen: list[str] = []

    async def pump() -> None:
        async for commit in client.generate(_prompt("hi"), "r1", None):
            seen.append(commit.text)

    task = asyncio.create_task(pump())
    await asyncio.wait_for(stream.parked.wait(), timeout=2)
    await client.abort("r1")
    await asyncio.wait_for(task, timeout=2)  # must not raise

    assert seen == ["c0"]
    assert stream.raised, "the parked reader never saw the close; the test proves nothing"


@pytest.mark.parametrize("method", ["prefill", "generate"])
async def test_a_stream_failure_that_is_not_an_abort_still_propagates(method: str) -> None:
    """Adversarial twin of H2: the swallow must be conditional on *this request* being aborted.

    A blanket ``except httpx.HTTPError: pass`` would make both tests above green while turning
    every dropped connection, dead engine and half-sent response into a turn that quietly
    produces nothing. Here nobody aborts, so the error is the caller's to see.
    """

    class _Failing(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"c0"}}]}\n\n'
            raise httpx.ReadError("engine died mid-stream")

        async def aclose(self) -> None:
            return None

    client = _stream_client(_Failing())
    with pytest.raises(httpx.ReadError):
        if method == "prefill":
            await client.prefill(_prompt("sys", "warm me"), "p1")
        else:
            _ = [c async for c in client.generate(_prompt("hi"), "r1", None)]


async def test_an_unaborted_request_still_reports_an_http_error_status() -> None:
    """Second half of the same guard, on the status code rather than the socket: a 500 from the
    engine must reach the caller. ``raise_for_status`` is inside the same ``try``."""
    client = _client(lambda request: httpx.Response(500, text="engine on fire"))
    with pytest.raises(httpx.HTTPStatusError):
        _ = [c async for c in client.generate(_prompt("hi"), "r1", None)]


# --- M9: a think token in the content channel never reaches the answer ---------------------------
#
# If the reasoning parser is off, misconfigured, or the delta arrives under a third spelling
# neither REASONING_DELTA_KEYS entry covers, the model's thinking arrives as ordinary ``content``
# and would be read aloud. CLAUDE.md's rule is absolute — "never voice text the LLM has not
# committed ... never store assistant thinking" — so this is the belt to amendment B1's braces.


@pytest.mark.parametrize("marker", THOUGHT_MARKERS)
def test_content_carrying_a_think_marker_is_routed_to_thought_whole(marker: str) -> None:
    """M9: the *whole* piece goes to ``thought``, not just the marker.

    Splitting around the marker would leave the reasoning either side of it in ``answer``, which
    is the text that gets spoken; the sanctioned failure direction is the other one (a misrouted
    answer loses a sentence, a misrouted thought is said out loud).
    """
    delta = f"Okay {marker} the user wants the column total, it is ninety"
    assert split_segments(None, delta, False) == [("thought", delta, False)]


@pytest.mark.parametrize("marker", THOUGHT_MARKERS)
def test_no_answer_piece_is_produced_for_content_that_carries_a_think_marker(marker: str) -> None:
    """The invariant behind M9, over the shapes a broken parser actually produces: the marker
    alone, glued to prose, after a completed UI fence, and alongside a reasoning delta."""
    for content in (
        marker,
        f"prefix{marker}suffix",
        UI_OPEN + '{"version":"v0.9.1"}' + UI_CLOSE + f" {marker} no wait",
        f"Ninety. {marker} actually let me recount",
    ):
        for reasoning in (None, "already thinking"):
            pieces = split_segments(reasoning, content, False)
            assert pieces, content
            assert all(segment != "answer" for segment, _, _ in pieces), (
                f"a think marker reached the answer channel: {content!r} -> {pieces}"
            )


@pytest.mark.parametrize("marker", THOUGHT_MARKERS)
async def test_a_think_marker_in_content_never_reaches_an_answer_commit(marker: str) -> None:
    """The same through ``generate``, because that is what the Conductor consumes: only
    ``Commit(segment="answer")`` may feed the speech gate (.claude/rules/conductor.md), so the
    assertion is about commits, not pieces. ``thought_open`` on that canvas is the second lock —
    a canvas that ends inside the thought never reaches TTS even if the segment were wrong.
    """
    text = f"{marker} the total is ninety"
    sse = _sse(_delta(content=text))
    commits = [c async for c in _client(_json_handler(sse)).generate(_prompt("hi"), "r1", None)]
    assert [(c.segment, c.text) for c in commits] == [("thought", text)]
    assert all(c.thought_open for c in commits)


def test_ordinary_prose_about_thinking_is_still_the_answer() -> None:
    """Adversarial twin of M9: the guard must be a marker test, not a vibe.

    "I think" is the most ordinary sentence opener an assistant has, and ``<`` shows up in any
    answer about code or arithmetic. An over-broad guard passes every M9 test above by routing
    everything to ``thought`` — and the assistant then says nothing at all, which is the failure
    mode nobody notices in a unit suite.
    """
    for content in (
        "I think the total is ninety.",
        "Use a < b to compare them.",
        "The tag is written think, in angle brackets.",
    ):
        assert split_segments(None, content, False) == [("answer", content, False)], content


# --- T-0052 criterion 14: what the fence carry must NOT change -----------------------------------
#
# These three rows are green today and their job is to *stay* green when the fence starts carrying
# across the canvases of a request (T-0052 D1). They are deliberately not `pending`: a preservation
# row that goes red the moment the carry lands is the point of it.


async def test_a_fence_in_the_reasoning_channel_does_not_open_the_content_fence() -> None:
    """Criterion 14: the reasoning channel is thought, fence or no fence — through ``generate``.

    On the same 2026-09-16 session (ruling A3) the model's *first* attempt at the flight interface
    built the whole surface inside its reasoning channel, and the client was right to treat it as
    thought: the turn was silent, which is a prompt problem, not a client one. What must never
    happen is that ``<a2ui>`` arriving in ``reasoning`` opens the *content* channel's fence — the
    next canvas's prose would then be classified ``ui`` and never spoken, which is how a thought
    silences a turn it is not even part of.
    """
    sse = _sse(
        _delta(reasoning=f"maybe {UI_OPEN}" + '{"version":"v0.9.1"'),
        _delta(content="Four."),
    )
    commits = [c async for c in _client(_json_handler(sse)).generate(_prompt("hi"), "r1", None)]
    assert [(c.segment, c.text) for c in commits] == [
        ("thought", f"maybe {UI_OPEN}" + '{"version":"v0.9.1"'),
        ("answer", "Four."),
    ], f"a fence in the reasoning channel changed how the content channel was split: {commits}"


async def test_a_response_that_ends_holding_a_marker_fragment_never_voices_it() -> None:
    """Criterion 7's degenerate case and criterion 14's stray-marker rule, together.

    A canvas cut off mid-marker ends in ``<a2u``. Today ``_strip_markup`` drops it (L10); with a
    carry it must be *held* for the next canvas and dropped only if the response ends while it is
    held — which is exactly this response. Either way the observable outcome is the same and must
    stay the same: the prose is spoken verbatim, the fragment is neither voiced nor shown.
    """
    sse = _sse(_delta(content="All done. <a2u"))
    commits = [c async for c in _client(_json_handler(sse)).generate(_prompt("hi"), "r1", None)]
    assert [(c.segment, c.text) for c in commits] == [("answer", "All done. ")], (
        f"a held marker fragment must never reach a commit: {commits}"
    )


def test_the_flight_fixture_is_the_recorded_turn() -> None:
    """The fixture's own arithmetic, so a corrupt or re-cut fixture fails here and not inside a
    criterion's row. Every number is from ``logs/sessions/20260916-181929-3900425.jsonl`` (turn
    ``t6``); the reconstruction rule is in the fixture's ``note`` and in the directory MANIFEST.
    """
    from tests.fakes.llm import flight_canvases

    fix = flight_canvases()
    rec = fix["recorded"]
    canvases = fix["canvases"]

    assert [len(c["reasoning"] or "") + len(c["content"] or "") for c in canvases] == [
        848,
        788 + len(UI_OPEN),
        902,
    ], "the committed canvases were 848 / 788 / 902 characters; canvas 1 gains the '<a2ui>' marker"
    assert canvases[1]["content"] == UI_OPEN + rec["ui_piece_canvas1"]
    assert canvases[2]["content"] == rec["answer_piece_canvas2"]
    assert rec["json_tail_canvas2"] + rec["prose_tail_canvas2"] == rec["answer_piece_canvas2"]
    assert (len(rec["json_tail_canvas2"]), len(rec["prose_tail_canvas2"])) == (750, 152)
    assert rec["fenced_body_chars"] == len(rec["ui_piece_canvas1"]) + 750 == 1159
    assert rec["closing_fence_emitted"] is False, "the model never wrote a closing fence (A1)"
    assert UI_OPEN not in rec["answer_piece_canvas2"]
    assert UI_CLOSE not in rec["answer_piece_canvas2"]
    assert len(rec["tts_utterance_text"]) == 807, (
        "what Pocket TTS actually said, the negative oracle"
    )
    assert rec["tts_utterance_source"].startswith(rec["json_tail_canvas2"][:80])
