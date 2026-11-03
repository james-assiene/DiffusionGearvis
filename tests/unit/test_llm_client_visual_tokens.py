"""T-0131 A2: the per-request visual-token budget on the wire (docs/tasks/T-0131.md; docs/02 §5 wire format).

The real ``VllmClient`` over ``httpx.MockTransport``; the bodies are read off the wire (``request.read()``), so a
phase that bypasses ``request_body`` is caught as well as one that forgets the key.

The interface the rows pin (the task's): ``VllmClient(endpoint=None, transport=None, *, detokenizer=None,
visual_tokens: int | None = None)`` (passed by keyword). When set, every body ``request_body`` builds carries
``"mm_processor_kwargs": {"max_soft_tokens": N}`` and nothing else changes; when ``None`` the body is BASE's, byte
for byte. The value never travels through ``LlmOptions`` or ``vllm_xargs``.

Oracles: the key's value is the criterion's literal; "nothing else changes" compares with the body a default
client (no budget) builds for the same call, whose own bytes are pinned against BASE by the dict literal and the
golden bodies below (``request_bodies_9669258.json``, captured before any T-0088 code; the tools-off body has not
changed since, as ``test_llm_request_body_tools.py`` pins).

Every row is ``pending(task="T-0131")``: at BASE (f30365d) ``VllmClient`` takes no ``visual_tokens``, and the
helper turns that ``TypeError`` into a named ``AssertionError``.

Regime: CPU, no engine; n = 1, deterministic.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

import httpx
import pytest

from local_astra.conductor import llm_client
from local_astra.conductor.llm_client import VllmClient
from local_astra.contracts.cd import CdRequest
from local_astra.contracts.llm import ImagePart, LlmOptions, Message, Prompt, TextPart
from tests.fakes import vllm_sse as V
from tests.unit import _llm_id_support as S

KEY = "mm_processor_kwargs"
_JPEG = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP////////////////////////////////////////////////////////"
    "////////////////////////////////8AAAsIAAEAAQEBEQD/xAAUAAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAA/AH//2Q=="
)
AUTOMATON = "AAAA"
CD = CdRequest(
    mode="perstep_mar",
    automaton=AUTOMATON,
    automaton_sha256=hashlib.sha256(AUTOMATON.encode()).hexdigest(),
    record_tag="t0131-cd",
)
PLAIN_SSE = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'


def make_client(transport: httpx.AsyncBaseTransport | None = None, **kw: Any) -> VllmClient:
    """``VllmClient(transport=..., detokenizer=<subset>, **kw)``; a missing ``visual_tokens`` keyword is a named
    failure."""
    params = inspect.signature(llm_client.VllmClient.__init__).parameters
    assert "visual_tokens" in params, (
        "T-0131 A2: VllmClient(..., visual_tokens: int | None = None) does not exist yet"
    )
    return VllmClient(transport=transport, detokenizer=S.detok(), **kw)


def frame_prompt(**kw: Any) -> Prompt:
    """System + a user turn holding a frame and a question (tools off unless ``tools=`` is given)."""
    user = Message(
        role="user",
        content=(
            ImagePart(url=_JPEG, phash="0" * 16),
            TextPart(text="Click the blue Add to cart."),
        ),
    )
    return S.make_prompt(messages=(S.SYSTEM, user), prefix_len=2, **kw)


def serve(sent: list[dict[str, Any]]) -> httpx.MockTransport:
    """Records each JSON body; answers a tools-on body with a valid id-mode stream and anything else with prose."""
    id_lines = V.sse_text(S.stream_lines([[*S.thought("t"), *S.prose("Ok."), S.TURN_CLOSE]]))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        sent.append(body)
        text = id_lines if "tools" in body else PLAIN_SSE
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text)

    return httpx.MockTransport(handler)


async def every_phase(**kw: Any) -> dict[str, dict[str, Any]]:
    """The wire body of each phase the session issues, through one client built with ``**kw``: prefill, a
    tools-off generate (the turn, the captioner), a tools-on prefill and generate (step 1 / a tool step), a
    continuation (step k) and a constrained repair."""
    sent: list[dict[str, Any]] = []
    cl = make_client(serve(sent), **kw)
    phases: list[str] = []

    async def gen(name: str, prompt: Prompt, **kw: Any) -> None:
        phases.append(name)
        async for _ in cl.generate(prompt, f"r-{name}", **kw):
            pass

    async def pre(name: str, prompt: Prompt) -> None:
        phases.append(name)
        await cl.prefill(prompt, f"r-{name}")

    off = frame_prompt()
    on = frame_prompt(tools=S.TOOLS)
    try:
        await pre("prefill", off.prefix())
        await gen("generate", off, options=LlmOptions(max_tokens=512))
        await pre("prefill_tools", on.prefix())
        await gen("tool_step", on)
        await gen("continuation", S.continuation_prompt("I will click it."))
        await gen("repair", S.continuation_prompt("I will click it."), cd=CD)
    finally:
        await cl.aclose()
    assert len(sent) == len(phases), f"premise: one request per phase; {len(sent)} for {phases}"
    return dict(zip(phases, sent, strict=True))


@pytest.mark.parametrize("budget", [280, 560, 1120])
async def test_body_carries_kwarg_on_every_phase(budget: int) -> None:
    """A2 (mutations 1, 2): with a budget, the prefill, the tools-off generate, the tools-on prefill and step, the
    continuation and the CD repair all carry ``mm_processor_kwargs == {"max_soft_tokens": N}``; removing that key
    leaves exactly the body a budget-less client sends for the same call."""
    got = await every_phase(visual_tokens=budget)
    plain = await every_phase()
    for phase, body in got.items():
        assert body.get(KEY) == {"max_soft_tokens": budget}, (
            f"T-0131 A2: the {phase} body carries mm_processor_kwargs {{'max_soft_tokens': {budget}}}; "
            f"it has {body.get(KEY)!r}"
        )
        rest = {k: v for k, v in body.items() if k != KEY}
        assert rest == plain[phase], f"T-0131 A2: no other key of the {phase} body changes"
        assert KEY not in plain[phase], f"premise: a budget-less {phase} body has no {KEY}"


def test_request_body_carries_kwarg_pure() -> None:
    """A2: ``request_body`` itself (the pure builder every phase goes through) adds the key, on a tools-off, a
    tools-on and a continuation prompt, with options and xargs."""
    cl = make_client(visual_tokens=560)
    for prompt in (frame_prompt(), frame_prompt(tools=S.TOOLS), S.continuation_prompt("Sure.")):
        body = cl.request_body(prompt, "r1", LlmOptions(top_k=40), {"astra_tap": "1"})
        assert body[KEY] == {"max_soft_tokens": 560}
        base = VllmClient().request_body(prompt, "r1", LlmOptions(top_k=40), {"astra_tap": "1"})
        assert {k: v for k, v in body.items() if k != KEY} == base
    a = cl.request_body(frame_prompt(), "r1")
    a[KEY]["max_soft_tokens"] = 1
    assert cl.request_body(frame_prompt(), "r1")[KEY] == {"max_soft_tokens": 560}, (
        "a fresh, unshared dict every call (request_body's contract): mutating one body changes no later body"
    )


#: A BASE-pinned body, written out from the docs/02 §5 wire format and the golden row ``history_and_frame``
#: (``request_bodies_9669258.json``): no ``mm_processor_kwargs``, nothing else added.
UNSET_BODY: dict[str, Any] = {
    "model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
    "messages": [
        {"role": "system", "content": [{"type": "text", "text": "You are Astra."}]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _JPEG}},
                {"type": "text", "text": "Click the blue Add to cart."},
            ],
        },
    ],
    "stream": True,
    "chat_template_kwargs": {"enable_thinking": True},
    "vllm_xargs": {"astra_request_id": "r-unset"},
    "max_tokens": 512,
}


@pytest.mark.parametrize("how", ["explicit_none", "default"])
async def test_unset_body_byte_identical(how: str) -> None:
    """A2 (negative): ``visual_tokens=None`` (and the keyword left out) sends BASE's body: the dict literal above,
    key order included, and the golden battery's generate and prefill wire bytes."""
    kw: dict[str, Any] = {"visual_tokens": None} if how == "explicit_none" else {}
    cl = make_client(**kw)
    body = cl.request_body(frame_prompt(), "r-unset", LlmOptions(max_tokens=512))
    assert body == UNSET_BODY and list(body) == list(UNSET_BODY), body
    assert json.dumps(body) == json.dumps(UNSET_BODY)
    golden = {r["name"]: r for r in json.loads(S.GOLDEN.read_text())["rows"]}
    for name, prompt, options, xargs in S.golden_battery():
        sent: list[bytes] = []

        def handler(request: httpx.Request, sent: list[bytes] = sent) -> httpx.Response:
            sent.append(request.read())
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, text=PLAIN_SSE
            )

        wire = make_client(httpx.MockTransport(handler), **kw)
        async for _ in wire.generate(prompt, f"g-{name}", options, xargs=xargs):
            pass
        await wire.prefill(prompt.prefix(), f"p-{name}")
        await wire.aclose()
        assert sent[0].decode() == golden[name]["generate_wire"], name
        assert sent[1].decode() == golden[name]["prefill_wire"], name


async def test_not_in_xargs_or_options() -> None:
    """A2: the budget is a top-level body key only: never under ``vllm_xargs`` (no ``max_soft_tokens``, no
    ``mm_processor_kwargs`` key there, on any phase), never an ``LlmOptions`` field, and the body's key set is the
    budget-less one plus exactly ``mm_processor_kwargs``."""
    got = await every_phase(visual_tokens=1120)
    plain = await every_phase()
    for phase, body in got.items():
        xargs = body.get("vllm_xargs", {})
        assert not ({"max_soft_tokens", KEY, "visual_tokens"} & set(xargs)), (phase, xargs)
        assert set(body) == set(plain[phase]) | {KEY}, (
            phase,
            sorted(set(body) ^ set(plain[phase])),
        )
        assert "max_soft_tokens" not in body and "visual_tokens" not in body, phase
    assert not ({"max_soft_tokens", "visual_tokens", KEY} & set(LlmOptions.model_fields))


async def test_prefill_and_generate_agree() -> None:
    """A2 (adversarial, mutation 1 "the kwarg sent on generate only", and its mirror): one turn's prefill and its
    generate, through one client with a budget, carry identical ``mm_processor_kwargs``. The multimodal hash
    includes the processor kwargs, so a pair that disagreed would miss the prefix cache on the image."""
    sent: list[dict[str, Any]] = []
    cl = make_client(serve(sent), visual_tokens=560)
    prompt = frame_prompt()
    try:
        await cl.prefill(prompt.prefix(), "t1-prefill")
        async for _ in cl.generate(prompt, "t1-gen"):
            pass
    finally:
        await cl.aclose()
    prefill, gen = sent
    assert KEY in prefill and KEY in gen, (
        f"T-0131 A2: both carry the kwarg; prefill {prefill.get(KEY)!r}, generate {gen.get(KEY)!r}"
    )
    assert prefill[KEY] == gen[KEY] == {"max_soft_tokens": 560}
    assert (
        prefill["messages"][: len(prompt.prefix().messages)]
        == gen["messages"][: len(prompt.prefix().messages)]
    ), "premise: the pair shares the image prefix"
