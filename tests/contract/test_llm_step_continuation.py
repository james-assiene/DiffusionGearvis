"""T-0114: a tool-loop step request on the wire (contract, ``fake``/``real``; ``real`` is ``gpu`` and runs in T-0115's
``/launch``).

The step body is built through the real ``VllmClient.request_body`` (the fake: a ``Wire`` transport that records the
body as sent; the real: a recording transport in front of the live engine). In both: ``continue_final_message`` is
true, ``add_generation_prompt`` false, and ``tools``/``tool_choice``/``skip_special_tokens``/``return_token_ids``/
``chat_template_kwargs`` and the leading messages are step 1's. The id reader starts in the normal state (the text
ends in ``<tool_response|>``, not in the call tag): on the fake, a scripted answer is an ``answer`` commit, while the
same ids after a repair-shaped prompt become a ``tool`` commit (the discriminating twin). The fake LLM's step routing
(``FakeLlm.script_steps``) is pinned beside it: a step prompt takes the step script, a repair prompt the
continuation script, and the step body is the real client's.

Regime (real): lean profile, vLLM v0.29.0, NVFP4 rev ``2ea8372``, template ``36e3a42e…``, tools_v1 ``bbfc53cc…``,
CL 256, 48 steps, thinking on, n = 1, unseeded. The real model's continuation is not predictable, so the real row
asserts the wire and that the stream ends, not the text.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_astra.conductor.llm_client import VllmClient
from local_astra.contracts.llm import Commit, Message, Prompt, TextPart
from local_astra.contracts.toolcall import SearchResult, SearchResults
from local_astra.toolcall import loop
from tests.fakes import vllm_sse as V
from tests.fakes.llm import FakeLlm, is_step_prompt
from tests.unit import _llm_id_support as S

pytestmark = pytest.mark.contract

QUERY = "weather in Paris"
STEP1_KEYS = (
    "tools",
    "tool_choice",
    "skip_special_tokens",
    "return_token_ids",
    "chat_template_kwargs",
)


def _refusing() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"request_body only builds; it tried to send {request.url}")

    return httpx.MockTransport(handler)


def _prompts() -> tuple[Prompt, Prompt]:
    step1 = S.tools_prompt()
    results = SearchResults(
        query=QUERY,
        results=(
            SearchResult(title="Paris weather", url="https://w.example.org/p", snippet="14C"),
        ),
    )
    step = loop.step_prompt(
        step1,
        [
            loop.LoopStep(
                "Let me look that up.",
                "google_search",
                {"query": QUERY},
                loop.search_payload(results),
            )
        ],
    )
    return step1, step


def _check_body(body: dict[str, Any], step1_body: dict[str, Any], step: Prompt) -> None:
    assert body.get("continue_final_message") is True, body.get("continue_final_message")
    assert body.get("add_generation_prompt") is False, body.get("add_generation_prompt")
    for key in STEP1_KEYS:
        assert body.get(key) == step1_body.get(key), f"T-0114: {key} differs from step 1's"
    assert body["messages"][:-1] == step1_body["messages"], "the leading messages are step 1's"
    last = body["messages"][-1]
    assert last["role"] == "assistant", last
    text = last["content"] if isinstance(last["content"], str) else last["content"][0]["text"]
    assert text == step.messages[-1].text() and text.endswith("<tool_response|>"), text[-80:]


async def _fake(step1: Prompt, step: Prompt) -> None:
    client = VllmClient(transport=_refusing())
    step1_body = client.request_body(step1, "r1", None, None)
    got, wire = await S.run_chunks([[*S.prose("It is 14C."), S.TURN_CLOSE]], step)
    assert len(wire.bodies) == 1
    _check_body(wire.bodies[0], step1_body, step)
    assert S.texts(got, "answer") == "It is 14C." and S.tool_commits(got) == [], (
        "T-0114: the id reader starts in the normal state for a step continuation"
    )
    # the twin: the same continuation shape ending in the call tag starts in the tool state
    repair = Prompt(
        **{
            **{f: getattr(step, f) for f in Prompt.model_fields},
            "messages": (
                *step.messages[:-1],
                Message(role="assistant", content=(TextPart(text="Let me look.<|tool_call>"),)),
            ),
        }
    )
    got, _ = await S.run_chunks(
        [[*S.prose("call:click_at{x:1,y:2}"), S.TOOL_CALL_CLOSE, S.TOOL_RESPONSE]], repair
    )
    assert S.texts(got, "answer") == "" and len(S.tool_commits(got)) == 1

    # FakeLlm's step routing (tests/fakes/llm.py): structural, step script vs continuation script
    assert is_step_prompt(step) and not is_step_prompt(repair) and not is_step_prompt(step1)
    fake = FakeLlm([Commit(request_id="?", canvas_idx=0, segment="answer", text="pass 1")])
    fake.script_steps([Commit(request_id="?", canvas_idx=0, segment="answer", text="step 2")])
    fake.script_continuation(FakeLlm.tool_commit("<|tool_call>call:click_at{x:1,y:2}<tool_call|>"))
    stepped = [c.text async for c in fake.generate(step, "s2")]
    repaired = [c.segment async for c in fake.generate(repair, "rep")]
    assert stepped == ["step 2"] and repaired == ["tool"], (stepped, repaired)
    assert fake.step_prompts == [("s2", step)] and len(fake.step_bodies) == 1
    assert fake.step_bodies[0] == client.request_body(step, "s2", None, None) | {
        k: v for k, v in fake.step_bodies[0].items() if k == "vllm_xargs"
    }
    assert fake.step_overruns == []


class _Recording(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.inner = httpx.AsyncHTTPTransport()
        self.bodies: list[dict[str, Any]] = []
        self.statuses: list[int] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        resp = await self.inner.handle_async_request(request)
        self.statuses.append(resp.status_code)
        return resp

    async def aclose(self) -> None:
        await self.inner.aclose()


async def _real(step1: Prompt, step: Prompt) -> None:
    from local_astra.toolcall.detok import HostDetokenizer

    rec = _Recording()
    client = VllmClient(transport=rec, detokenizer=HostDetokenizer.load())
    try:
        try:
            got = [c async for c in client.generate(step, "t0114-step-contract")]
        except httpx.ConnectError:
            pytest.skip("the vLLM engine is not up on loopback (T-0115 /launch runs this row)")
    finally:
        await client.aclose()
    step1_body = VllmClient(transport=_refusing()).request_body(step1, "r1", None, None)
    assert rec.statuses == [200], rec.statuses
    _check_body(rec.bodies[0], step1_body, step)
    assert got, "the engine answered the step continuation with no commit"
    print(
        "T-0114 gpu regime [step continuation]: lean vLLM0.29.0 rev2ea8372 CL256 steps48 thinking=on n=1 "
        f"seed=none commits={[(c.segment, len(c.text)) for c in got]}"
    )


async def test_step_continuation_wire(backend):
    step1, step = _prompts()
    if backend == "fake":
        await _fake(step1, step)
    else:
        await _real(step1, step)


assert V  # the fake SSE module is the fake backend (through S.run_chunks)
