"""T-0088 criterion 4: ``LlmBackend.tokenize(text) -> list[int]`` (H9: the engine's ``/tokenize`` on loopback).

``VllmClient`` POSTs ``{model, prompt: text, add_special_tokens: false}`` to ``/tokenize`` at the server **root**
(the base URL without ``/v1``; T-0087 q10 measured it there) and returns ``tokens``. The real side here is
``VllmClient`` over ``httpx.MockTransport``; gpu row 4 of ``test_llm_tools_wire_gpu.py`` checks the live engine
against q10's ids.

Both backends: deterministic, and each special literal is one id. The ``vllm`` mock server answers with the
fake's map (``tests.fakes.llm.fake_tokenize``), so the shared rows test the client's plumbing; the client-only
rows test the URL, the body and the refusals.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import httpx
import pytest

from local_astra.conductor.llm_client import VllmClient
from local_astra.toolcall.special_tokens import SPECIAL_IDS
from tests.fakes import FakeLlm
from tests.fakes.llm import fake_tokenize
from tests.unit import _llm_id_support as S

pytestmark = pytest.mark.contract

MODEL = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"


def _server(seen: list[httpx.Request], reply: Any = None, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path != "/tokenize":
            raise AssertionError(
                f"tokenize must hit /tokenize at the server root; got {request.url}"
            )
        if status != 200:
            return httpx.Response(
                status, json={"object": "error", "message": "boom", "code": status}
            )
        body = json.loads(request.read())
        tokens = fake_tokenize(body["prompt"]) if reply is None else reply
        payload = (
            tokens
            if isinstance(tokens, dict)
            else {"count": 0, "max_model_len": 16384, "tokens": tokens}
        )
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


async def _tokenize(backend: Any, text: str) -> Any:
    fn = S.need(backend, "tokenize", "criterion 4")
    assert inspect.iscoroutinefunction(fn), (
        f"ruling E5: {type(backend).__name__}.tokenize must be `async def`"
    )
    return await fn(text)


@pytest.fixture(params=["fake", "vllm"])
def backend(request: pytest.FixtureRequest) -> Any:
    if request.param == "fake":
        return FakeLlm()
    return VllmClient(transport=_server([]))


def test_tokenize_is_async_on_the_protocol_and_both_backends() -> None:
    """Ruling E5: ``LlmBackend.tokenize`` is ``async``, like ``generate``; a sync one on either side is refused."""
    from local_astra.contracts.protocols import LlmBackend

    for owner in (LlmBackend, FakeLlm, VllmClient):
        fn = S.need(owner, "tokenize", "criterion 4")
        assert inspect.iscoroutinefunction(fn), (
            f"ruling E5: {owner.__name__}.tokenize must be `async def`"
        )
        assert list(inspect.signature(fn).parameters) == ["self", "text"], owner.__name__


async def test_tokenize_is_deterministic(backend: Any) -> None:
    a = await _tokenize(backend, 'I will.<|tool_call>call:x{q:<|"|>a<|"|>}')
    b = await _tokenize(backend, 'I will.<|tool_call>call:x{q:<|"|>a<|"|>}')
    assert a == b and all(type(t) is int for t in a)


@pytest.mark.parametrize("lit", list(SPECIAL_IDS.values()))
async def test_each_special_literal_is_one_id(backend: Any, lit: str) -> None:
    assert await _tokenize(backend, lit) == [{v: k for k, v in SPECIAL_IDS.items()}[lit]]


async def test_the_real_client_posts_to_the_root_tokenize_without_special_tokens() -> None:
    seen: list[httpx.Request] = []
    cl = VllmClient(transport=_server(seen, reply=[7, 8, 9]))
    got = await _tokenize(cl, "hello")
    await cl.aclose()
    assert got == [7, 8, 9], "the ids are the server's: nothing is tokenised locally"
    (req,) = seen
    assert req.method == "POST"
    assert str(req.url) == "http://127.0.0.1:8001/tokenize"
    assert json.loads(req.read()) == {
        "model": MODEL,
        "prompt": "hello",
        "add_special_tokens": False,
    }


async def test_the_real_client_raises_on_a_500() -> None:
    cl = VllmClient(transport=_server([], status=500))
    S.need(cl, "tokenize", "criterion 4")
    with pytest.raises(Exception):  # noqa: B017 - criterion 4 says "raises", not which class
        await _tokenize(cl, "hello")
    await cl.aclose()


async def test_the_real_client_raises_on_a_500_even_with_a_well_formed_body_e13() -> None:
    """Ruling E13: the status alone refuses the reply. A 500 whose body is a valid ``{"tokens": [1, 2]}`` must
    still raise, so the row above cannot pass merely because the error body is malformed."""

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(500, json={"tokens": [1, 2]})

    cl = VllmClient(transport=httpx.MockTransport(handler))
    with pytest.raises(Exception) as info:
        await _tokenize(cl, "hello")
    await cl.aclose()
    assert not isinstance(info.value, AssertionError), info.value
    assert seen == ["/tokenize"], "the request reached the server and was refused on its status"


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param({"count": 1}, id="no-tokens"),
        pytest.param({"tokens": "1 2"}, id="a-string"),
        pytest.param({"tokens": [1, "2"]}, id="a-str-inside"),
        pytest.param({"tokens": [1, 2.0]}, id="a-float-inside"),
        pytest.param({"tokens": None}, id="null"),
        pytest.param({"tokens": [True, 2]}, id="a-bool-inside"),
    ],
)
async def test_the_real_client_raises_on_a_malformed_body(reply: dict[str, Any]) -> None:
    cl = VllmClient(transport=_server([], reply=reply))
    S.need(cl, "tokenize", "criterion 4")
    with pytest.raises(Exception):  # noqa: B017
        await _tokenize(cl, "hello")
    await cl.aclose()


async def test_the_fakes_ids_decode_back_to_the_text() -> None:
    """The fake's map is honest: the pinned decoder (T-0088a subset) gives the text back."""
    text = 'Sí, I will.<|tool_call>call:x{q:<|"|>€<|"|>}<tool_call|>'
    assert S.decode(await FakeLlm().tokenize(text)) == text
