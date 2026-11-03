"""T-0171: the fake's engine-empty error is the real client's, and the wire fake's role chunk is the recorded one.

``tests/fakes/llm.engine_empty_error`` builds the exception ``FakeLlm`` raises for the engine's empty "retryable"
response by running the real ``VllmClient`` against ``tests/fakes/vllm_sse.engine_empty``. These rows pin that it
cannot drift from what the client raises for the response as vLLM 0.29.0 streams it, written out here by hand from
the source (``vllm_sse``'s docstring cites the lines): the first-iteration role chunk, then ``data: [DONE]``.

Two sides, both through the real client: ``fake`` is ``engine_empty_error`` (the wire fake), ``vllm`` is a
``MockTransport`` serving ``HAND_BODY``. The drift miss has never been recorded on the wire; the GPU repro is H-T171-1
(task file, follow-ups), so there is no ``gpu`` side yet. The role chunk itself *is* recorded: the second row compares
the wire fake's with the first line of ``tests/fixtures/llm/t0095_row2_live.jsonl`` (a live tools-on stream).

Regime: CPU, httpx ``MockTransport``, no engine; n = 1 per row, deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from local_astra.conductor import llm_client
from tests.fakes import vllm_sse as V
from tests.fakes.llm import engine_empty_error, engine_empty_prompt
from tests.unit import _llm_id_support as S

pytestmark = pytest.mark.contract

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "t0095_row2_live.jsonl"
#: per-response values of the role chunk (the stream id, the creation time, the prompt's ids)
VARIABLE = ("id", "created", "prompt_token_ids")

#: the drift miss as the chat server streams it, written by hand: the first-iteration role chunk, then [DONE]
HAND_BODY = (
    'data: {"id":"chatcmpl-hand","object":"chat.completion.chunk","created":1,'
    '"model":"nvidia/diffusiongemma-26B-A4B-it-NVFP4","choices":[{"index":0,'
    '"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],'
    '"prompt_token_ids":[2,105,9731],"prompt_text":null}\n\n'
    "data: [DONE]\n\n"
)


async def _from_vllm(attempt_id: str) -> BaseException:
    """What the real client raises for ``HAND_BODY`` (id mode: the tools-on image prompt the fake uses)."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions"), request.url
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=HAND_BODY)

    cl = S.client(httpx.MockTransport(handler))
    try:
        async for c in cl.generate(engine_empty_prompt(), attempt_id):
            raise AssertionError(f"an empty response yielded {c!r}")
    except AssertionError:
        raise
    except Exception as exc:
        return exc
    finally:
        await cl.aclose()
    raise AssertionError("the real client ended the empty response without raising")


@pytest.mark.parametrize("side", ["fake", "vllm"])
async def test_engine_empty_error_is_the_same_on_both_sides(side: str) -> None:
    """Criterion 1 on each side: ``LlmEngineEmpty`` exactly, naming the attempt and saying ``retryable``; the message
    depends on the attempt id only; and the fake's error equals the real client's (type and message)."""
    attempt = "sess01-t1-gen-2"
    exc = engine_empty_error(attempt) if side == "fake" else await _from_vllm(attempt)
    cls = getattr(llm_client, "LlmEngineEmpty", None)
    assert isinstance(cls, type), "T-0171 criterion 1: llm_client.LlmEngineEmpty does not exist yet"
    assert type(exc) is cls, f"{side}: expected LlmEngineEmpty, got {type(exc).__name__}: {exc!r}"
    assert attempt in str(exc) and "retryable" in str(exc), f"{side}: {str(exc)!r}"
    other = engine_empty_error("other-9") if side == "fake" else await _from_vllm("other-9")
    assert str(other).replace("other-9", attempt) == str(exc), (
        f"{side}: the message varies only by the attempt id; {str(other)!r} vs {str(exc)!r}"
    )
    fake, real = engine_empty_error(attempt), await _from_vllm(attempt)
    assert (type(fake), str(fake)) == (type(real), str(real)), "fake and real client agree"


def test_engine_empty_role_chunk_is_the_recorded_role_chunk() -> None:
    """The wire fake's role chunk is the live one (keys, key order, values but the per-response ones), and the
    ``with_role=False`` variant is ``[DONE]`` alone."""
    first = FIXTURE.read_text().splitlines()[0]
    live = json.loads(first[len("data: ") :])
    assert "token_ids" not in live["choices"][0], (
        "premise: the recorded first line is the role chunk"
    )
    built = json.loads(V.engine_empty()[0][len("data: ") :])
    assert {k: v for k, v in built.items() if k not in VARIABLE} == {
        k: v for k, v in live.items() if k not in VARIABLE
    }
    assert list(built) == list(live), "same keys, same order"
    assert V.engine_empty(with_role=False) == [V.DONE_LINE]
    assert V.engine_empty()[-1] == V.DONE_LINE
