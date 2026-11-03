"""What an over-long prompt looks like coming back, in the fake and in the real client (T-0020).

``tests/fakes/llm.py`` grew ``http_error(413)`` so a session test can script the engine rejecting
a prompt. Per ``.claude/rules/tests.md`` a behaviour added to a fake owes a test that the real
thing behaves the same way, so this module asserts the two sides of the seam T-0007 owes T-0020:

* the exception type, the status and the message are the same object shape on both sides — the
  fake builds it with ``httpx.Response.raise_for_status``, which is the call
  ``VllmClient.generate``/``prefill`` make, so it cannot drift;
* the status is *reachable* (``exc.response.status_code``) and also visible in ``str(exc)``, which
  is what the session flattens into ``LlmError.message`` and the HUD shows.

The ``vllm`` parametrisation drives the real ``VllmClient`` over ``httpx.MockTransport`` (the real
request builder, the real streaming code path, no server) as ``test_llm_backend_xargs.py`` does.
``test_the_live_engine_rejects_an_over_long_prompt`` is the same assertion against the service and
is marked ``gpu``: it is the one that can tell us *which* status vLLM 0.29.0 actually returns.
"""

from __future__ import annotations

import httpx
import pytest

from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
from local_astra.contracts.llm import LlmOptions, Message, Prompt, TextPart
from tests.fakes.llm import OVER_LONG_DETAIL, http_error

STATUSES = (400, 413)


def _prompt(text: str = "hello") -> Prompt:
    return Prompt(
        messages=(Message(role="user", content=(TextPart(text=text),)),),
        prefix_len=1,
    )


async def _from_vllm(status: int) -> httpx.HTTPStatusError:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"object": "error", "message": OVER_LONG_DETAIL, "code": status},
            request=request,
        )

    client = VllmClient(transport=httpx.MockTransport(handler))
    try:
        async for _ in client.generate(_prompt(), "req-1", LlmOptions()):
            raise AssertionError("an over-long prompt must not yield commits")
    except httpx.HTTPStatusError as exc:
        return exc
    finally:
        await client.aclose()
    raise AssertionError(f"HTTP {status} did not surface as an httpx.HTTPStatusError")


@pytest.mark.parametrize("status", STATUSES)
@pytest.mark.parametrize("side", ["fake", "vllm"])
async def test_an_over_long_prompt_surfaces_as_the_same_error_on_both_sides(side, status):
    """The fake and the real client raise the same thing, and the status survives.

    Without this, the session tests could be scripting a failure mode the engine never produces —
    the way an unmeasured 4-chars/token guess pretended to enforce a budget (T-0005 B10).
    """
    exc = http_error(status) if side == "fake" else await _from_vllm(status)

    assert isinstance(exc, httpx.HTTPStatusError), (
        f"{side}: an over-long prompt is an httpx.HTTPStatusError, not a generic exception"
    )
    assert exc.response.status_code == status, (
        f"{side}: the status must be reachable on the exception; got {exc.response.status_code}"
    )
    assert str(status) in str(exc), (
        f"{side}: the status must survive str(exc) — that is what reaches LlmError.message and "
        f"the HUD; got {str(exc)!r}"
    )
    assert exc.request.method == "POST"


@pytest.mark.gpu
async def test_the_live_engine_rejects_an_over_long_prompt_with_a_4xx():
    """The same assertion against the running service: which status is it, really?

    The fake claims 413 and the session treats 400 and 413 alike; this is the test that says what
    vLLM 0.29.0 on the pinned container actually returns for a prompt past ``--max-model-len``.
    Record the answer in ``.claude/rules/gpu-services.md`` when it first runs.

    T-0020 ruling A11: the first ``/launch`` run of this row proved the status is *in* {400, 413}
    but never printed *which*, so the rule file still cannot record it. The status is therefore
    printed (visible under ``-rA``) — recorded, not gated: A4 stands, the session treats both
    alike, and this assertion stays the membership test. ``print`` and not ``record_property``:
    this repo's ``junit_family`` is ``xunit2``, under which ``record_property`` records nothing and
    only warns (measured on the T-0007a census run f114adf).
    """
    client = VllmClient(VllmEndpoint())
    long_prompt = _prompt("word " * 200_000)
    try:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            async for _ in client.generate(long_prompt, "over-long-probe", LlmOptions()):
                pass
    except httpx.ConnectError:
        pytest.skip("vLLM is not running on 127.0.0.1:8001")
    finally:
        await client.aclose()
    status = caught.value.response.status_code
    print(f"over-long status: {status}")
    print(f"over-long message: {str(caught.value)[:200]}")
    assert status in STATUSES, f"an over-long prompt must be a 400/413, not {status}"
