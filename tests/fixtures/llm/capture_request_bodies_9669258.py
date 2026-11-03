"""Capture ``request_bodies_9669258.json`` (T-0088 criterion 2's negative): main's tools-off request bodies.

Run once, on a clean checkout of the T-0088 base ``9669258`` (main with T-0088a merged), before any T-0088
implementation exists::

    uv run python tests/fixtures/llm/capture_request_bodies_9669258.py

For every prompt of ``tests/unit/_llm_id_support.golden_battery()`` it records ``json.dumps(request_body(...))``
(key order included) and the exact bytes ``generate`` and ``prefill`` put on the wire over ``httpx.MockTransport``.
Never imported by a test.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import httpx

from local_astra.conductor.llm_client import VllmClient
from tests.unit._llm_id_support import BASE_SHA, GOLDEN, golden_battery

SSE = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'


async def _wire(name: str, prompt, options, xargs) -> tuple[str, str]:
    sent: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.read())
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=SSE)

    client = VllmClient(transport=httpx.MockTransport(handler))
    async for _ in client.generate(prompt, f"g-{name}", options, xargs=xargs):
        pass
    await client.prefill(prompt.prefix(), f"p-{name}")
    await client.aclose()
    return sent[0].decode(), sent[1].decode()


def main() -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    # local_astra/ (committed and working tree) must be byte-identical to the base: tests-only commits may sit on top
    diff = subprocess.run(
        ["git", "diff", "--name-only", BASE_SHA, "--", "local_astra"],
        capture_output=True,
        text=True,
    )
    assert diff.returncode == 0 and not diff.stdout.strip(), (
        f"local_astra differs from {BASE_SHA}: {diff.stdout}"
    )
    out = {"base_sha": BASE_SHA, "captured_at_head": head, "rows": []}
    for name, prompt, options, xargs in golden_battery():
        body = VllmClient().request_body(prompt, f"b-{name}", options, xargs)
        gen, pre = asyncio.run(_wire(name, prompt, options, xargs))
        out["rows"].append(
            {
                "name": name,
                "request_body": json.dumps(body),
                "generate_wire": gen,
                "prefill_wire": pre,
            }
        )
    Path(GOLDEN).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {GOLDEN} ({len(out['rows'])} rows at {head[:9]})")


if __name__ == "__main__":
    main()
