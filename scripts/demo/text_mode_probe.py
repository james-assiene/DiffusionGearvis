#!/usr/bin/env python3
"""One typed turn through the live conductor, the way the HUD's text box does it (T-0039).

Usage (the session must already be up, e.g. ``uv run astra session --silent --serve-hud``):

    uv run python scripts/demo/text_mode_probe.py "what is the capital of france" \
        --ws ws://127.0.0.1:8000/v1/realtime --timeout-s 120

It says ``astra.hello``, sends one ``astra.text`` and prints every server message until the turn
returns to IDLE (or the timeout), then the transcript and the spoken lines. It touches no device
and writes nothing under ``artifacts/``; it is a demo script (ruling D5) and spends GPU time only
through the session it talks to, which is why the run itself goes through ``/launch``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import websockets

from local_astra.contracts.codec import decode, encode
from local_astra.contracts.protocol import (
    AstraError,
    AstraHello,
    AstraSpoken,
    AstraText,
    AstraTranscript,
    AstraTurnState,
)


async def run(question: str, ws_url: str, timeout_s: float) -> int:
    started = time.monotonic()
    spoken: list[str] = []
    transcript: str | None = None
    saw_idle_after_turn = False
    async with websockets.connect(ws_url, origin="http://127.0.0.1:1420", max_size=None) as ws:
        await ws.send(encode(AstraHello()))
        # wait for the hello reply's turn state before typing
        while True:
            m = decode(await asyncio.wait_for(ws.recv(), timeout_s))
            print(f"[{time.monotonic() - started:6.2f}s] <- {getattr(m, 'type', type(m).__name__)}")
            if isinstance(m, AstraTurnState):
                if m.state != "IDLE":
                    print(f"session is {m.state}, not IDLE; try again later")
                    return 2
                break
        await ws.send(encode(AstraText(text=question)))
        print(f"[{time.monotonic() - started:6.2f}s] -> astra.text {question!r}")
        turn_opened = False
        while time.monotonic() - started < timeout_s:
            try:
                raw = await asyncio.wait_for(ws.recv(), 5.0)
            except TimeoutError:
                continue
            m = decode(raw)
            kind = getattr(m, "type", type(m).__name__)
            extra = ""
            if isinstance(m, AstraTurnState):
                extra = f" state={m.state} turn={m.turn_id}"
                if m.state != "IDLE":
                    turn_opened = True
                elif turn_opened:
                    saw_idle_after_turn = True
            elif isinstance(m, AstraTranscript):
                extra = f" final={m.final} text={m.text!r}"
                if m.final:
                    transcript = m.text
            elif isinstance(m, AstraSpoken):
                extra = f" {m.utterance_id} text={m.text!r}"
                spoken.append(m.text)
            elif isinstance(m, AstraError):
                extra = f" code={m.code} {m.message!r}"
            print(f"[{time.monotonic() - started:6.2f}s] <- {kind}{extra}")
            if saw_idle_after_turn:
                break
    print()
    print(f"transcript: {transcript!r}")
    print(f"spoken    : {' '.join(spoken)!r}")
    print(f"turn completed: {saw_idle_after_turn}  wall {time.monotonic() - started:.1f}s")
    return 0 if saw_idle_after_turn and spoken else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("question")
    p.add_argument("--ws", default="ws://127.0.0.1:8000/v1/realtime")
    p.add_argument("--timeout-s", type=float, default=120.0)
    a = p.parse_args(argv)
    return asyncio.run(run(a.question, a.ws, a.timeout_s))


if __name__ == "__main__":
    sys.exit(main())
