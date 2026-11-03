"""Builds ``requests.jsonl``: a synthetic session log shaped like seq 16202 and 3957 (T-0201 B9). No real utterance.

Run: ``uv run python tests/fixtures/t0201/build_requests.py > tests/fixtures/t0201/requests.jsonl``.
Uses the real ``loop.step_prompt``, ``session.log_dump`` and ``session_log.redact`` so the logged shape is the
session's; the label texts are typed by hand (T-0134 A3, T-0136 B-3, T-0134 A4, T-0177 A7).
"""

from __future__ import annotations

import base64
import json
import sys
from typing import Any

from local_astra.conductor.session import log_dump
from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
from local_astra.telemetry.session_log import redact
from local_astra.toolcall import loop

L1 = "[This screenshot is the browser screen now.]"
LU = "[This screenshot is the browser screen now, after your last action.]"
LN = "[This screenshot is the browser screen now; your last action did not change it.]"
LU_OUTCOME = (
    "[This screenshot is the browser screen now, after your last action. Compared with the previous screenshot: "
    "changed elsewhere (top right).]"
)
LU_EXTENT = (
    "[This screenshot is the browser screen now, after your last action. Page beyond this screen: about 2.0 screens "
    "below, nothing above.]"
)
SHARED_SCREEN = "[This screenshot is the user's shared screen now, not your browser; click_at and type_at cannot act on it.]"
SHARED_CAMERA = "[This image is the user's camera now, not your browser; click_at and type_at cannot act on it.]"
TOOLS = [{"type": "function", "function": {"name": "scroll", "parameters": {"type": "object"}}}]


def image(tag: str) -> ImagePart:
    data = base64.b64encode(b"\xff\xd8\xff\xe0" + tag.encode() * 40).decode()
    return ImagePart(
        url="data:image/jpeg;base64," + data, phash="capture:" + tag.encode().hex()[:32]
    )


def base(label: str | None, tag: str, question: str) -> Prompt:
    parts: list[Any] = [image(tag)]
    if label is not None:
        parts.append(TextPart(text=label))
    parts.append(TextPart(text=question))
    return Prompt(
        messages=(
            Message(role="system", content=(TextPart(text="You are a synthetic test assistant."),)),
            Message(role="user", content=tuple(parts)),
        ),
        prefix_len=2,
        thinking=True,
        tools=TOOLS,
    )


def steps(n: int) -> list[loop.LoopStep]:
    out = [
        loop.LoopStep(
            text="I will open the example site.",
            name="type_at",
            arguments={"press_enter": True, "text": "example.org", "x": 500, "y": 40},
            response={
                "action": "clicked at (640, 32) and typed 11 chars",
                "screenshot": "updated",
                "status": "executed",
            },
        )
    ]
    for _ in range(n - 1):
        out.append(
            loop.LoopStep(
                text="",
                name="scroll",
                arguments={"direction": "down"},
                response={
                    "action": "scrolled down by 600 px",
                    "screenshot": "updated",
                    "status": "executed",
                },
            )
        )
    return out


def record(
    seq: int, step: int, label: str | None, *, phase: str = "tool_step", report: bool = False
) -> dict[str, Any]:
    p = base(label, f"t0201-{seq}", "read the first pages of the synthetic comic")
    if phase == "tool_step":
        p = loop.step_prompt(p, steps(step - 1))
        if report:
            p = loop.with_report_fact(p)
    data: dict[str, Any] = {
        "phase": phase,
        "step": step,
        "request_id": f"t1-gen-s{step}" if step > 1 else "t1-gen",
        "attempt_id": f"abc-t1-gen-s{step}-{seq}",
        "options": {"top_k": None, "top_p": None, "max_tokens": None, "stop": []},
        "prompt": log_dump(p),
    }
    if report:
        data["report_fact"] = True
    if label is not None:
        data["live_label"] = {
            "text": label,
            "above_px": None,
            "below_px": None,
            "viewport_px": None,
        }
    return {
        "seq": seq,
        "t_ms": seq,
        "wall": "2026-10-29T00:00:00+00:00",
        "session_ms": seq,
        "kind": "llm.request",
        "turn_id": "t1",
        "data": redact(data),
    }


ROWS = [
    record(10, 1, L1, phase="generate"),
    record(11, 7, LU),  # seq 16202's shape: type_at then scrolls, labelled updated
    record(12, 15, LU),  # seq 3957's shape: a long scroll run
    record(13, 3, LN),
    record(14, 2, L1),
    record(15, 4, LU_OUTCOME),
    record(16, 5, LU_EXTENT),
    record(17, 2, SHARED_SCREEN),
    record(18, 3, SHARED_CAMERA),
    record(19, 4, None),  # not labelled: no row
    record(20, 9, LU, report=True),  # the report step: no row (B4)
]

if __name__ == "__main__":
    sys.stdout.write(
        json.dumps(
            {
                "seq": 0,
                "t_ms": 0,
                "wall": "2026-10-29T00:00:00+00:00",
                "kind": "log.started",
                "turn_id": None,
                "data": {"pid": 1},
            }
        )
        + "\n"
    )
    for r in ROWS:
        sys.stdout.write(json.dumps(r) + "\n")
