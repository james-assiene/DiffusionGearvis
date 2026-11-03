"""ReGround, the pure half (T-0131 group B): the crop, the mapping back, the outcome and the pass-2 prompt.

ReGround is cua-diffusion-gemma's two-pass crop, ported. Pass 1 is the call the model made on the full frame; pass
2 asks the same prompt again with the call's frame replaced by a crop around pass 1's point, and only pass 2's x and
y are kept (R-b: never its tool name or its text). The session (``conductor/session.py``) runs the request; this
module only decides, so it has no clock, no I/O and no asyncio. PIL is imported inside ``crop_jpeg`` and nowhere
else, so importing this module loads no image library.

Coordinates are the model's 0-999 grid; ``tool_executor.to_pixels`` is the executor's own mapping onto pixels, and
B4 is stated in it: the full-frame pixel of the mapped value is within 1 px of ``x0 + to_pixels(x2, w)`` for frames
up to 1999 px a side (one 0-999 step is then at most 2 px).
"""

from __future__ import annotations

import base64
import functools
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from local_astra.conductor.tool_executor import to_pixels
from local_astra.contracts.llm import ImagePart, Prompt
from local_astra.toolcall import native, score
from local_astra.toolcall import tools as _tools

#: B2: the crop is this fraction of the frame on each side (512x320 at 1280x800)
ZOOM = 0.4

#: B3: every outcome a ``tool.reground`` record can carry, in the table's order
OUTCOMES: tuple[str, ...] = (
    "ok",
    "pass2_unparseable",
    "pass2_no_point",
    "pass2_other_tool",
    "pass2_timeout",
    "error",
    "skipped_no_point",
)

#: B1: the coordinate tools; every other tool is ``skipped_no_point``
COORDINATE_TOOLS = frozenset({"click_at", "type_at"})

#: The stop texts the client may leave at the end of a ``tool`` commit (``live._STOP_TEXTS``: ``<eos>``,
#: ``<|tool_response>``, ``<turn|>``), dropped before parsing as ``live.classify`` does. A local copy, because this
#: module imports only the contracts, ``native``, ``score``, ``tools`` and ``tool_executor``.
_STOP_TEXTS = ("<eos>", "<|tool_response>", "<turn|>")

_COORD_MAX = 999

Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class Decision:
    """What pass 2 decided: ``outcome`` (one of the four parse outcomes of B3), ``arguments`` (what executes: pass
    1's, with only x and y replaced on ``ok``) and ``point`` (pass 2's own 0-999 point in the crop, for the log;
    ``None`` unless pass 2 named a coordinate tool)."""

    outcome: str
    arguments: dict[str, Any]
    point: tuple[int, int] | None = None


def applies(name: str) -> bool:
    """B1: only ``click_at`` and ``type_at`` are regrounded."""
    return name in COORDINATE_TOOLS


def crop_rect(frame_w: int, frame_h: int, px: int, py: int) -> Rect:
    """B2: ``(x0, y0, w, h)``, ``ZOOM`` of the frame on each side, centred on pass 1's pixel point ``(px, py)``
    and clamped inside the frame."""
    w = max(1, min(frame_w, int(ZOOM * frame_w + 0.5)))
    h = max(1, min(frame_h, int(ZOOM * frame_h + 0.5)))
    x0 = max(0, min(frame_w - w, px - w // 2))
    y0 = max(0, min(frame_h - h, py - h // 2))
    return x0, y0, w, h


def _to_grid(target_px: int, size: int) -> int:
    """The 0-999 value whose ``to_pixels`` is nearest ``target_px`` (the nearest of the three around the exact
    inverse, so the result never depends on a float)."""
    if size <= 1:
        return 0
    guess = (2 * target_px * _COORD_MAX + (size - 1)) // (2 * (size - 1))
    candidates = [max(0, min(_COORD_MAX, guess + d)) for d in (0, -1, 1)]
    return min(candidates, key=lambda v: abs(to_pixels(v, size) - target_px))


def map_point(x2: int, y2: int, rect: Rect, frame_w: int, frame_h: int) -> tuple[int, int]:
    """B4: pass 2's 0-999 point in the crop ``rect``, mapped back to the full frame's 0-999."""
    x0, y0, w, h = rect
    return (
        _to_grid(x0 + to_pixels(x2, w), frame_w),
        _to_grid(y0 + to_pixels(y2, h), frame_h),
    )


@functools.cache
def _toolset() -> _tools.ToolSet:
    return _tools.load()


def _parse(tool_text: str | None) -> Mapping[str, Any] | None:
    """Pass 2's ``tool`` commit as ``{"name", "arguments"}``, or ``None`` when it is absent, does not parse
    (``native.parse_call``) or is not schema-valid (``score.schema_valid``)."""
    if tool_text is None:
        return None
    body = tool_text.lstrip()
    if body.startswith(native.TOOL_CALL_OPEN):
        body = body[len(native.TOOL_CALL_OPEN) :]
    for stop in _STOP_TEXTS:
        if body.endswith(stop):
            body = body[: -len(stop)]
            break
    parsed = native.parse_call(body)
    if isinstance(parsed, score.ParseFailure):
        return None
    if not score.schema_valid(parsed, _toolset()):
        return None
    return parsed


def decide(
    name: str,
    arguments: Mapping[str, Any],
    pass2_tool: str | None,
    rect: Rect,
    frame_w: int,
    frame_h: int,
) -> Decision:
    """B3's parse outcomes for pass 1's call ``name(arguments)`` and pass 2's ``tool`` commit text.

    ``ok``: pass 2 names the same coordinate tool, so pass 1's call executes with only x and y replaced by pass
    2's point mapped back (B4). Otherwise pass 1's arguments execute unchanged: ``pass2_unparseable`` (no call or an
    invalid one), ``pass2_no_point`` (a coordinate-free tool: observed, never executed, R-b) or
    ``pass2_other_tool`` (the other coordinate tool)."""
    pass1 = dict(arguments)
    parsed = _parse(pass2_tool)
    if parsed is None:
        return Decision("pass2_unparseable", pass1)
    other = str(parsed["name"])
    if not applies(other):
        return Decision("pass2_no_point", pass1)
    args2 = parsed["arguments"]
    point = (int(args2["x"]), int(args2["y"]))
    if other != name:
        return Decision("pass2_other_tool", pass1, point)
    x, y = map_point(point[0], point[1], rect, frame_w, frame_h)
    pass1["x"], pass1["y"] = x, y
    return Decision("ok", pass1, point)


def replace_frame_image(prompt: Prompt, identity: str, crop_jpeg: bytes) -> Prompt:
    """B2 / R-a: ``prompt`` with every image part whose ``phash`` is ``identity`` (the call's frame) replaced by
    the crop; every other part and field unchanged. The crop's part carries its own ``crop:`` identity, which no
    frame of the session has."""
    url = "data:image/jpeg;base64," + base64.b64encode(crop_jpeg).decode("ascii")
    crop = ImagePart(url=url, phash="crop:" + hashlib.sha256(crop_jpeg).hexdigest()[:32])
    changed = False
    messages = []
    for message in prompt.messages:
        content = tuple(
            crop if isinstance(part, ImagePart) and part.phash == identity else part
            for part in message.content
        )
        if content != message.content:
            changed = True
            message = message.model_copy(update={"content": content})
        messages.append(message)
    if not changed:
        return prompt
    return prompt.model_copy(update={"messages": tuple(messages)})


def crop_jpeg(frame_jpeg: bytes, rect: Rect) -> bytes:
    """B2: the region ``rect`` of the frame's own JPEG bytes, as a JPEG at its native size."""
    import io

    from PIL import Image

    x0, y0, w, h = rect
    with Image.open(io.BytesIO(frame_jpeg)) as img:
        region = img.convert("RGB").crop((x0, y0, x0 + w, y0 + h))
    out = io.BytesIO()
    region.save(out, format="JPEG", quality=90)
    return out.getvalue()
