"""T-0177 A-P1: the one declared difference the older live differentials may discount (docs/tasks/T-0177.md, "Group A
pin moves").

A10 puts ``vision_source: "browser" | "shared"`` on every live turn line (``extras.vision_source``). The differentials
and BASE-equality rows written before T-0177 compare live turn lines with an older commit's; under the browser
source, which is all those rows ever run, the only change A allows there is this key with the value ``"browser"``.
:func:`without_browser_vision_source` returns a capture with that key removed from every ``extras`` dict, after
checking its shape:

- the key is removed only from a dict that is the value of an ``extras`` key (a turn line, or a session-log record
  carrying one), never from anywhere else;
- the value must be ``"browser"`` or ``"shared"``; anything else raises :class:`VisionSourcePinError`;
- ``"shared"`` is **never** stripped, so a browser-regime row that silently became a shared turn still differs from
  BASE and fails.

The base side is never normalised. :func:`tests.unit._t0162c_pins.observed` chains this, and the modules that do not
use ``observed`` call it directly on their observed side.

Regime: CPU, pure, no I/O.
"""

from __future__ import annotations

from typing import Any

from tests.unit._t0195_pins import without_diagnostic_notes

KEY = "vision_source"
#: A10's two values
VALUES = frozenset({"browser", "shared"})


class VisionSourcePinError(AssertionError):
    """An ``extras.vision_source`` that is not A10's shape: not the declared difference."""


def without_browser_vision_source(tree: Any) -> Any:
    """``tree`` with ``vision_source: "browser"`` removed from every ``extras`` dict (shape checked first); a new
    tree, the input is not mutated."""

    def extras(node: dict[str, Any]) -> dict[str, Any]:
        out = {k: walk(v) for k, v in node.items() if k != KEY}
        if KEY not in node:
            return out
        value = node[KEY]
        if not isinstance(value, str) or value not in VALUES:
            raise VisionSourcePinError(
                f"T-0177 A10: extras.vision_source is 'browser' or 'shared'; got {value!r}"
            )
        if value != "browser":
            out[KEY] = value  # never stripped: a shared turn is a real difference
        return out

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                k: extras(v) if k == "extras" and isinstance(v, dict) else walk(v)
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, tuple):
            return tuple(walk(v) for v in node)
        return node

    return walk(tree)


# --- B-P1 ------------------------------------------------------------------------------------------------------------
#
# B7 publishes ``astra.vision.frame`` for the requests a session sends (in every mode) and the HUD server logs one
# ``wire.out`` per published message. The rows written before T-0177 B compare HUD captures and session logs with an
# older commit's; the only change B allows there is those frames and their ``wire.out`` records. They are removed
# only as *elements of a list or tuple* (a capture, a section, a record list) and only after their shape is checked:
#
# - a message (a dict, an ``AstraVisionFrame`` model or the JSON string of one) must validate as ``AstraVisionFrame``;
# - a ``wire.out`` record (``{"kind": "wire.out", "data": {"msg": ...}}``) must carry a ``msg`` whose non-image fields
#   validate as ``AstraVisionFrame`` and whose images each carry an ``identity`` and an A1 ``kind`` (its ``jpeg_b64``
#   is redacted by B7, so the redacted string is not checked).
#
# An item that names the type but fails the shape raises :class:`VisionFramePinError`: it is not the declared
# difference. Nothing else is touched.

FRAME_TYPE = "astra.vision.frame"


class VisionFramePinError(AssertionError):
    """An ``astra.vision.frame`` (or its ``wire.out``) that is not A1's shape: not the declared difference."""


def _frame_dict(item: Any) -> dict[str, Any] | None:
    """``item`` as a message dict if it names ``astra.vision.frame`` (a dict, a model or a JSON string), else None."""
    import json

    from pydantic import BaseModel

    if isinstance(item, BaseModel):
        if getattr(item, "type", None) != FRAME_TYPE:
            return None
        return item.model_dump(mode="json")
    if isinstance(item, str):
        text = item.strip()
        if not (text.startswith("{") and FRAME_TYPE in text):
            return None
        try:
            item = json.loads(text)
        except ValueError:
            return None
    if isinstance(item, dict) and item.get("type") == FRAME_TYPE:
        return item
    return None


def _check_frame(msg: dict[str, Any]) -> None:
    from pydantic import ValidationError

    from local_astra.contracts.protocol import AstraVisionFrame

    try:
        AstraVisionFrame.model_validate(msg)
    except ValidationError as exc:
        raise VisionFramePinError(f"T-0177 A1: not an astra.vision.frame: {msg!r}: {exc}") from exc


def _check_logged_frame(msg: Any) -> None:
    from pydantic import ValidationError

    from local_astra.contracts.protocol import AstraVisionFrame, VisionImage

    if not isinstance(msg, dict):
        raise VisionFramePinError(f"T-0177 B7: a wire.out msg is a dict; got {msg!r}")
    images = msg.get("images", [])
    if not isinstance(images, list):
        raise VisionFramePinError(f"T-0177 B7: wire.out msg.images is a list; got {images!r}")
    try:
        AstraVisionFrame.model_validate({**msg, "images": []})
        kinds = VisionImage.model_fields["kind"].annotation
        for image in images:
            if not (
                isinstance(image, dict)
                and isinstance(image.get("identity"), str)
                and image.get("kind") in getattr(kinds, "__args__", ())
            ):
                raise VisionFramePinError(
                    f"T-0177 B7: a logged vision image has an identity and a kind; got {image!r}"
                )
    except ValidationError as exc:
        raise VisionFramePinError(
            f"T-0177 B7: not a logged astra.vision.frame: {msg!r}: {exc}"
        ) from exc


def is_vision_frame(item: Any) -> bool:
    """True for an ``astra.vision.frame`` message (dict, model, JSON string) or its ``wire.out`` record, shape
    checked (raises :class:`VisionFramePinError` on a malformed one); False for anything else."""
    frame = _frame_dict(item)
    if frame is not None:
        _check_frame(frame)
        return True
    if isinstance(item, dict) and item.get("kind") == "wire.out":
        data = item.get("data")
        msg = data.get("msg") if isinstance(data, dict) else None
        if isinstance(msg, dict) and msg.get("type") == FRAME_TYPE:
            _check_logged_frame(msg)
            return True
    return False


def without_vision_frames(tree: Any) -> Any:
    """``tree`` with every ``astra.vision.frame`` message and ``wire.out`` record of one removed from every list and
    tuple (shape checked first); a new tree, the input is not mutated.

    T-0195 group B census amendment: it also removes the ``mic.level``/``mic.unmute_check`` note records (B1/B4 shape
    checked, ``tests.unit._t0195_pins``), the one shared funnel of the full-session-log differentials."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node if not is_vision_frame(v)]
        if isinstance(node, tuple):
            return tuple(walk(v) for v in node if not is_vision_frame(v))
        return node

    return without_diagnostic_notes(walk(tree))


def published_json_outside_frames(messages: Any) -> str:
    """B-P2: the JSON of every published message except the ``astra.vision.frame`` ones (shape checked). The rows
    that pinned "overview pixels never reach the HUD" (T-0140 A7/A9, T-0150 A2/A4, T-0151 A2-7) read it as "never in
    any message other than ``astra.vision.frame``" (H-T177-2: the panel shows exactly the images sent)."""
    import json

    return json.dumps([m.model_dump(mode="json") for m in without_vision_frames(list(messages))])


def observed(tree: Any) -> Any:
    """Both T-0177 normalisations of an observed side: A-P1 (``extras.vision_source == "browser"``) and B-P1 (the
    vision frames and their ``wire.out`` records)."""
    return without_vision_frames(without_browser_vision_source(tree))
