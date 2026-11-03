"""Total JSON codec for the HUD protocol: ``decode`` never raises (docs/03 §3 "Protocol codecs").

- known ``astra.*`` type → the strict model;
- non-``astra.`` type → ``RealtimeEvent`` passthrough;
- unknown ``astra.*`` type → ``Unknown`` (the caller ignores it);
- anything malformed (bad JSON, missing/invalid fields, wrong ``type`` kind) → ``Malformed``,
  which the Conductor answers with ``astra.error`` and never a crash.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from local_astra.contracts.base import Contract
from local_astra.contracts.protocol import (
    ASTRA_MESSAGE_TYPES,
    AstraMessageAdapter,
    RealtimeEvent,
)

_KNOWN_ASTRA_TYPES: frozenset[str] = frozenset(
    str(m.model_fields["type"].default) for m in ASTRA_MESSAGE_TYPES
)

MAX_MESSAGE_BYTES = 8 * 1024 * 1024  # a 1080p JPEG frame is ~300 KB; leave head-room, not infinity

# A ``\uD800``-``\uDFFF`` escape, any case: the only way valid UTF-8 on the wire decodes to a
# surrogate. A frame without one takes no walk (T-0216c).
_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F]")
_SURROGATE = re.compile("[\ud800-\udfff]")


@dataclass(frozen=True)
class Unknown:
    type: str


@dataclass(frozen=True)
class Malformed:
    error: str
    type: str | None = None


Decoded = Contract | Unknown | Malformed


def _replace_surrogates(obj: Any) -> Any:
    """Replace every surrogate code point in every string of a ``json.loads`` value with U+FFFD.

    After ``json.loads`` a valid escaped pair is one code point, so any surrogate left is lone and
    has no UTF-8 encoding: the echo, the LLM body and the memory store would all raise on it
    (T-0216c). The value is fresh from the parser, so containers are fixed in place; an explicit
    stack, because a frame under the size limit can nest ~990 deep. Dict key order is kept.
    """
    if isinstance(obj, str):
        return _SURROGATE.sub("\ufffd", obj)
    stack: list[Any] = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            items = list(node.items())
            node.clear()
            for k, v in items:
                if isinstance(v, str):
                    v = _SURROGATE.sub("\ufffd", v)
                elif isinstance(v, dict | list):
                    stack.append(v)
                node[_SURROGATE.sub("\ufffd", k)] = v
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if isinstance(v, str):
                    node[i] = _SURROGATE.sub("\ufffd", v)
                elif isinstance(v, dict | list):
                    stack.append(v)
    return obj


def encode(msg: Contract) -> str:
    """Serialise a protocol message. ``RealtimeEvent`` is flattened back to its wire shape."""
    if isinstance(msg, RealtimeEvent):
        return json.dumps({"type": msg.type, **msg.payload}, separators=(",", ":"))
    return msg.model_dump_json(exclude_none=True)


def decode(raw: str | bytes) -> Decoded:
    if isinstance(raw, bytes | bytearray):
        if len(raw) > MAX_MESSAGE_BYTES:
            return Malformed(error=f"message exceeds {MAX_MESSAGE_BYTES} bytes")
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as e:
            return Malformed(error=f"not utf-8: {e}")
    elif len(raw) * 4 > MAX_MESSAGE_BYTES:
        # The limit counts UTF-8 bytes, as ``hud/src/protocol.js`` does (T-0008 R11): a message
        # of 3M euro signs is 9 MB on the wire. A code point is at most 4 UTF-8 bytes, so below
        # this guard the message is certainly under the limit and is not encoded at all -- the
        # common case stays as cheap as the character count was. (The page's own short-circuit
        # is `* 3`, which is right for its UTF-16 units and wrong for code points.)
        try:
            size = len(raw.encode("utf-8"))
        except UnicodeEncodeError as e:
            # A lone surrogate has no UTF-8 encoding; ``decode`` is total, so this is Malformed
            # with the bytes branch's wording, not a raise.
            return Malformed(error=f"not utf-8: {e}")
        if size > MAX_MESSAGE_BYTES:
            return Malformed(error=f"message exceeds {MAX_MESSAGE_BYTES} bytes")
    try:
        obj: Any = json.loads(raw)
    except json.JSONDecodeError as e:
        return Malformed(error=f"invalid json: {e.msg} at {e.pos}")
    except ValueError:
        # An int literal past Python's int-from-string digit limit (4300 by default) raises a
        # plain ``ValueError``, after ``JSONDecodeError`` (its subclass) above. The reason is
        # fixed: the exception text would quote the digit count, never the payload, but a fixed
        # text cannot leak (T-0216).
        return Malformed(error="invalid json: integer literal too long")
    except RecursionError:
        # Nesting deeper than the parser's recursion limit, well under the size limit (T-0216).
        return Malformed(error="invalid json: nesting too deep")
    # ``"\\" in`` first: a one-character find is a memchr, ~1 us on a 150 KB base64 frame where
    # the regex scan alone is ~27 us (T-0216c).
    if "\\" in raw and _SURROGATE_ESCAPE.search(raw):
        obj = _replace_surrogates(obj)
    if not isinstance(obj, dict):
        return Malformed(error="top-level value must be an object")
    t = obj.get("type")
    if not isinstance(t, str) or not t:
        return Malformed(error="missing or non-string 'type'")
    if t.startswith("astra."):
        if t not in _KNOWN_ASTRA_TYPES:
            return Unknown(type=t)
        try:
            return AstraMessageAdapter.validate_python(obj)
        except ValidationError as e:
            first = e.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            return Malformed(error=f"{loc}: {first.get('msg', 'invalid')}", type=t)
    payload = {k: v for k, v in obj.items() if k != "type"}
    try:
        return RealtimeEvent(type=t, payload=payload)
    except ValidationError as e:
        return Malformed(error=f"type: {e.errors()[0].get('msg', 'invalid')}", type=t)
