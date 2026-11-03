"""The native Gemma 4 call format (T-0084b): parse a call, name a draft, build the ``tools`` request field.

The pinned chat template renders an assistant call as ``<|tool_call>call:NAME{key:value,...}<tool_call|>``
(keys dictsorted, strings wrapped in ``<|"|>`` and not escaped, no whitespace). This module reads the text
after ``<|tool_call>``:

- ``parse_call(text)`` -> ``{"name": str, "arguments": dict}``, the same call ``score.parse_call`` returns
  for the JSON format, so ``score.schema_valid``/``score.exact`` judge both formats unchanged; or a
  ``score.ParseFailure`` with one of ``REASONS``. It never raises on a ``str`` (the parser is iterative).
- ``draft_name(draft_call_text)`` -> the v1 tool named by ``\\A\\s*call:NAME\\s*\\{``, else None.
- ``tools_field(tools)`` -> the OpenAI ``tools`` request field, deep-copied, in file order.

Leniency (QN4, rulings B1-B6): whitespace around ``{ } [ ] , :`` and around the call (none after ``call:``),
key order free, ``<tool_call|>`` required, bare values are JSON literals only, duplicate keys refused, nothing
but whitespace after the close. The schema is not used to coerce anything: the scorer stays the judge.

Failure precedence: ``empty``, ``no call prefix``, then a string-aware scan of the rest (``unterminated
string`` for an unpaired ``<|"|>``, ``no close`` when no ``<tool_call|>`` lies outside a string), then
``bad name``, then the arguments left to right (``duplicate key``, ``bare value`` for an invalid bare token,
``malformed arguments`` for a structural break), then the tail (``multiple calls`` for a second
``<|tool_call>``, otherwise ``text after call``).

Pure: the stdlib plus ``toolcall.score``/``.tools`` (no ``local_astra.decoding``, torch, numpy or outlines).
"""

from __future__ import annotations

import copy
import functools
import re
import sys
from typing import Any

from local_astra.toolcall import tools as _tools
from local_astra.toolcall.score import ParseFailure

__all__ = [
    "REASONS",
    "STRING_DELIM",
    "TOOL_CALL_CLOSE",
    "TOOL_CALL_OPEN",
    "draft_name",
    "parse_call",
    "tools_field",
]

TOOL_CALL_OPEN = "<|tool_call>"  # id 48
TOOL_CALL_CLOSE = "<tool_call|>"  # id 49
STRING_DELIM = '<|"|>'  # id 52

REASONS = frozenset(
    {
        "empty",
        "no call prefix",
        "bad name",
        "no close",
        "text after call",
        "multiple calls",
        "duplicate key",
        "bare value",
        "unterminated string",
        "malformed arguments",
    }
)

_PREFIX = "call:"
_WS = re.compile(r"\s*")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: NAME, optional whitespace, then the opening brace of the arguments.
_HEAD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\{")
_DRAFT_HEAD = re.compile(r"\A\s*call:([A-Za-z_][A-Za-z0-9_]*)\s*\{")
#: A bare token runs to whitespace, a structural character or the start of a special piece.
_TOKEN = re.compile(r'(?:[^\s{}\[\],:<]|<(?!\|"\|>|tool_call\||\|tool_call>))+')
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")
_LITERALS: dict[str, Any] = {"true": True, "false": False, "null": None}

_MALFORMED = ParseFailure("malformed arguments")


def _skip_ws(s: str, i: int) -> int:
    m = _WS.match(s, i)
    return m.end() if m else i


def _scan_for_close(s: str, i: int) -> ParseFailure | None:
    """Pair the string delimiters from ``i`` and look for a close tag outside every string."""
    n_delim = len(STRING_DELIM)
    while True:
        q = s.find(STRING_DELIM, i)
        c = s.find(TOOL_CALL_CLOSE, i)
        if c >= 0 and (q < 0 or c < q):
            return None
        if q < 0:
            return ParseFailure("no close")
        end = s.find(STRING_DELIM, q + n_delim)
        if end < 0:
            return ParseFailure("unterminated string")
        i = end + n_delim


def _literal(tok: str) -> tuple[bool, Any]:
    """A bare token as a JSON literal: ``(True, value)``, or ``(False, None)`` if it is not one."""
    if tok in _LITERALS:
        return True, _LITERALS[tok]
    m = _NUMBER.fullmatch(tok)
    if m is None:
        return False, None
    if m.group(1) is None and m.group(2) is None:
        limit = sys.get_int_max_str_digits()
        if limit and len(tok.lstrip("-")) > limit:
            return False, None
        return True, int(tok)
    return True, float(tok)


def _parse_arguments(s: str, i: int) -> tuple[dict[str, Any], int] | ParseFailure:
    """Parse the object whose ``{`` is at ``s[i]``; returns it and the index after its ``}``.

    Iterative (an explicit stack), so deep nesting cannot exhaust the interpreter's recursion limit.
    """
    root: dict[str, Any] = {}
    stack: list[dict[str, Any] | list[Any]] = [root]
    keys: list[str] = [""]  # the key awaiting its value, per open object ("" for arrays)
    state = "key_or_end"  # key_or_end | key | value_or_end | value | after
    i += 1
    while True:
        i = _skip_ws(s, i)
        top = stack[-1]
        if state in ("key_or_end", "value_or_end") and s.startswith(
            "}" if state == "key_or_end" else "]", i
        ):
            stack.pop()
            keys.pop()
            i += 1
            if not stack:
                return root, i
            state = "after"
            continue
        if state in ("key_or_end", "key"):
            assert isinstance(top, dict)
            m = _IDENT.match(s, i)
            if m is None:
                return _MALFORMED
            key = m.group()
            i = _skip_ws(s, m.end())
            if not s.startswith(":", i):
                return _MALFORMED
            if key in top:
                return ParseFailure("duplicate key")
            keys[-1] = key
            i += 1
            state = "value"
            continue
        if state == "after":
            if s.startswith(",", i):
                i += 1
                state = "key" if isinstance(top, dict) else "value"
                continue
            if s.startswith("}" if isinstance(top, dict) else "]", i):
                stack.pop()
                keys.pop()
                i += 1
                if not stack:
                    return root, i
                continue
            return _MALFORMED
        # state is "value_or_end" (not at "]") or "value": one value.
        value: Any
        child: dict[str, Any] | list[Any] | None = None
        if s.startswith(STRING_DELIM, i):
            start = i + len(STRING_DELIM)
            end = s.find(STRING_DELIM, start)
            if end < 0:
                return ParseFailure("unterminated string")
            value = s[start:end]
            i = end + len(STRING_DELIM)
        elif s.startswith("{", i):
            child = {}
            value = child
            i += 1
        elif s.startswith("[", i):
            child = []
            value = child
            i += 1
        else:
            m = _TOKEN.match(s, i)
            if m is None:
                return _MALFORMED
            ok, value = _literal(m.group())
            if not ok:
                return ParseFailure("bare value")
            i = m.end()
        if isinstance(top, dict):
            top[keys[-1]] = value
        else:
            top.append(value)
        if child is None:
            state = "after"
        else:
            stack.append(child)
            keys.append("")
            state = "key_or_end" if isinstance(child, dict) else "value_or_end"


def parse_call(text: str) -> dict[str, Any] | ParseFailure:
    """Parse ``call:NAME{ARGS}<tool_call|>`` (the text after ``<|tool_call>``) into ``{"name", "arguments"}``.

    The caller removes the stop text first; anything but whitespace after the close is refused.
    """
    if not text.strip():
        return ParseFailure("empty")
    i = _skip_ws(text, 0)
    if not text.startswith(_PREFIX, i):
        return ParseFailure("no call prefix")
    i += len(_PREFIX)
    scan = _scan_for_close(text, i)
    if scan is not None:
        return scan
    head = _HEAD.match(text, i)
    if head is None:
        return ParseFailure("bad name")
    parsed = _parse_arguments(text, head.end() - 1)
    if isinstance(parsed, ParseFailure):
        return parsed
    arguments, i = parsed
    i = _skip_ws(text, i)
    if not text.startswith(TOOL_CALL_CLOSE, i):
        return _MALFORMED
    tail = text[i + len(TOOL_CALL_CLOSE) :].lstrip()
    if tail:
        if tail.startswith(TOOL_CALL_OPEN):
            return ParseFailure("multiple calls")
        return ParseFailure("text after call")
    return {"name": head.group(1), "arguments": arguments}


@functools.cache
def _v1_names() -> frozenset[str]:
    return frozenset(_tools.load().names)


def draft_name(draft_call_text: str) -> str | None:
    """The v1 tool named by a (possibly truncated) native draft: ``\\A\\s*call:NAME\\s*\\{``, else None."""
    m = _DRAFT_HEAD.match(draft_call_text)
    if m is None or m.group(1) not in _v1_names():
        return None
    return m.group(1)


def tools_field(tools: _tools.ToolSet) -> list[dict[str, Any]]:
    """The OpenAI ``tools`` request field: one ``{"type": "function", "function": decl}`` per declaration."""
    return [{"type": "function", "function": dict(copy.deepcopy(d))} for d in tools.declarations]
