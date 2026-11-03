# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored from vLLM v0.29.0 ``vllm/parser/gemma4.py`` (T-0084a test oracle).

Source: the local image ``vllm/vllm-openai:v0.29.0`` (image id sha256:c2914767605584b6...), file
``/usr/local/lib/python3.12/dist-packages/vllm/parser/gemma4.py``, sha256
``478c04e66ffdda42a1457cb3b824166ec7de98fd8067953178624ad042879c94``. Copied verbatim: the SPDX header,
the delimiter constants (source lines 38-43), ``_PARTIAL_DELIM_SUFFIXES``/``_strip_partial_delim``,
``_parse_gemma4_args`` and ``_parse_gemma4_array`` (source lines 47-282). The only change: vLLM's
``init_logger`` is replaced by a no-op logger, and the vLLM imports the two functions do not use are not
copied. Provenance is recorded in ``tests/cd/oracle/MANIFEST.md``. Do not edit the copied bodies (``# fmt: off`` keeps ruff from reformatting them).
"""

from __future__ import annotations

from typing import Any

UPSTREAM_SHA256 = "478c04e66ffdda42a1457cb3b824166ec7de98fd8067953178624ad042879c94"


class _NoOpLogger:
    def warning(self, *args: Any, **kwargs: Any) -> None:
        return None


logger = _NoOpLogger()

# ---------------------------------------------------------------------------- verbatim from here
# fmt: off
CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'
_DELIM_LEN = len(STRING_DELIM)


# ---------------------------------------------------------------------------
# Gemma4 argument parser
# ---------------------------------------------------------------------------

_PARTIAL_DELIM_SUFFIXES = tuple(
    STRING_DELIM[:k] for k in range(len(STRING_DELIM), 0, -1)
)


def _strip_partial_delim(value: str) -> str:
    """Strip a trailing partial ``STRING_DELIM`` prefix from *value*.

    Prevents partial delimiters from leaking into the streamed JSON diff.
    """
    for suffix in _PARTIAL_DELIM_SUFFIXES:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _parse_gemma4_args(args_str: str, *, partial: bool = False) -> dict:
    """Parse Gemma4's custom key:value format into a Python dict.

    Format examples::

        location:<|"|>Tokyo<|"|>
        location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
        count:42,flag:true
        nested:{inner_key:<|"|>val<|"|>}
        items:[<|"|>a<|"|>,<|"|>b<|"|>]

    Args:
        args_str: The raw Gemma4 argument string.
        partial: When True (streaming), bare values at end of string are
            omitted because they may be incomplete and type-unstable
            (e.g. partial boolean parsed as bare string).

    Returns a dict ready for ``json.dumps()``.
    """
    if not args_str or not args_str.strip():
        return {}

    result: dict = {}
    i = 0
    n = len(args_str)

    while i < n:
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        if key.startswith(STRING_DELIM) and key.endswith(STRING_DELIM):
            key = key[_DELIM_LEN:-_DELIM_LEN]
        i += 1

        if i >= n:
            if not partial:
                result[key] = ""
            break

        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            if not partial:
                result[key] = ""
            break

        if args_str[i : i + _DELIM_LEN] == STRING_DELIM:
            i += _DELIM_LEN
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                # Unterminated string — take rest, strip partial delimiter.
                value = args_str[val_start:]
                if partial:
                    value = _strip_partial_delim(value)
                result[key] = value
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + _DELIM_LEN

        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i : i + _DELIM_LEN] == STRING_DELIM:
                    # Skip over string contents to avoid counting { inside strings
                    i += _DELIM_LEN
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + _DELIM_LEN
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                # Incomplete nested object — use i (not i-1) to avoid
                # dropping the last char, and recurse as partial.
                result[key] = _parse_gemma4_args(args_str[obj_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])

        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i : i + _DELIM_LEN] == STRING_DELIM:
                    i += _DELIM_LEN
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + _DELIM_LEN
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(args_str[arr_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1])

        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                # Value may be incomplete (e.g. partial boolean) —
                # withhold to avoid type instability during streaming.
                break
            if i == val_start:
                logger.warning(
                    "Gemma4 args parser made no progress at position %d; "
                    "aborting on malformed input.",
                    i,
                )
                break
            raw_val = args_str[val_start:i].strip()
            if partial and raw_val.endswith("."):
                # Digits may still arrive (e.g. "108." -> "108.2");
                # withhold to avoid corrupting the streaming diff.
                break
            result[key] = raw_val

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list:
    items: list = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        if arr_str[i : i + _DELIM_LEN] == STRING_DELIM:
            i += _DELIM_LEN
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + _DELIM_LEN

        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i : i + _DELIM_LEN] == STRING_DELIM:
                    i += _DELIM_LEN
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + _DELIM_LEN if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_args(arr_str[obj_start:i], partial=True))
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i : i + _DELIM_LEN] == STRING_DELIM:
                    i += _DELIM_LEN
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + _DELIM_LEN if nd != -1 else n
                    continue
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_array(arr_str[sub_start:i], partial=True))
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            if i == val_start:
                logger.warning(
                    "Gemma4 array parser made no progress at position %d; "
                    "aborting on malformed input.",
                    i,
                )
                break
            raw_val = arr_str[val_start:i].strip()
            if partial and raw_val.endswith("."):
                break
            items.append(raw_val)

    return items
# fmt: on
