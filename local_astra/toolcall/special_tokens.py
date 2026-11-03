"""The frozen special-token and stop tables of the pinned tokenizer (T-0088a criterion 1). Stdlib only.

``SPECIAL_TOKENS`` is the pinned ``tokenizer.json``'s (sha256 ``cc8d3a0c…``, rev ``2ea8372…``) added tokens that
have ``special: true``, in id order, written out as a literal so that nothing that reads it (the speech gate
first) depends on a file being present. ``tests/cd/unit/test_host_detok_parity.py`` proves all three tables
against the pinned files.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

__all__ = ["SPECIAL_IDS", "SPECIAL_TOKENS", "STOP_IDS", "drop_special_literals"]

_BY_ID: Final[tuple[tuple[int, str], ...]] = (
    (0, "<pad>"),
    (1, "<eos>"),
    (2, "<bos>"),
    (3, "<unk>"),
    (4, "<mask>"),
    (46, "<|tool>"),
    (47, "<tool|>"),
    (48, "<|tool_call>"),
    (49, "<tool_call|>"),
    (50, "<|tool_response>"),
    (51, "<tool_response|>"),
    (52, '<|"|>'),
    (98, "<|think|>"),
    (100, "<|channel>"),
    (101, "<channel|>"),
    (105, "<|turn>"),
    (106, "<turn|>"),
    (255999, "<|image>"),
    (256000, "<|audio>"),
    (258880, "<|image|>"),
    (258881, "<|audio|>"),
    (258882, "<image|>"),
    (258883, "<audio|>"),
    (258884, "<|video|>"),
)

#: The 24 special literals of the pinned tokenizer, in id order.
SPECIAL_TOKENS: Final[tuple[str, ...]] = tuple(lit for _, lit in _BY_ID)

#: id -> literal, read-only.
SPECIAL_IDS: Final = MappingProxyType(dict(_BY_ID))

#: The pinned ``generation_config.json`` ``eos_token_id`` (``<eos>``, ``<|tool_response>``, ``<turn|>``); equals
#: ``local_astra.eval.cd.replay.STOP_IDS``.
STOP_IDS: Final[frozenset[int]] = frozenset({1, 106, 50})


def drop_special_literals(text: str) -> str:
    """``text`` with every ``SPECIAL_TOKENS`` literal deleted (nothing inserted), repeated until none is left,
    because one deletion can splice a new literal (the ``router._drop_channel_markers`` rule)."""
    while any(lit in text for lit in SPECIAL_TOKENS):
        for lit in SPECIAL_TOKENS:
            text = text.replace(lit, "")
    return text
