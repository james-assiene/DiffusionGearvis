"""Shared helpers for the T-0088a rows (host detokeniser, special-token tables, speech gate).

Nothing here imports the code under test at module level. ``detok_mod()`` / ``special_mod()`` import it inside
a row, so while ``local_astra.toolcall.detok`` or ``special_tokens`` does not exist each row fails
*individually* with an ``AssertionError`` naming the missing module, and collection never errors.

API the rows assume (docs/tasks/T-0088a.md criterion 2; where the task is silent, the choice is stated here and
in the test-author's report):

- ``HostDetokenizer.load()``: the pinned file from the local HF cache;
- ``HostDetokenizer(path, expected_sha256)``: "a constructor that takes an explicit path and expected sha",
  both positional. Every refusal raises ``ValueError`` (or a subclass) whose message names the file;
- ``det.decode(ids) -> str``; ``det.stream()`` -> an object with ``feed(ids) -> str`` and ``finish() -> str``;
- ``UnknownTokenId`` is an attribute of ``local_astra.toolcall.detok``.

Also used by ``tests/cd/unit/test_host_detok_parity.py``, which proves every hand-computed row below against
the ``tokenizers`` oracle on the pinned file (so the expected strings here are not taken on trust).
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Iterator, Sequence
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "llm" / "vocab_subset_cc8d3a0c.json"

DETOK = "local_astra.toolcall.detok"
SPECIAL = "local_astra.toolcall.special_tokens"

MODEL_ID = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
MODEL_REVISION = "2ea837236295d617ac27f8c17d61228081932c40"
PINNED_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"

#: The 24 ``special: true`` added tokens of the pinned file, in id order (read off the file on 2026-10-12;
#: ``test_special_tokens.py`` checks this copy against the fixture, the cd row checks the fixture against the
#: file). Written out so the speech-gate rows do not depend on any file either.
SPECIALS_BY_ID: tuple[tuple[int, str], ...] = (
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
SPECIAL_LITERALS: tuple[str, ...] = tuple(lit for _, lit in SPECIALS_BY_ID)

FFFD = "�"


def _need(modname: str) -> ModuleType:
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError as exc:
        if exc.name == modname:
            raise AssertionError(
                f"T-0088a not implemented: `{modname}` does not exist ({exc})"
            ) from exc
        raise


def detok_mod() -> ModuleType:
    return _need(DETOK)


def special_mod() -> ModuleType:
    return _need(SPECIAL)


@cache
def _fixture_text() -> str:
    return FIXTURE.read_text()


def fixture() -> dict[str, Any]:
    """A fresh copy every call (rows mutate it)."""
    data: dict[str, Any] = json.loads(_fixture_text())
    return data


def piece_ids() -> dict[str, int]:
    vocab: dict[str, int] = fixture()["tokenizer"]["model"]["vocab"]
    return vocab


def ids_of(pieces: Sequence[str]) -> list[int]:
    v = piece_ids()
    return [v[p] for p in pieces]


def write_tokenizer(
    directory: Path, mutate: Callable[[dict[str, Any]], None] | None = None
) -> tuple[Path, str]:
    """Write the fixture's ``tokenizer.json``-shaped subset to ``directory/tokenizer.json``.

    ``mutate`` edits the object first. Returns the path and the sha256 of the bytes written, so a refusal row
    can pass the *correct* sha and reach the check it is aimed at.
    """
    tok = fixture()["tokenizer"]
    if mutate is not None:
        mutate(tok)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tokenizer.json"
    data = json.dumps(tok, ensure_ascii=False).encode()
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def subset_detokenizer(directory: Path) -> Any:
    path, sha = write_tokenizer(directory)
    return detok_mod().HostDetokenizer(path, sha)


def partitions(seq: Sequence[int]) -> Iterator[list[list[int]]]:
    """Every partition of ``seq`` into contiguous non-empty chunks (2 ** (len - 1) of them)."""
    n = len(seq)
    for mask in range(1 << max(n - 1, 0)):
        chunks: list[list[int]] = []
        start = 0
        for k in range(1, n):
            if mask & (1 << (k - 1)):
                chunks.append(list(seq[start:k]))
                start = k
        chunks.append(list(seq[start:]))
        yield chunks


def run_stream(det: Any, chunks: Sequence[Sequence[int]]) -> tuple[list[str], str]:
    s = det.stream()
    outs = [s.feed(list(c)) for c in chunks]
    return outs, s.finish()


# --------------------------------------------------------------------------- hand rows
#
# Expected strings follow HF ``ByteFallback`` + ``Fuse`` (tokenizers 0.23.2): consecutive ``<0xNN>`` pieces
# form one run; a run that is valid UTF-8 decodes to its string, and a run that is *not* becomes one U+FFFD
# **per byte of the whole run** — including bytes that would be valid on their own (``A`` + ``0xFF`` ->
# two U+FFFD). That is not what ``bytes.decode("utf-8", "replace")`` does (it gives ``"A�"``). ``▁`` -> space,
# no stripping; specials verbatim. Each row is re-proved against the oracle in the cd module.

HAND_ROWS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("ascii_byte", ("<0x41>",), "A"),
    ("two_byte_char", ("▁I", "<0xC3>", "<0xA9>", "."), " Ié."),
    ("three_byte_char", ("▁I", "<0xE2>", "<0x82>", "<0xAC>", "."), " I€."),
    ("four_byte_char", ("▁I", "<0xF0>", "<0x9F>", "<0x98>", "<0x80>", "."), " I😀."),
    ("invalid_continuation", ("<0xE2>", "<0x41>", "<0xAC>"), FFFD * 3),
    ("valid_byte_lost_in_invalid_run", ("<0x41>", "<0xFF>"), FFFD * 2),
    ("complete_char_then_invalid_byte", ("<0xE2>", "<0x82>", "<0xAC>", "<0xFF>"), FFFD * 4),
    ("truncated_run", ("▁I", "<0xE2>", "<0x82>"), " I" + FFFD * 2),
    ("special_right_after_run", ("<0xE2>", "<0x82>", "<0xAC>", "<tool_call|>"), "€<tool_call|>"),
    (
        "special_inside_run",
        ("<0xE2>", "<|tool_call>", "<0x82>", "<0xAC>"),
        FFFD + "<|tool_call>" + FFFD * 2,
    ),
    (
        "specials_verbatim",
        ("<bos>", "<|channel>", "<channel|>", "<eos>"),
        "<bos><|channel><channel|><eos>",
    ),
    ("space_runs", ("▁▁", "▁▁▁", "▁the"), "      the"),
    ("run_between_pieces", ("▁a", "<0xC3>", "<0xA9>", "▁b"), " aé b"),
    ("runs_split_by_special", ("<0x41>", '<|"|>', "<0xFF>"), 'A<|"|>' + FFFD),
)
