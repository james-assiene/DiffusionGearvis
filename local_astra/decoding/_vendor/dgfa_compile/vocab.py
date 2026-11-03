"""Gemma tokenizer -> `outlines_core.Vocabulary`. SPEC §1.3(3), §4.3.

`Vocabulary.from_pretrained` is not usable: it needs `google/gemma-3-4b-it`,
which is gated, and returns HTTP 401 (Phase 0). Building from Gemma's own
SentencePiece model needs no HF access, takes ~3 s, and is what SPEC §1.3 now
prescribes.

[local_astra deviation, T-0063] The SentencePiece model came from the JAX
`gemma` package. In local_astra the pieces are read from the HF
`tokenizer.json` of the pinned model revision instead (located and hashed by
`local_astra.decoding.vocab`); the mapping rule is unchanged. See
`../PROVENANCE.md`.
"""

from __future__ import annotations

import functools
from typing import Any

__all__ = ["build_vocabulary", "gemma_tokenizer", "HfGemmaTokenizer", "END_TOKENS", "CHANNEL_OPEN", "CHANNEL_CLOSE"]

#: Verified in Phase 0. `Sampler`/`ChatSampler` build
#: `end_tokens = (EOS, END_OF_TURN, BEGIN_OF_TOOL_RESPONSE, *stop_tokens)`.
#: Handling all of them is SPEC §3.5 trap 3 — the most likely source of an
#: empty state set at a block boundary.
END_TOKENS: tuple[int, ...] = (1, 106, 50)  # EOS, END_OF_TURN, BEGIN_OF_TOOL_RESPONSE

PAD_TOKEN = 0

#: The channel delimiters. Both are **single dedicated token ids** (Phase 0),
#: which is what makes SPEC §3.6's two-state construction work.
CHANNEL_OPEN = 100   # <|channel>
CHANNEL_CLOSE = 101  # <channel|>


#: The SentencePiece control/unknown pieces, which the upstream loop skipped via
#: `sp.IsControl(i) or sp.IsUnknown(i)` (4 pieces, Phase 0). `tokenizer.json`
#: has no control flag, so they are named: these are the four pieces that emit
#: no bytes when decoded.
CONTROL_PIECES: frozenset[str] = frozenset({"<pad>", "<eos>", "<bos>", "<unk>"})


class HfGemmaTokenizer:
    """The pinned HF `tokenizer.json`, exposing what this package reads of it."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = int(tokenizer.get_vocab_size(with_added_tokens=True))


@functools.cache
def gemma_tokenizer() -> Any:
    """The Gemma 4 tokenizer (cached), read from the pinned `tokenizer.json`."""
    from local_astra.decoding.vocab import load_tokenizer

    return HfGemmaTokenizer(load_tokenizer())


#: Tokens that must never appear **inside** a compiled grammar.
#:
#: This is a correctness requirement, not hygiene, and it is easy to miss:
#: `<eos>` (1) and `<pad>` (0) *are* SentencePiece control tokens and so drop
#: out of the vocabulary automatically — but **`<turn|>` (106) and
#: `<|tool_response>` (50) are NOT**. They are ordinary pieces whose bytes are
#: plain ASCII, so `outlines_core` will happily let a JSON string body match
#: them. If the grammar can emit one, `_truncate_canvas_at_stop_tokens` cuts
#: the canvas mid-grammar and `A_{k+1}` goes empty — SPEC §3.1b failure mode 3,
#: reached from the *grammar* region rather than the free-text region SPEC
#: describes. Augmenting on top of that also makes the automaton spuriously
#: nondeterministic, which silently switches eq (8) onto its slow path.
#:
#: The channel delimiters are excluded for the same reason: they are single
#: dedicated ids (100/101) that the §3.6 HEADER construction places explicitly,
#: and a grammar that can re-open a channel mid-value is not something the
#: model was trained to produce.
RESERVED_TOKENS: frozenset[int] = frozenset(
    {PAD_TOKEN, *END_TOKENS, CHANNEL_OPEN, CHANNEL_CLOSE, 105}  # 105 = <|turn>
)


@functools.cache
def build_vocabulary(exclude: frozenset[int] | None = None) -> Any:
    """An `outlines_core.Vocabulary` over Gemma's byte-level pieces.

    SentencePiece normalisation: `<0xXX>` byte-fallback pieces map to the raw
    byte; every other piece has U+2581 standing in for a leading space. Control
    and unknown pieces emit no bytes and are excluded — they must not appear in
    any automaton alphabet (SPEC §3.6), and stop tokens are handled out of band
    anyway (SPEC §4.3: `guide.advance(eos)` raises).

    Args:
      exclude: token ids to keep out of the grammar alphabet. Defaults to
        `RESERVED_TOKENS` — read its docstring before overriding, the default
        is load-bearing.

    Measured: 262,140 pieces map from SentencePiece (256 byte-fallback, 4
    control/unknown skipped); the reserved set removes 5 more, leaving 262,135.
    ~2.9 s.
    """
    import outlines_core as oc

    exclude = RESERVED_TOKENS if exclude is None else exclude
    tok = gemma_tokenizer()
    pieces = tok.tokenizer.get_vocab(with_added_tokens=True)
    by_id = {i: p for p, i in pieces.items()}
    vocab = oc.Vocabulary(int(tok.tokenizer.token_to_id("<eos>")), {})

    for i in range(tok.vocab_size):
        piece = by_id.get(i)
        if piece is None or i in exclude or piece in CONTROL_PIECES:
            continue
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            b = bytes([int(piece[3:5], 16)])
        else:
            b = piece.replace("▁", " ").encode("utf-8")
            if not b:
                continue
        vocab.insert(b, i)
    return vocab
