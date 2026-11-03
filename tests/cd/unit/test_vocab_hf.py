"""T-0063 criterion 4: the grammar vocabulary comes from the pinned HF ``tokenizer.json``.

Expected ids are diffgemma-fa's Phase 0 facts (``compile/vocab.py`` docstrings, SPEC §3.6: the header
``[100, 45518, 107, 101]``; ``END_TOKENS = (1, 106, 50)``; ``<|channel>``/``<channel|>`` = 100/101;
``<|turn>`` = 105). The tokenizer itself is read with the ``tokenizers`` library as an independent oracle,
never through the code under test.

API these rows assume (named in the T-0063 test-author report): ``vocab.build_vocabulary()`` returns the
``outlines_core.Vocabulary`` the lift uses; ``grammars.compile_sudoku(puzzle)`` returns a vendored
``CompiledAutomaton`` with ``tokenizer_hash`` and ``vocab_size`` filled in.
"""

from __future__ import annotations

import json
import random

import pytest

from tests.cd._compile_support import (
    CHANNEL_CLOSE,
    CHANNEL_OPEN,
    EOS,
    NEWLINE,
    PHASE0_HEADER,
    RESERVED_IDS,
    THOUGHT,
    TOKENIZER_SHA256,
    TOOL_RESPONSE,
    TURN_CLOSE,
    TURN_OPEN,
    VOCAB_SIZE,
    compiled_sudoku,
    encode,
    hf_tokenizer,
    sudoku_cut,
    tokenizer_json_path,
    vocab,
    walker,
)

#: Phase 0 facts: piece -> id.
PHASE0_IDS = {
    "<|channel>": CHANNEL_OPEN,
    "<channel|>": CHANNEL_CLOSE,
    "\n": NEWLINE,
    "thought": THOUGHT,
    "<eos>": EOS,
    "<turn|>": TURN_CLOSE,
    "<|tool_response>": TOOL_RESPONSE,
    "<|turn>": TURN_OPEN,
}


def _first_sudoku() -> tuple[tuple[int, ...], ...]:
    return sudoku_cut("test")[0][1]


def test_the_oracle_tokenizer_has_the_phase0_ids() -> None:
    """Fixture check: the pinned tokenizer.json itself carries the Phase 0 ids (else every row lies)."""
    tok = hf_tokenizer()
    assert tok.get_vocab_size() == VOCAB_SIZE
    assert {p: tok.token_to_id(p) for p in PHASE0_IDS} == PHASE0_IDS
    assert encode("<|channel>thought\n<channel|>") == PHASE0_HEADER


def test_tokenizer_hash_is_the_sha256_of_the_pinned_tokenizer_json() -> None:
    a = compiled_sudoku(_first_sudoku())
    assert a.tokenizer_hash == TOKENIZER_SHA256, (
        f"automaton.tokenizer_hash={a.tokenizer_hash!r}; criterion 4 wants the sha256 of "
        f"{tokenizer_json_path()} ({TOKENIZER_SHA256})"
    )


def test_the_automaton_spans_the_whole_262144_id_vocabulary() -> None:
    a = compiled_sudoku(_first_sudoku())
    assert a.vocab_size == VOCAB_SIZE
    # the Σ tail after acceptance must reach the top id, not stop at a smaller SentencePiece size
    w = walker(a)
    acc = w.acc()
    tail = [cls for src, dst, cls in w.edges if src == acc and dst == acc]
    assert tail and any(w.member(c, VOCAB_SIZE - 1) for c in tail)


def test_the_compiled_header_accepts_the_phase0_prefix() -> None:
    """The header is built on ids 100 . name . 107 . 101; ``thought`` (45518) is a legal name."""
    puzzle = _first_sudoku()
    a = compiled_sudoku(puzzle)
    solution = sudoku_cut("test")[0][2]
    grid = "\n".join("".join(str(c) for c in row) for row in solution)
    assert walker(a).accepts([*PHASE0_HEADER, *encode(grid), EOS])


def _piece_bytes(piece: str) -> bytes:
    """What a vocabulary piece stands for, per tokenizer.json's own decoder (Replace ▁->' ', ByteFallback)."""
    if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
        return bytes([int(piece[3:5], 16)])
    return piece.replace("▁", " ").encode("utf-8")


def test_the_grammar_vocabulary_maps_pieces_as_tokenizer_json_does() -> None:
    """A seeded sample of ordinary pieces, plus the load-bearing ones, map to their own ids."""
    voc = vocab().build_vocabulary()
    data = json.loads(tokenizer_json_path().read_text())
    special = {t["id"] for t in data["added_tokens"] if t.get("special")}
    pieces = {i: p for p, i in data["model"]["vocab"].items()}
    ordinary = sorted(i for i in pieces if i not in special and i not in RESERVED_IDS)
    rng = random.Random(20260929)
    sample = sorted(set(rng.sample(ordinary, 2000)) | {NEWLINE, THOUGHT, 138, 2717, 3723, 236782})
    # 138 = "▁▁" (two spaces), 2717 = "```", 3723 = "json", 236782 = "{"
    byte_fallback_newline = data["model"]["vocab"]["<0x0A>"]
    sample.append(byte_fallback_newline)
    wrong = {}
    for i in sample:
        b = _piece_bytes(pieces[i])
        ids = voc.get(b) or []
        if i not in ids:
            wrong[i] = (pieces[i], b, ids)
    assert not wrong, (
        f"{len(wrong)} pieces not mapped to their tokenizer.json id, e.g. {list(wrong.items())[:5]}"
    )


@pytest.mark.parametrize("reserved", sorted(RESERVED_IDS))
def test_reserved_ids_are_not_in_the_grammar_vocabulary(reserved: int) -> None:
    voc = vocab().build_vocabulary()
    piece = hf_tokenizer().id_to_token(reserved)
    for b in {piece.encode("utf-8"), _piece_bytes(piece)}:
        assert reserved not in (voc.get(b) or []), (
            f"id {reserved} ({piece!r}) is in the grammar alphabet"
        )


def test_the_reserved_row_can_see_a_reserved_id() -> None:
    """Probe self-check for the row above: the lookup it relies on does find an ordinary special-looking
    piece, so its "absent" is a real absence and not a lookup that never matches."""
    voc = vocab().build_vocabulary()
    assert NEWLINE in (voc.get(b"\n") or [])
    assert THOUGHT in (voc.get(b"thought") or [])


#: diffgemma-fa ``preserved/bfcl_compile_report.json`` (27c7d0d): Phase 1 compile metrics under the
#: stock report settings, measured upstream on the SentencePiece vocabulary. (states, edges, classes,
#: k_max, max_neg_size, nnz_sum)
UPSTREAM_PHASE1 = {
    "live_simple_0-0-0": (39, 93, 76, 1585, 1585, 9022),
    "live_simple_40-17-0": (214, 676, 281, 1100, 0, 372),
}


@pytest.mark.parametrize("record_id", sorted(UPSTREAM_PHASE1))
def test_stock_compile_metrics_match_upstream(record_id: str) -> None:
    """Ties the HF vocabulary to upstream's *content*, not just its size: the per-class member counts
    (``nnz_sum``, ``k_max``, ``max_neg_size``) only reproduce if the same pieces map to the same bytes.
    Settings are ``tasks/bfcl.py``'s report job: ``from_bfcl``, its ``ALLOW``, ``allow_wildcard``, no
    whitespace (``""``, the Phase-1 default), no channel header, no fence."""
    from tests.cd._compile_support import bfcl_function, vendored

    pipeline = vendored("pipeline")
    allow = vendored("tasks.bfcl").ALLOW
    a = pipeline.compile_json_schema(
        bfcl_function(record_id)["parameters"],
        name=record_id,
        from_bfcl=True,
        allow=allow,
        allow_wildcard=True,
        whitespace_pattern="",
        channel_header=False,
        fence=False,
    ).automaton
    t = a.tables
    got = (a.n_states, a.n_edges, t.n_classes, t.k_max, t.max_neg_size, t.nnz_sum)
    assert got == UPSTREAM_PHASE1[record_id]
