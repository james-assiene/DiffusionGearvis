"""T-0159 A2, the cd-group differential: ``HostDetokenizer.pieces`` against ``tokenizers`` on the pinned file.

Oracles, both from ``tokenizers`` on the pinned ``tokenizer.json`` (``cc8d3a0c…``, rev ``2ea8372…``, read from the
local HF cache through ``local_astra.decoding.vocab``), never from the code under test:

* ``Tokenizer.decode([id], skip_special_tokens=False)`` -- the file's own decoder pipeline on one id. For every id
  that is not a byte-fallback piece this is the display piece (``▁`` -> space, added tokens verbatim).
* ``Tokenizer.id_to_token(id)`` -- the piece as written in the file. For the 256 byte-fallback ids this is the
  literal ``<0xNN>`` the rule returns (where ``decode`` would merge bytes or give U+FFFD); for the specials it is
  their content.

Rows: every id of the subset fixture; every special; all 256 byte ids; every non-byte id of the **full** vocabulary
(262 144 single-id decodes, batched); one id past the top raises ``UnknownTokenId`` where the oracle has no token.
The detokeniser under test is ``HostDetokenizer.load()`` on the same file. Run::

    ASTRA_CD_TESTS=1 uv run --all-extras --group cd pytest tests/cd/unit/test_canvas_pieces_parity.py -q

Without ``ASTRA_CD_TESTS=1`` nothing here is collected (``tests/cd/conftest.py``), exactly as
``test_host_detok_parity.py``. Regime: CPU, the ``cd`` group per ``uv.lock`` (tokenizers 0.23.2).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from tests.unit._detok_support import PINNED_SHA256, SPECIALS_BY_ID, detok_mod, piece_ids

BYTE_LITERALS = [f"<0x{b:02X}>" for b in range(256)]


@pytest.fixture(scope="module")
def pinned() -> dict[str, Any]:
    from local_astra.decoding import vocab

    raw = vocab.tokenizer_json_path().read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PINNED_SHA256
    return dict(json.loads(raw))


@pytest.fixture(scope="module")
def oracle() -> Any:
    from local_astra.decoding import vocab

    return vocab.load_tokenizer()


@pytest.fixture(scope="module")
def host() -> Any:
    return detok_mod().HostDetokenizer.load()


def pieces(host: Any, ids: list[int]) -> tuple[str, ...]:
    method = getattr(host, "pieces", None)
    assert callable(method), "T-0159 A2: HostDetokenizer.pieces(ids) does not exist yet"
    out = method(ids)
    assert isinstance(out, tuple) and len(out) == len(ids), "T-0159 A2: one str per id, a tuple"
    return tuple(out)


def _byte_ids(oracle: Any) -> list[int]:
    ids = [oracle.token_to_id(lit) for lit in BYTE_LITERALS]
    assert all(i is not None for i in ids), "premise: the pinned vocab carries all 256 byte pieces"
    return [int(i) for i in ids]


def test_every_subset_id_is_the_oracles_piece(host: Any, oracle: Any) -> None:
    """Every id of ``vocab_subset_cc8d3a0c.json``: a byte id is ``id_to_token`` (its literal); any other id is
    the oracle's single-id decode."""
    ids = sorted(piece_ids().values())
    byte_ids = set(_byte_ids(oracle))
    got = pieces(host, ids)
    for i, p in zip(ids, got, strict=True):
        want = (
            oracle.id_to_token(i)
            if i in byte_ids
            else oracle.decode([i], skip_special_tokens=False)
        )
        assert p == want, f"id {i}: {p!r} != {want!r}"


def test_every_special_is_its_literal(host: Any, oracle: Any) -> None:
    ids = [i for i, _ in SPECIALS_BY_ID]
    assert len(ids) == 24
    want = tuple(oracle.id_to_token(i) for i in ids)
    assert want == tuple(lit for _, lit in SPECIALS_BY_ID), (
        "premise: the support table is the file's"
    )
    assert pieces(host, ids) == want
    assert want == tuple(oracle.decode([i], skip_special_tokens=False) for i in ids)


def test_all_256_byte_ids_are_their_literals_not_their_bytes(host: Any, oracle: Any) -> None:
    """A2 adversarial: ``decode`` turns ``<0x41>`` into ``A`` and ``<0xE2>`` into U+FFFD; the piece is the literal."""
    ids = _byte_ids(oracle)
    assert oracle.decode([ids[0x41]], skip_special_tokens=False) == "A", (
        "premise: decode resolves the byte"
    )
    assert pieces(host, ids) == tuple(BYTE_LITERALS)
    assert tuple(oracle.id_to_token(i) for i in ids) == tuple(BYTE_LITERALS)


def test_every_non_byte_id_of_the_full_vocab_is_the_oracles_single_id_decode(
    host: Any, oracle: Any, pinned: dict[str, Any]
) -> None:
    """The whole pinned vocabulary (model vocab and added tokens), byte ids excluded."""
    byte_ids = set(_byte_ids(oracle))
    every = {int(i) for i in pinned["model"]["vocab"].values()}
    every |= {int(a["id"]) for a in pinned["added_tokens"]}
    ids = sorted(every - byte_ids)
    assert len(ids) > 250_000, f"premise: the full vocab ({len(ids)})"
    want = oracle.decode_batch([[i] for i in ids], skip_special_tokens=False)
    got = pieces(host, ids)
    bad = [(i, g, w) for i, g, w in zip(ids, got, want, strict=True) if g != w]
    assert not bad, f"{len(bad)} ids differ from the oracle, e.g. {bad[:5]}"


def test_an_id_past_the_top_raises_where_the_oracle_has_no_token(host: Any, oracle: Any) -> None:
    top = 262144
    assert oracle.id_to_token(top) is None, "premise: the oracle has no token at this id"
    with pytest.raises(detok_mod().UnknownTokenId):
        pieces(host, [top])
