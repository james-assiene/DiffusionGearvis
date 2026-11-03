"""T-0080a criterion 6: ``compile_bfcl``/``compile_sudoku`` payloads are byte-identical before and after.

The pins are the sha256 of ``artifact.encode_automaton(...)`` (the wire string the plugin receives, so the
arrays, the class tables and the stamped cache key are all covered), measured at 505ebc4 -- the fork point
of ``T-0080a-toolcall-grammars``, before any T-0080a code existed -- with uncached compiles (CPU;
outlines-core per uv.lock; tokenizer.json sha256 cc8d3a0c...; model rev 2ea8372...; the same digests under
``PYTHONHASHSEED=123``). They are "before" values for a no-change criterion, not current behaviour of the
code under test; T-0080a must leave them as they are. Green today by design; the existing parity rows
(``test_grammar_parity_jax.py``) stay unedited beside them.

Records: the first two ids of the BFCL test cut (``exp_e5_grammar130_map`` order) under the baseline's
pretty whitespace and under JSON whitespace (T-0062 decision 11), and the first two Sudoku test-cut puzzles.
"""

from __future__ import annotations

import hashlib

import pytest

from tests.cd._compile_support import bfcl_function, fresh_compile, need, sudoku_cut

#: (record id, whitespace) -> sha256 of the encoded payload at 505ebc4.
BFCL_PINS = {
    (
        "live_simple_0-0-0",
        "pretty",
    ): "494c69be2b92adab8f7fe56e3b4849803b60c44ed579a7b1082c78cf57611fa3",
    (
        "live_simple_0-0-0",
        "json",
    ): "f0c591fcc5efd1450f25d5c032d8bf2614fbad8b4e36fd4652f2c3f4183eb4f8",
    (
        "live_simple_1-1-0",
        "pretty",
    ): "3a3cbd88a1338ef04fc9bb37b14c550e48a01db43354539263325b1cbb0b54a0",
    (
        "live_simple_1-1-0",
        "json",
    ): "088e03ae085bad24ac33fdec503bd7fe4ab82281e1d91df10e4546f4fc815f83",
}
#: Sudoku test-cut id -> (puzzle as in the fixture, sha256 of the encoded payload at 505ebc4).
SUDOKU_PINS = {
    "sudoku4_0": (
        ((3, 1, 2, 4), (0, 0, 0, 1), (2, 4, 0, 0), (1, 0, 4, 2)),
        "d8b3f8f89815ae1d41d3772410d6bc9c1865b5ec578db89d40728d5830f6882f",
    ),
    "sudoku4_1": (
        ((2, 1, 4, 3), (0, 3, 2, 0), (0, 0, 1, 4), (1, 4, 0, 0)),
        "4366dec9a7439646e441518b8daef5180f209d0baade73962a6042b681047dd7",
    ),
}


def _digest(automaton: object) -> str:
    artifact = need("local_astra.decoding.artifact")
    return hashlib.sha256(artifact.encode_automaton(automaton).encode("ascii")).hexdigest()


@pytest.mark.parametrize(("rid", "ws"), sorted(BFCL_PINS))
def test_compile_bfcl_payload_is_byte_identical(rid: str, ws: str) -> None:
    overrides = {} if ws == "pretty" else {"whitespace": "json"}
    a = fresh_compile("bfcl", bfcl_function(rid), **overrides)
    assert _digest(a) == BFCL_PINS[(rid, ws)]


@pytest.mark.parametrize("rid", sorted(SUDOKU_PINS))
def test_compile_sudoku_payload_is_byte_identical(rid: str) -> None:
    puzzle, pin = SUDOKU_PINS[rid]
    fixture = {r: p for r, p, _ in sudoku_cut("test")}
    assert fixture[rid] == puzzle, "the Sudoku fixture moved; the pin no longer names this puzzle"
    assert _digest(fresh_compile("sudoku", puzzle)) == pin
