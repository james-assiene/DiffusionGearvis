"""T-0159 A2: ``HostDetokenizer.pieces(ids) -> tuple[str, ...]``, one display piece per position.

Every expected string is written by hand from the vocab subset fixture
(``tests/fixtures/llm/vocab_subset_cc8d3a0c.json``, the pinned file's own entries; its MANIFEST and
``tests/cd/unit/test_host_detok_parity.py`` tie it to the pinned ``tokenizer.json``), never from an
implementation. The rule (docs/tasks/T-0159.md A2):

* an ordinary vocab piece, with every ``▁`` replaced by a space (no stripping);
* an added token -- the 24 specials included -- verbatim;
* a byte-fallback piece as its own literal (``<0xE2>``), never merged with its neighbours;
* an id outside the vocabulary raises ``UnknownTokenId`` (a negative id included);
* for ids with no byte-fallback piece, ``"".join(pieces(ids)) == decode(ids)`` (Hypothesis);
* ``decode``/``stream``/``load`` unchanged: ``pieces`` shares no state with a stream.

The cd-group differential against ``tokenizers`` over the pinned file is
``tests/cd/unit/test_canvas_pieces_parity.py``. Regime: CPU, stdlib only, the subset fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.toolcall import detok
from tests.unit._detok_support import (
    FFFD,
    HAND_ROWS,
    SPECIALS_BY_ID,
    ids_of,
    piece_ids,
    run_stream,
    subset_detokenizer,
)


def pieces(det: Any, ids: Any) -> tuple[str, ...]:
    """``det.pieces(ids)``, or an assertion naming T-0159 while the method does not exist."""
    method = getattr(det, "pieces", None)
    assert callable(method), "T-0159 A2: HostDetokenizer.pieces(ids) does not exist yet"
    out = method(ids)
    assert isinstance(out, tuple), f"T-0159 A2: pieces returns a tuple, not {type(out).__name__}"
    assert all(isinstance(p, str) for p in out), f"T-0159 A2: every piece is a str: {out!r}"
    return out


@pytest.fixture(scope="module")
def det(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return subset_detokenizer(tmp_path_factory.mktemp("t0159-subset"))


# --------------------------------------------------------------------------- ordinary pieces


#: (vocab entry as written in the fixture, the display piece by hand)
ORDINARY = (
    ("▁the", " the"),
    ("▁I", " I"),
    ("▁", " "),
    ("▁▁", "  "),
    ("▁▁▁", "   "),
    ("▁<", " <"),
    (".", "."),
    ("A", "A"),
    ("é", "é"),
    ("€", "€"),
    ("tool", "tool"),
    ("_", "_"),
    ("|>", "|>"),
    ("\n", "\n"),
    ("\n\n", "\n\n"),
)


def test_an_ordinary_piece_has_every_word_marker_replaced_by_a_space(det: Any) -> None:
    """A2: ``▁`` -> space, everything else as written, no stripping. One position per id."""
    entries = [entry for entry, _ in ORDINARY]
    got = pieces(det, ids_of(entries))
    assert got == tuple(want for _, want in ORDINARY), got


def test_one_piece_per_id_in_order_with_repeats(det: Any) -> None:
    """A2: one string per id, in the ids' order; a repeated id is two positions, not one."""
    v = piece_ids()
    ids = [v["▁a"], v["▁b"], v["▁a"], v["▁a"], v["."]]
    assert pieces(det, ids) == (" a", " b", " a", " a", ".")
    assert pieces(det, []) == ()
    assert pieces(det, tuple(ids)) == pieces(det, ids), "any sequence of ids"


# --------------------------------------------------------------------------- specials


def test_every_special_is_returned_as_its_literal(det: Any) -> None:
    """A2: the 24 specials verbatim (``_detok_support.SPECIALS_BY_ID``, read off the pinned file)."""
    ids = [i for i, _ in SPECIALS_BY_ID]
    assert len(ids) == 24
    assert pieces(det, ids) == tuple(lit for _, lit in SPECIALS_BY_ID)


def test_a_thought_block_keeps_its_markers_as_their_own_positions(det: Any) -> None:
    """A2 / ruling 1 (all tokens, thought included): ``<|channel>`` ``thought`` ``\\n`` ...
    ``<channel|>`` is one position per token, the markers verbatim."""
    v = piece_ids()
    ids = [100, v["▁the"], v["\n"], 101, 1]
    assert pieces(det, ids) == ("<|channel>", " the", "\n", "<channel|>", "<eos>")


# --------------------------------------------------------------------------- byte fallback


def test_a_byte_run_is_three_literals_never_the_euro_sign(det: Any) -> None:
    """A2 adversarial (the failure mode: merging a byte run as ``decode`` does): ``<0xE2><0x82>
    <0xAC>`` is ``€`` to ``decode`` but three positions here, each its own literal."""
    ids = ids_of(["<0xE2>", "<0x82>", "<0xAC>"])
    assert det.decode(ids) == "€", "premise: decode merges the run"
    assert pieces(det, ids) == ("<0xE2>", "<0x82>", "<0xAC>")


def test_an_ascii_byte_piece_is_its_literal_not_the_character(det: Any) -> None:
    """A2: ``<0x41>`` decodes to ``A`` but its piece is ``<0x41>``; ``<0x0A>`` is not a newline."""
    ids = ids_of(["<0x41>", "<0x0A>", "<0xFF>"])
    assert det.decode(ids[:1]) == "A", "premise"
    assert pieces(det, ids) == ("<0x41>", "<0x0A>", "<0xFF>")


def test_byte_literals_between_ordinary_pieces_and_specials_stay_apart(det: Any) -> None:
    """A2: a truncated run (``decode`` gives U+FFFD per byte) and a run split by a special are
    still one literal per position; their neighbours are unaffected."""
    entries = ["▁I", "<0xE2>", "<0x82>", "<|tool_call>", "<0xAC>", "."]
    ids = ids_of(entries)
    assert FFFD in det.decode(ids), "premise: decode replaces the broken run"
    assert pieces(det, ids) == (" I", "<0xE2>", "<0x82>", "<|tool_call>", "<0xAC>", ".")


def test_all_256_byte_pieces_are_their_literals(det: Any) -> None:
    """A2: every ``<0xNN>`` entry of the fixture (all 256 are in it) is returned as written."""
    v = piece_ids()
    literals = [f"<0x{b:02X}>" for b in range(256)]
    assert all(lit in v for lit in literals), "premise: the subset carries all 256 byte pieces"
    assert pieces(det, [v[lit] for lit in literals]) == tuple(literals)


# --------------------------------------------------------------------------- unknown ids


@pytest.mark.parametrize(
    "bad",
    [5, 99, 258885, 10**9, -1, -2],
    ids=["gap-5", "gap-99", "one-past-the-top", "huge", "minus-one", "minus-two"],
)
def test_an_id_outside_the_vocabulary_raises(det: Any, bad: int) -> None:
    """A2: ``UnknownTokenId``, never a guess. Gaps of the subset (5, 99), one past its top id
    (258884 is ``<|video|>``) and negative ids (a list index of -1 would silently answer)."""
    v = piece_ids()
    assert bad not in v.values(), "premise: the id is not in the subset"
    method = getattr(det, "pieces", None)
    assert callable(method), "T-0159 A2: HostDetokenizer.pieces(ids) does not exist yet"
    with pytest.raises(detok.UnknownTokenId):
        method([v["▁the"], bad])


# --------------------------------------------------------------------------- join == decode


def _non_byte_ids() -> list[int]:
    return sorted(i for p, i in piece_ids().items() if detok._byte_of(p) is None)


@settings(
    max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(data=st.data())
def test_joined_pieces_equal_decode_for_ids_without_byte_pieces(det: Any, data: Any) -> None:
    """A2 property: no byte-fallback id -> ``"".join(pieces(ids)) == decode(ids)``, and there is
    one piece per id. Ids drawn from every non-byte entry of the subset, specials included."""
    pool = _non_byte_ids()
    assert 1 in pool and 100 in pool and piece_ids()["▁"] in pool, "premise: the pool"
    ids = data.draw(st.lists(st.sampled_from(pool), max_size=40))
    got = pieces(det, ids)
    assert len(got) == len(ids)
    assert "".join(got) == det.decode(ids)


# --------------------------------------------------------------------------- decode/stream unchanged


@pytest.mark.parametrize(("name", "entries", "text"), HAND_ROWS, ids=[r[0] for r in HAND_ROWS])
def test_decode_is_unchanged_on_the_t0088a_hand_rows(
    det: Any, name: str, entries: Any, text: str
) -> None:
    """A2: ``decode`` keeps every T-0088a hand row (byte runs merged there, U+FFFD rules intact)
    after ``pieces`` was called on the same ids."""
    ids = ids_of(entries)
    pieces(det, ids)
    assert det.decode(ids) == text, name


def test_pieces_shares_no_state_with_an_open_stream(det: Any) -> None:
    """A2 adversarial: ``pieces`` is a pure per-id lookup. A stream holding the open run
    ``<0xE2><0x82>`` still completes ``€`` after ``pieces`` was called on other bytes between its
    feeds (a shared byte buffer would corrupt it)."""
    first, last = ids_of(["▁I", "<0xE2>", "<0x82>"]), ids_of(["<0xAC>", "."])
    s = det.stream()
    out = s.feed(first)
    assert pieces(det, ids_of(["<0xFF>", "<0x41>", "<0xC3>"])) == ("<0xFF>", "<0x41>", "<0xC3>")
    out += s.feed(last) + s.finish()
    assert out == " I€.", out
    outs, tail = run_stream(det, [first, last])
    assert "".join(outs) + tail == " I€."


def test_load_is_unchanged_and_returns_an_object_with_pieces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A2: ``load`` keeps its contract (the pinned path, a named ``FileNotFoundError`` when it is
    absent) and the loaded object is a piece source."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    with pytest.raises(FileNotFoundError) as info:
        detok.HostDetokenizer.load()
    assert "tokenizer.json" in str(info.value)
    assert callable(getattr(detok.HostDetokenizer, "pieces", None)), (
        "T-0159 A2: HostDetokenizer.pieces does not exist yet"
    )
