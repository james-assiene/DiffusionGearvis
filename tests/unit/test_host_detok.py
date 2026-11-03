"""T-0088a criterion 2: the stdlib host detokeniser of the pinned ``tokenizer.json``.

Fixture: ``tests/fixtures/llm/vocab_subset_cc8d3a0c.json`` (562 pieces of the pinned file: the 24 specials, the
256 byte-fallback pieces, every id of the 24 T-0087 arm-(b)/(e) streams, a few ``▁`` and fragment pieces), written
to a temporary ``tokenizer.json`` and loaded through the explicit-path constructor. Expected strings are either
the oracle text frozen in the fixture (``tokenizers`` 0.23.2 on the pinned file) or hand-computed from HF
``ByteFallback`` semantics and re-proved against the oracle in ``tests/cd/unit/test_host_detok_parity.py``.

Rows:
- loading: the path rule of ``load()`` (HF_HUB_CACHE / HF_HOME), and each refusal — wrong sha, model not BPE or
  without ``byte_fallback``, a decoder other than ``Sequence[Replace(▁→ ), ByteFallback, Fuse]``, special added
  tokens other than ``SPECIAL_TOKENS`` — raising ``ValueError`` naming the file; no socket is opened;
- decoding: the 24 recorded streams; the hand rows (2/3/4-byte characters, invalid continuation, a run truncated at
  ``finish()``, a special right after and inside a byte run, ``▁`` runs);
- incremental decoding: every partition of every hand row, the hold of a byte run across a boundary, a special id
  emitted whole by the ``feed`` that carries it, and a Hypothesis partition property over the subset;
- ``UnknownTokenId`` from ``decode`` and from ``feed``;
- **adversarial**: a decoder without the byte hold (each chunk decoded on its own) fails the split-character rows;
  one that re-decodes the whole prefix on every ``feed`` fails the linear-time row (4096 ids, one per ``feed``,
  under 50 ms total, median of 5; regime: CPU, fast suite serial or ``make fast-parallel``).
"""

from __future__ import annotations

import itertools
import json
import socket
import statistics
import time
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.unit._detok_support import (
    FFFD,
    HAND_ROWS,
    MODEL_ID,
    MODEL_REVISION,
    SPECIALS_BY_ID,
    detok_mod,
    fixture,
    ids_of,
    partitions,
    piece_ids,
    run_stream,
    subset_detokenizer,
    write_tokenizer,
)

HAND = {name: (pieces, expected) for name, pieces, expected in HAND_ROWS}
SPLIT_CHAR_ROWS = ("two_byte_char", "three_byte_char", "four_byte_char")


class _Deferred:
    """Builds the detokeniser on first use, inside the row, so a missing module *fails* the row (an assertion
    naming it) instead of erroring its setup. Every attribute is delegated to the real object."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._real: Any = None

    def __getattr__(self, name: str) -> Any:
        if self._real is None:
            self._real = subset_detokenizer(self._directory)
        return getattr(self._real, name)


@pytest.fixture
def det(tmp_path: Path) -> Any:
    return _Deferred(tmp_path / "subset")


def _refusal(path: Path, sha: str) -> str:
    """Construct over ``path`` and return the refusal message (the row fails if nothing is raised)."""
    mod = detok_mod()
    with pytest.raises(ValueError) as info:
        mod.HostDetokenizer(path, sha)
    msg = str(info.value)
    assert str(path) in msg, f"the refusal must name the file {path}: {msg!r}"
    return msg


# --------------------------------------------------------------------------- loading


def test_the_subset_loads_through_the_explicit_path_constructor(tmp_path: Path) -> None:
    det = subset_detokenizer(tmp_path)
    assert det.decode(ids_of(["▁I", "▁the"])) == " I the"


def test_a_wrong_sha_is_refused_and_named(tmp_path: Path) -> None:
    path, _ = write_tokenizer(tmp_path)
    wrong = "0" * 64
    msg = _refusal(path, wrong)
    assert "sha" in msg.lower(), f"the reason must say it is the sha256: {msg!r}"


def test_the_pinned_sha_is_refused_for_a_file_that_is_not_the_pinned_one(tmp_path: Path) -> None:
    """The explicit constructor checks the bytes, not the caller's word: the subset is not ``cc8d3a0c…``."""
    path, _ = write_tokenizer(tmp_path)
    _refusal(path, "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f")


def _drop_fuse(tok: dict[str, Any]) -> None:
    tok["decoder"]["decoders"] = tok["decoder"]["decoders"][:2]


def _swap_order(tok: dict[str, Any]) -> None:
    d = tok["decoder"]["decoders"]
    d[0], d[1] = d[1], d[0]


def _replace_content(tok: dict[str, Any]) -> None:
    tok["decoder"]["decoders"][0]["content"] = "_"


def _replace_pattern(tok: dict[str, Any]) -> None:
    tok["decoder"]["decoders"][0]["pattern"] = {"String": "_"}


def _extra_strip(tok: dict[str, Any]) -> None:
    tok["decoder"]["decoders"].append({"type": "Strip", "content": " ", "start": 1, "stop": 0})


def _not_a_sequence(tok: dict[str, Any]) -> None:
    tok["decoder"] = {"type": "ByteFallback"}


def _metaspace(tok: dict[str, Any]) -> None:
    tok["decoder"] = {
        "type": "Metaspace",
        "replacement": "▁",
        "prepend_scheme": "always",
        "split": True,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        _drop_fuse,
        _swap_order,
        _replace_content,
        _replace_pattern,
        _extra_strip,
        _not_a_sequence,
        _metaspace,
    ],
)
def test_a_decoder_other_than_replace_bytefallback_fuse_is_refused(
    tmp_path: Path, mutate: Any
) -> None:
    path, sha = write_tokenizer(tmp_path, mutate)
    msg = _refusal(path, sha)
    assert "decoder" in msg.lower(), f"the reason must name the decoder: {msg!r}"


def _unigram(tok: dict[str, Any]) -> None:
    tok["model"]["type"] = "Unigram"


def _no_byte_fallback(tok: dict[str, Any]) -> None:
    tok["model"]["byte_fallback"] = False


def _byte_fallback_absent(tok: dict[str, Any]) -> None:
    del tok["model"]["byte_fallback"]


@pytest.mark.parametrize("mutate", [_unigram, _no_byte_fallback, _byte_fallback_absent])
def test_a_model_that_is_not_bpe_with_byte_fallback_is_refused(tmp_path: Path, mutate: Any) -> None:
    path, sha = write_tokenizer(tmp_path, mutate)
    msg = _refusal(path, sha)
    assert any(w in msg.lower() for w in ("bpe", "byte_fallback", "byte fallback", "model")), msg


def _drop_one_special(tok: dict[str, Any]) -> None:
    tok["added_tokens"] = [a for a in tok["added_tokens"] if a["content"] != "<|tool_call>"]


def _unflag_one_special(tok: dict[str, Any]) -> None:
    for a in tok["added_tokens"]:
        if a["content"] == "<channel|>":
            a["special"] = False


def _rename_one_special(tok: dict[str, Any]) -> None:
    for a in tok["added_tokens"]:
        if a["content"] == "<turn|>":
            a["content"] = "<end_of_turn>"


def _move_one_special(tok: dict[str, Any]) -> None:
    for a in tok["added_tokens"]:
        if a["content"] == "<|tool_response>":
            a["id"] = 53


def _add_a_special(tok: dict[str, Any]) -> None:
    tok["added_tokens"].append(
        {
            "id": piece_ids()["tool"],
            "content": "tool",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )


@pytest.mark.parametrize(
    "mutate",
    [
        _drop_one_special,
        _unflag_one_special,
        _rename_one_special,
        _move_one_special,
        _add_a_special,
    ],
)
def test_special_added_tokens_other_than_the_frozen_table_are_refused(
    tmp_path: Path, mutate: Any
) -> None:
    path, sha = write_tokenizer(tmp_path, mutate)
    msg = _refusal(path, sha)
    assert "special" in msg.lower(), f"the reason must name the special tokens: {msg!r}"


def _pinned_snapshot(root: Path) -> Path:
    return root / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / MODEL_REVISION


@pytest.mark.parametrize("env", ["HF_HUB_CACHE", "HF_HOME"])
def test_load_reads_the_pinned_revision_by_the_hf_cache_path_rule_and_checks_its_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: str
) -> None:
    """``load()`` finds ``tokenizer.json`` where ``decoding.vocab.snapshot_dir`` does, and refuses it because it
    is not ``cc8d3a0c…`` (the subset is a different file). No socket may open while it does so."""
    mod = detok_mod()
    base = tmp_path / "hf"
    hub = base if env == "HF_HUB_CACHE" else base / "hub"
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv(env, str(base))
    path, _ = write_tokenizer(_pinned_snapshot(hub))

    def no_network(*a: Any, **k: Any) -> Any:
        raise AssertionError("HostDetokenizer.load() opened a socket")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    with pytest.raises(ValueError) as info:
        mod.HostDetokenizer.load()
    assert str(path) in str(info.value), f"load() must name the file it refused: {info.value}"
    assert "sha" in str(info.value).lower()


def test_load_with_no_file_in_the_cache_raises_naming_the_expected_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = detok_mod()
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises((OSError, ValueError)) as info:
        mod.HostDetokenizer.load()
    assert str(_pinned_snapshot(tmp_path) / "tokenizer.json") in str(info.value)


# --------------------------------------------------------------------------- decoding


def test_the_24_recorded_streams_decode_to_the_frozen_oracle_text(det: Any) -> None:
    streams = fixture()["streams"]
    assert len(streams) == 24
    bad = [f"{s['case']}/{s['arm']}" for s in streams if det.decode(s["ids"]) != s["text"]]
    assert not bad, f"host decode differs from the tokenizers decode on {bad}"


def test_the_24_recorded_streams_fed_one_id_at_a_time_give_the_same_text(det: Any) -> None:
    for s in fixture()["streams"]:
        outs, tail = run_stream(det, [[i] for i in s["ids"]])
        assert "".join(outs) + tail == s["text"], f"{s['case']}/{s['arm']}"


@pytest.mark.parametrize(("name", "pieces", "expected"), HAND_ROWS, ids=[r[0] for r in HAND_ROWS])
def test_decode_matches_the_hand_rows(
    det: Any, name: str, pieces: tuple[str, ...], expected: str
) -> None:
    assert det.decode(ids_of(pieces)) == expected


def test_an_invalid_run_renders_one_replacement_per_byte_not_python_replace(det: Any) -> None:
    """HF ByteFallback: ``0x41 0xFF`` is two U+FFFD (the ``A`` is lost too); Python's ``"replace"`` gives ``A�``."""
    ids = ids_of(["<0x41>", "<0xFF>"])
    assert det.decode(ids) == FFFD * 2
    assert det.decode(ids) != b"\x41\xff".decode("utf-8", "replace")


def test_specials_are_kept_never_skipped(det: Any) -> None:
    ids = [i for i, _ in SPECIALS_BY_ID]
    assert det.decode(ids) == "".join(lit for _, lit in SPECIALS_BY_ID)


# --------------------------------------------------------------------------- incremental decoding


@pytest.mark.parametrize(("name", "pieces", "expected"), HAND_ROWS, ids=[r[0] for r in HAND_ROWS])
def test_every_partition_of_every_hand_row_concatenates_to_the_decode(
    det: Any, name: str, pieces: tuple[str, ...], expected: str
) -> None:
    ids = ids_of(pieces)
    for chunks in partitions(ids):
        outs, tail = run_stream(det, chunks)
        assert "".join(outs) + tail == expected, f"partition {chunks} gave {outs!r} + {tail!r}"


@pytest.mark.parametrize("name", SPLIT_CHAR_ROWS)
def test_a_character_split_one_byte_per_feed_never_emits_a_replacement(det: Any, name: str) -> None:
    """Adversarial (no byte hold): each byte of a 2/3/4-byte character in its own ``feed``. A decoder that
    decodes every chunk on its own emits U+FFFD for each lone byte; the hold must emit the character whole."""
    pieces, expected = HAND[name]
    outs, tail = run_stream(det, [[i] for i in ids_of(pieces)])
    assert FFFD not in "".join(outs) + tail, f"{outs!r} + {tail!r}"
    assert "".join(outs) + tail == expected


@pytest.mark.parametrize("name", SPLIT_CHAR_ROWS)
def test_a_character_split_at_every_boundary_of_its_run(det: Any, name: str) -> None:
    pieces, expected = HAND[name]
    ids = ids_of(pieces)
    first_byte = next(k for k, p in enumerate(pieces) if p.startswith("<0x"))
    last_byte = max(k for k, p in enumerate(pieces) if p.startswith("<0x"))
    for cut in range(first_byte + 1, last_byte + 1):
        outs, tail = run_stream(det, [ids[:cut], ids[cut:]])
        assert "".join(outs) + tail == expected, f"cut at {cut}: {outs!r} + {tail!r}"


def test_a_byte_run_is_held_until_a_non_byte_piece_ends_it(det: Any) -> None:
    """A complete, valid ``€`` run may still be invalidated by the next byte (HF renders the whole run as
    U+FFFD per byte), so nothing of an open run may be emitted before the run ends."""
    s = det.stream()
    assert s.feed(ids_of(["▁I"])) == " I"
    assert s.feed(ids_of(["<0xE2>", "<0x82>", "<0xAC>"])) == "", "an open byte run must be held"
    assert s.feed(ids_of(["<0xFF>"])) == ""
    assert s.feed(ids_of(["<channel|>"])) == FFFD * 4 + "<channel|>"
    assert s.finish() == ""


def test_a_valid_run_is_released_by_the_next_piece(det: Any) -> None:
    s = det.stream()
    assert s.feed(ids_of(["<0xE2>", "<0x82>"])) == ""
    assert s.feed(ids_of(["<0xAC>"])) == ""
    assert s.feed(ids_of(["<tool_call|>"])) == "€<tool_call|>"
    assert s.finish() == ""


def test_a_run_truncated_at_finish_is_flushed_by_finish(det: Any) -> None:
    s = det.stream()
    assert s.feed(ids_of(["▁I", "<0xE2>", "<0x82>"])) == " I"
    assert s.finish() == FFFD * 2


@pytest.mark.parametrize("sid", [i for i, _ in SPECIALS_BY_ID])
def test_a_special_id_is_emitted_whole_by_the_feed_that_carries_it(det: Any, sid: int) -> None:
    lit = dict(SPECIALS_BY_ID)[sid]
    s = det.stream()
    assert s.feed(ids_of(["▁I"])) == " I"
    assert s.feed([sid]) == lit, "a special is one piece: never split, never held"
    assert s.finish() == ""


_SUBSET = fixture()
_VOCAB_IDS = sorted(_SUBSET["tokenizer"]["model"]["vocab"].values())
_BYTE_IDS = sorted(
    v for p, v in _SUBSET["tokenizer"]["model"]["vocab"].items() if p.startswith("<0x")
)
_SPECIAL_IDS = [i for i, _ in SPECIALS_BY_ID]
_ids = st.one_of(
    st.sampled_from(_BYTE_IDS), st.sampled_from(_SPECIAL_IDS), st.sampled_from(_VOCAB_IDS)
)


@settings(
    max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(ids=st.lists(_ids, min_size=0, max_size=40), cuts=st.lists(st.integers(0, 40), max_size=12))
def test_feed_finish_concatenation_equals_decode_for_any_partition(
    det: Any, ids: list[int], cuts: list[int]
) -> None:
    """Hypothesis, 300 examples, ids drawn 1:1:1 from byte-fallback pieces, specials and the whole subset."""
    bounds = sorted({c for c in cuts if 0 < c < len(ids)})
    edges = [0, *bounds, len(ids)]
    chunks = [ids[a:b] for a, b in itertools.pairwise(edges)]
    outs, tail = run_stream(det, chunks)
    assert "".join(outs) + tail == det.decode(ids)


def test_an_empty_feed_emits_nothing_and_breaks_no_run(det: Any) -> None:
    s = det.stream()
    out = s.feed(ids_of(["<0xC3>"])) + s.feed([]) + s.feed(ids_of(["<0xA9>"])) + s.finish()
    assert out == "é"


# --------------------------------------------------------------------------- unknown ids


@pytest.mark.parametrize("bad", [262144, 10**9, -1])
def test_an_id_outside_the_vocabulary_raises_unknown_token_id(det: Any, bad: int) -> None:
    mod = detok_mod()
    with pytest.raises(mod.UnknownTokenId):
        det.decode([*ids_of(["▁I"]), bad])
    with pytest.raises(mod.UnknownTokenId):
        det.stream().feed([bad])


def test_an_id_missing_from_the_loaded_file_raises_unknown_token_id(det: Any) -> None:
    """The subset lacks id 5: the detokeniser never guesses a piece it was not given (the oracle drops it)."""
    assert 5 not in set(_VOCAB_IDS)
    mod = detok_mod()
    with pytest.raises(mod.UnknownTokenId):
        det.decode([5])


# --------------------------------------------------------------------------- linear time


def _linear_time_ids() -> list[int]:
    """4096 ids: the 24 streams back to back (specials, prose pieces) with a 3-byte run between them."""
    euro = ids_of(["<0xE2>", "<0x82>", "<0xAC>"])
    out: list[int] = []
    while len(out) < 4096:
        for s in fixture()["streams"]:
            out.extend(s["ids"])
            out.extend(euro)
    return out[:4096]


def test_4096_ids_fed_one_per_feed_take_under_50_ms(det: Any) -> None:
    """Adversarial (re-decoding the whole prefix on each ``feed``). Median of 5 runs < 50 ms; CPU; the output
    must also equal ``decode`` so a fast but wrong stream cannot pass."""
    ids = _linear_time_ids()
    expected = det.decode(ids)
    times = []
    for _ in range(5):
        s = det.stream()
        t0 = time.perf_counter()
        parts = [s.feed([i]) for i in ids]
        parts.append(s.finish())
        times.append(time.perf_counter() - t0)
        assert "".join(parts) == expected
    med = statistics.median(times)
    assert med < 0.050, (
        f"4096 one-id feeds took {med * 1000:.1f} ms (median of 5; runs {json.dumps([round(t * 1000, 1) for t in times])})"
    )


# --------------------------------------------------------------------------- rows after review (eeee0ab)


@pytest.mark.parametrize(
    "pieces",
    [
        ("<0xC3>", "▁I"),  # the ruling's row: [byte, word, bad]
        ("▁I", "<0xC3>"),  # a byte left open by the failed call would leak into the next one
        ("<0xC3>",),
    ],
)
def test_a_feed_with_an_unknown_id_raises_and_changes_nothing(
    det: Any, pieces: tuple[str, ...]
) -> None:
    """Ruling "atomic feed": ``feed`` resolves every id before it touches the stream, so a refused call leaves the
    stream exactly as it was. A one-pass ``feed`` (append, then fail) leaves a byte held (rows 2 and 3)."""
    mod = detok_mod()
    s = det.stream()
    with pytest.raises(mod.UnknownTokenId):
        s.feed([*ids_of(pieces), 262144])
    word = ids_of(["▁I"])
    assert s.feed(word) + s.finish() == det.decode(word)


def test_a_failed_feed_keeps_the_run_held_before_it(det: Any) -> None:
    """A run held from an earlier call survives a refused call untouched (a one-pass ``feed`` flushes it into the
    lost output of the failed call)."""
    mod = detok_mod()
    s = det.stream()
    assert s.feed(ids_of(["<0xC3>"])) == ""
    with pytest.raises(mod.UnknownTokenId):
        s.feed([*ids_of(["<0xA9>", "▁I"]), 262144])
    assert s.feed(ids_of(["<0xA9>"])) + s.finish() == "é"


def test_finish_twice_returns_nothing_the_second_time(det: Any) -> None:
    s = det.stream()
    assert s.feed(ids_of(["▁I", "<0xE2>", "<0x82>"])) == " I"
    assert s.finish() == FFFD * 2
    assert s.finish() == ""
