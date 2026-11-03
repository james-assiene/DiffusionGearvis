"""T-0088a criteria 1 and 2, the cd-group differential: the stdlib host detokeniser equals ``tokenizers``.

Oracle: ``tokenizers.Tokenizer.decode(ids, skip_special_tokens=False)`` on the pinned ``tokenizer.json``
(``cc8d3a0c…``, rev ``2ea8372…``) read from the local HF cache through ``local_astra.decoding.vocab``. The detokeniser
under test is ``HostDetokenizer.load()`` on the same file. Run::

    ASTRA_CD_TESTS=1 uv run --all-extras --group cd pytest tests/cd/unit/test_host_detok_parity.py -q

Rows:
- ``load()`` reads the pinned file with no socket opened, and ``decode`` equals the oracle on every recorded stream
  (the 24 arm-(b)/(e) streams of T-0087, from the subset fixture) and on every hand row of
  ``tests/unit/_detok_support.py`` (which also proves those hand-computed expectations against the oracle);
- Hypothesis over the **full** vocabulary, ids weighted toward the 256 byte-fallback pieces, the 24 specials and
  ``▁`` pieces: ``decode`` equals the oracle, and so does the ``feed``/``finish`` concatenation of a random partition;
- ``SPECIAL_TOKENS``/``SPECIAL_IDS`` equal the file's ``special: true`` added tokens in id order; ``STOP_IDS`` equals
  ``vocab.stop_token_ids()`` and ``replay.STOP_IDS``;
- the subset fixture: every vocab entry, the decoder/model config, the added tokens and each stream's frozen text
  equal the pinned file and the oracle; when the untracked T-0087 ``raw.jsonl`` is present in the main checkout,
  each stream's ids equal its recorded ``token_ids``.

Regime: CPU, the ``cd`` group per ``uv.lock`` (tokenizers 0.23.2). The oracle silently drops an id outside the
vocabulary; the host raises ``UnknownTokenId`` (criterion 2), so the property draws in-vocabulary ids only and the
divergence is pinned by its own row.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.unit._detok_support import (
    HAND_ROWS,
    PINNED_SHA256,
    detok_mod,
    fixture,
    ids_of,
    run_stream,
    special_mod,
)

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def pinned() -> dict[str, Any]:
    from local_astra.decoding import vocab

    path = vocab.tokenizer_json_path()
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PINNED_SHA256
    return {"path": path, "json": json.loads(raw)}


@pytest.fixture(scope="module")
def oracle() -> Any:
    from local_astra.decoding import vocab

    return vocab.load_tokenizer()


class _DeferredHost:
    """``HostDetokenizer.load()`` on first use, inside the row (a missing module fails the row, it does not
    error the fixture), with ``socket.socket.connect`` refused while it loads. Every attribute is delegated."""

    def __init__(self) -> None:
        self._real: Any = None

    def __getattr__(self, name: str) -> Any:
        if self._real is None:
            real_connect = socket.socket.connect

            def no_network(*a: Any, **k: Any) -> Any:
                raise AssertionError("HostDetokenizer.load() opened a socket")

            socket.socket.connect = no_network  # type: ignore[method-assign]
            try:
                self._real = detok_mod().HostDetokenizer.load()
            finally:
                socket.socket.connect = real_connect  # type: ignore[method-assign]
        return getattr(self._real, name)


@pytest.fixture(scope="module")
def host() -> Any:
    return _DeferredHost()


def _odec(oracle: Any, ids: list[int]) -> str:
    text: str = oracle.decode(ids, skip_special_tokens=False)
    return text


# --------------------------------------------------------------------------- recorded streams and hand rows


def test_every_recorded_stream_decodes_like_the_oracle(host: Any, oracle: Any) -> None:
    streams = fixture()["streams"]
    assert len(streams) == 24
    for s in streams:
        assert host.decode(s["ids"]) == _odec(oracle, s["ids"]), f"{s['case']}/{s['arm']}"
        outs, tail = run_stream(host, [[i] for i in s["ids"]])
        assert "".join(outs) + tail == _odec(oracle, s["ids"]), f"{s['case']}/{s['arm']} streamed"


@pytest.mark.parametrize(("name", "pieces", "expected"), HAND_ROWS, ids=[r[0] for r in HAND_ROWS])
def test_the_unit_hand_rows_expect_what_the_oracle_decodes(
    oracle: Any, name: str, pieces: tuple[str, ...], expected: str
) -> None:
    """Fixture proof (green before implementation): ``tests/unit/test_host_detok.py``'s hand-computed strings are
    exactly the ``tokenizers`` decode, so the unit rows do not rest on the test-author's reading of HF."""
    assert _odec(oracle, ids_of(pieces)) == expected


@pytest.mark.parametrize(("name", "pieces", "expected"), HAND_ROWS, ids=[r[0] for r in HAND_ROWS])
def test_the_hand_rows_are_the_oracles_decode(
    host: Any, oracle: Any, name: str, pieces: tuple[str, ...], expected: str
) -> None:
    ids = ids_of(pieces)
    assert _odec(oracle, ids) == expected, (
        "the unit row's hand-computed expectation disagrees with tokenizers"
    )
    assert host.decode(ids) == expected


def test_the_oracle_drops_unknown_ids_and_the_host_refuses_them(host: Any, oracle: Any) -> None:
    """The one stated divergence: ``tokenizers`` skips an unknown id silently; the host never guesses."""
    a = ids_of(["<0x41>"])[0]
    assert _odec(oracle, [a, 262144, a]) == "AA"
    with pytest.raises(detok_mod().UnknownTokenId):
        host.decode([a, 262144, a])


# --------------------------------------------------------------------------- Hypothesis over the full vocab

_SPECIAL_IDS = [0, 1, 2, 3, 4, 46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105, 106]
_SPECIAL_IDS += [255999, 256000, 258880, 258881, 258882, 258883, 258884]


def _strategies(pinned_json: dict[str, Any]) -> Any:
    vocab = pinned_json["model"]["vocab"]
    bytes_ = sorted(
        v for p, v in vocab.items() if len(p) == 6 and p.startswith("<0x") and p.endswith(">")
    )
    spaces = sorted(v for p, v in vocab.items() if "▁" in p)
    n = len(vocab)
    return st.one_of(
        st.sampled_from(bytes_),
        st.sampled_from(bytes_),
        st.sampled_from(_SPECIAL_IDS),
        st.sampled_from(spaces),
        st.integers(0, n - 1),
    )


def _full_vocab_ids() -> Any:
    from local_astra.decoding import vocab

    return _strategies(json.loads(vocab.tokenizer_json_path().read_bytes()))


_IDS = _full_vocab_ids()


@settings(
    max_examples=500, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(data=st.data())
def test_decode_equals_the_oracle_on_random_full_vocab_sequences(
    host: Any, oracle: Any, data: Any
) -> None:
    """500 examples, sequences of 0–64 ids: 2/5 byte-fallback, 1/5 specials, 1/5 ``▁`` pieces, 1/5 uniform."""
    ids = data.draw(st.lists(_IDS, max_size=64))
    assert host.decode(ids) == _odec(oracle, ids)


@settings(
    max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(data=st.data())
def test_any_partition_streams_to_the_oracle_text(host: Any, oracle: Any, data: Any) -> None:
    ids = data.draw(st.lists(_IDS, max_size=48))
    cuts = sorted(set(data.draw(st.lists(st.integers(1, max(len(ids) - 1, 1)), max_size=10))))
    edges = [0, *[c for c in cuts if 0 < c < len(ids)], len(ids)]
    chunks = [ids[a:b] for a, b in itertools.pairwise(edges)]
    outs, tail = run_stream(host, chunks)
    assert "".join(outs) + tail == _odec(oracle, ids)


# --------------------------------------------------------------------------- the tables


def test_special_tables_equal_the_pinned_files_special_added_tokens(pinned: dict[str, Any]) -> None:
    mod = special_mod()
    added = sorted(
        (a for a in pinned["json"]["added_tokens"] if a["special"]), key=lambda a: a["id"]
    )
    assert len(added) == 24
    assert tuple(a["content"] for a in added) == mod.SPECIAL_TOKENS
    assert dict(mod.SPECIAL_IDS) == {a["id"]: a["content"] for a in added}


def test_stop_ids_equal_the_generation_config_and_replay() -> None:
    from local_astra.decoding import vocab
    from local_astra.eval.cd import replay

    mod = special_mod()
    assert frozenset(vocab.stop_token_ids()) == mod.STOP_IDS
    assert mod.STOP_IDS == replay.STOP_IDS


# --------------------------------------------------------------------------- the subset fixture


def test_every_subset_entry_equals_the_pinned_file(pinned: dict[str, Any], oracle: Any) -> None:
    f = fixture()
    src = pinned["json"]
    sub = f["tokenizer"]
    assert f["provenance"]["source_sha256"] == PINNED_SHA256
    full = src["model"]["vocab"]
    wrong = {p: i for p, i in sub["model"]["vocab"].items() if full.get(p) != i}
    assert not wrong, f"subset pieces whose id differs from the pinned file: {list(wrong)[:10]}"
    assert {k: v for k, v in sub["model"].items() if k not in ("vocab", "merges")} == {
        k: v for k, v in src["model"].items() if k not in ("vocab", "merges")
    }
    for key in (
        "version",
        "truncation",
        "padding",
        "added_tokens",
        "normalizer",
        "pre_tokenizer",
        "post_processor",
    ):
        assert sub[key] == src[key], key
    assert sub["decoder"] == src["decoder"]
    byte_pieces = {f"<0x{b:02X}>" for b in range(256)}
    assert byte_pieces <= set(sub["model"]["vocab"])
    for s in f["streams"]:
        assert s["text"] == _odec(oracle, s["ids"]), f"{s['case']}/{s['arm']}"
        assert set(s["ids"]) <= set(sub["model"]["vocab"].values())
    assert len({i for s in f["streams"] for i in s["ids"]}) == 276


def _main_checkout() -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return Path(proc.stdout.strip()).parent if proc.returncode == 0 else REPO


def test_the_subset_streams_are_the_recorded_token_ids() -> None:
    """Needs the untracked ``logs/t0087-probe-724c0c4/raw.jsonl`` of the main checkout; skipped (stated) without it."""
    prov = fixture()["provenance"]["recorded_streams"]
    raw_path = _main_checkout() / prov["path"]
    if not raw_path.is_file():
        pytest.skip(f"{raw_path} is not on this machine (logs/ is untracked)")
    raw = raw_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == prov["sha256"]
    recorded = {}
    for line in raw.decode().splitlines():
        row = json.loads(line)
        if row.get("what") == "arm" and row.get("arm") in prov["arms"]:
            ids: list[int] = []
            for ln in row["lines"]:
                if ln.startswith("data: ") and ln.strip() != "data: [DONE]":
                    for c in json.loads(ln[6:])["choices"]:
                        ids.extend(c.get("token_ids") or [])
            recorded[(row["case"], row["arm"])] = ids
    assert {(s["case"], s["arm"]): s["ids"] for s in fixture()["streams"]} == recorded
