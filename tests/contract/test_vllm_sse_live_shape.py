"""T-0095 part B: the wire fake (``tests/fakes/vllm_sse.py``) produces the live multi-canvas shape.

The real side of this contract is T-0088's gpu row 2 (``tests/contract/test_llm_tools_wire_gpu.py``), which compares
the live engine with the fake on every run. Its one recording is committed here, so the fake's multi-canvas shape is
pinned on CPU between GPU runs.

**The fixture** ``tests/fixtures/llm/t0095_row2_live.jsonl`` is row 2's ``row2_live.jsonl``, copied verbatim (sha256
pinned below) from the T-0093 ``/launch`` at commit ``efadf45``, 2026-10-13 15:29 local
(``logs/t0093-launch-efadf45/``; ``tests/fixtures/llm/MANIFEST.md``). Regime: vLLM v0.29.0,
``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev ``2ea8372``, RTX 5090, gpu-mem 0.86, CL 256, 48 steps, prefix caching
on, thinking on, tools_v1 ``bbfc53ccdd80``, n = 1, no seed, row 2's ``MULTI_CANVAS_USER`` prompt ("Write the numbers
from 1 to 300 separated by spaces."). Six chunks: the role chunk and five content chunks; only the last content chunk
(``finish_reason: "stop"``, ``stop_reason: 106``) carries ``system_fingerprint``.

Rows (criteria 6 and 7):

- the fake, built from the recording's ids and partition, equals the recording under row 2's ``strip`` (the same
  serialisation, the same dropped keys);
- a content chunk carries ``system_fingerprint`` iff its ``finish_reason`` is non-null: on synthetic multi-canvas
  streams ending in a stop id and in ``length``, and with ``truncate`` (no chunk has one).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from tests import chunk_shape_diff
from tests.fakes import vllm_sse as V
from tests.unit import _llm_id_support as S
from tests.unit._detok_support import detok_mod, write_tokenizer

pytestmark = pytest.mark.contract

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "t0095_row2_live.jsonl"
#: sha256 of ``logs/t0093-launch-efadf45/row2_live.jsonl`` as written by the launch (the copy is verbatim)
FIXTURE_SHA256 = "37737a2ab6b9833cc33f7f6c3dc197d54236f9c7be81343c8e5267cff7b0c66a"

#: The recording's ids that the T-0088a subset lacks, with their pieces in the pinned ``tokenizer.json``
#: (sha256 ``cc8d3a0c…``, rev ``2ea8372``; read from the local HF cache on 2026-10-13). The row's own oracle
#: is the served text: a wrong piece here makes the built ``content``/``reasoning`` differ from the recording.
ROW2_PIECES: dict[str, int] = {
    "▁by": 684,
    "▁from": 699,
    "▁list": 1694,
    "▁provide": 2847,
    "▁response": 3072,
    "▁numbers": 4945,
    "▁simply": 5181,
    "▁generate": 8729,
    "▁spaces": 9952,
    "▁separated": 15914,
}


@cache
def _detok() -> Any:
    """``HostDetokenizer`` over the T-0088a subset plus ``EXTRA_PIECES`` and ``ROW2_PIECES``."""

    def add(tok: dict[str, Any]) -> None:
        vocab = tok["model"]["vocab"]
        for piece, i in {**S.EXTRA_PIECES, **ROW2_PIECES}.items():
            assert vocab.get(piece, i) == i, f"{piece!r} already has another id in the subset"
            vocab[piece] = i

    path, sha = write_tokenizer(Path(tempfile.mkdtemp(prefix="t0095-detok-")), add)
    return detok_mod().HostDetokenizer(path, sha)


def _strip(c: dict[str, Any]) -> str:
    """Row 2's ``strip`` (``test_llm_tools_wire_gpu.py``), verbatim: default ``json.dumps``, four keys dropped."""
    return json.dumps(
        {
            k: v
            for k, v in c.items()
            if k not in ("id", "created", "prompt_token_ids", "prompt_text")
        }
    )


def _recording() -> list[dict[str, Any]]:
    data = FIXTURE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == FIXTURE_SHA256, (
        "the fixture is not the verbatim recording"
    )
    lines = data.decode().splitlines()
    assert lines[-1] == V.DONE_LINE
    return [json.loads(ln[len("data: ") :]) for ln in lines[:-1]]


def test_the_recording_is_the_multi_canvas_shape_the_task_describes() -> None:
    """Guard on the oracle itself (T-0095 amendment): role + 5 content chunks, every content chunk with ids, the
    fingerprint on the finishing chunk only. If this fails the fixture is not the recording the task is about."""
    chunks = _recording()
    assert len(chunks) == 6
    content = chunks[1:]
    assert all(isinstance(c["choices"][0].get("token_ids"), list) for c in content)
    assert [c["choices"][0]["finish_reason"] for c in content] == [None] * 4 + ["stop"]
    assert ["system_fingerprint" in c for c in chunks] == [False] * 5 + [True]


def test_the_fake_reproduces_the_live_multi_canvas_recording() -> None:
    """Criterion 7: ``V.build_chunks`` over the recording's ids, partitioned as recorded, equals the recording
    chunk for chunk under row 2's ``strip`` (role chunk included)."""
    live = _recording()
    built = V.build_chunks(
        V.StreamSpec(chunks=[c["choices"][0]["token_ids"] for c in live[1:]]), _detok()
    )
    if [_strip(c) for c in built] != [_strip(c) for c in live]:
        diff = chunk_shape_diff.first_mismatch(
            [json.loads(_strip(c)) for c in built], [json.loads(_strip(c)) for c in live]
        )
        pytest.fail(
            f"the fake (built) differs from the live recording (live):\n{diff}", pytrace=False
        )


def _ids(n: int) -> list[int]:
    """``n`` ids of plain prose from the subset (no special id), so a stream has several canvases."""
    pieces = S.words("Hi there, here are the numbers from one to ten in a list for you.")
    ids = S.ids(*pieces)
    assert len(ids) >= n
    return ids[:n]


def _content_chunks(spec: V.StreamSpec) -> list[dict[str, Any]]:
    built = V.build_chunks(spec, _detok())
    assert "system_fingerprint" not in built[0], "the role chunk never carries a fingerprint"
    return built[1:]


@pytest.mark.parametrize("ending", ["stop", "length"])
def test_only_the_finishing_chunk_of_a_multi_canvas_stream_carries_the_fingerprint(
    ending: str,
) -> None:
    """Criterion 6: on a four-canvas stream, chunks 1–3 (``finish_reason: null``) have no ``system_fingerprint``;
    the last (``"stop"`` with ``stop_reason`` 106, or ``"length"``) has the recorded value."""
    body = _ids(12)
    ids = [*body, S.TURN_CLOSE] if ending == "stop" else body
    chunks = _content_chunks(V.StreamSpec(chunks=V.partition(ids, 4)))
    assert len(chunks) >= 3
    reasons = [c["choices"][0]["finish_reason"] for c in chunks]
    assert reasons == [None] * (len(chunks) - 1) + [ending]
    assert [c.get("system_fingerprint") for c in chunks] == [None] * (len(chunks) - 1) + [
        "vllm-0.29.0-b591a092"
    ]
    assert ["system_fingerprint" in c for c in chunks[:-1]] == [False] * (len(chunks) - 1), (
        "absent, not null"
    )


def test_a_truncated_stream_carries_no_fingerprint() -> None:
    """Criterion 6, inferred: a truncated stream never sends its finishing chunk, so no chunk has one."""
    chunks = _content_chunks(V.StreamSpec(chunks=V.partition(_ids(12), 4), truncate=True))
    assert len(chunks) == 3
    assert [c["choices"][0]["finish_reason"] for c in chunks] == [None] * 3
    assert not any("system_fingerprint" in c for c in chunks)


@pytest.mark.parametrize("canvas", [1, 2, 3, 5, 13])
@pytest.mark.parametrize("truncate", [False, True])
@pytest.mark.parametrize("stop", [False, True])
def test_fingerprint_iff_non_null_finish_reason_under_every_knob(
    canvas: int, truncate: bool, stop: bool
) -> None:
    """Criterion 6 as a property over partitions and knobs (``drop_token_ids``, ``foreign_ids``, ``truncate``,
    stop/length): every content chunk has the key exactly when its ``finish_reason`` is non-null."""
    ids = [*_ids(12), S.TURN_CLOSE] if stop else _ids(12)
    spec = V.StreamSpec(
        chunks=V.partition(ids, canvas),
        truncate=truncate,
        drop_token_ids=frozenset({0}),
        foreign_ids={0: [999_999]},
    )
    for k, c in enumerate(_content_chunks(spec)):
        has = "system_fingerprint" in c
        assert has == (c["choices"][0]["finish_reason"] is not None), (
            f"chunk {k}: fingerprint={has}"
        )
