"""Build ``vocab_subset_cc8d3a0c.json`` (T-0088a) from the pinned ``tokenizer.json`` and the T-0087 streams.

One-off build step, run in the ``cd`` group (it needs ``tokenizers`` for the oracle text)::

    uv run --all-extras --group cd python tests/fixtures/llm/build_vocab_subset_cc8d3a0c.py \\
        --raw /path/to/logs/t0087-probe-724c0c4/raw.jsonl

Never imported by a test. The source of truth for every entry is the pinned file;
``tests/cd/unit/test_host_detok_parity.py`` re-checks the fixture against it.

What it writes:

- ``tokenizer``: a ``tokenizer.json``-shaped object. ``version``, ``truncation``, ``padding``, ``added_tokens``,
  ``normalizer``, ``pre_tokenizer``, ``post_processor`` and ``decoder`` are verbatim; ``model`` is the pinned
  model config verbatim with ``vocab`` cut down to the subset and ``merges`` emptied (the detokeniser reads the
  decoder only). Tests write it to a temporary ``tokenizer.json`` and load it with an explicit path and sha.
- ``streams``: the token ids of the 24 arm-(b)/(e) streams (every ``choices[0].token_ids`` of every
  ``data:`` chunk, in order) and ``text``, the oracle decode
  ``tokenizers.Tokenizer.decode(ids, skip_special_tokens=False)`` on the pinned file.
- ``provenance``: the source sha, revision, path rule, the raw.jsonl sha and the oracle's version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import __version__ as tokenizers_version

from local_astra.decoding import vocab

OUT = Path(__file__).resolve().parent / "vocab_subset_cc8d3a0c.json"
RAW_SHA256 = "f871d037fde4414abd53f8ce6011b3d7efb76cbc79bf9817bd805bbe58a848b0"
RAW_REL = "logs/t0087-probe-724c0c4/raw.jsonl"
ARMS = ("b", "e")

#: A few extra pieces: ``▁`` runs and ``▁``-words, plus prose and fragment pieces the T-0088 rows can use
#: to splice a literal out of ordinary pieces (``<``, ``|``, ``|>``, ``>``, ``tool``, ``_``, ``call``).
EXTRA_PIECES = (
    "▁",
    "▁▁",
    "▁▁▁",
    "▁I",
    "▁the",
    "▁a",
    "▁b",
    "▁<",
    ".",
    "A",
    "B",
    "é",
    "€",
    "<",
    "|",
    "|>",
    ">",
    "tool",
    "_",
    "call",
)


def stream_ids(row: dict) -> list[int]:
    ids: list[int] = []
    for line in row["lines"]:
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        chunk = json.loads(line[len("data: ") :])
        for choice in chunk["choices"]:
            ids.extend(int(t) for t in (choice.get("token_ids") or []))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    args = ap.parse_args()

    raw_bytes = args.raw.read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == RAW_SHA256, (
        "raw.jsonl is not the T-0087 recording"
    )
    path = vocab.tokenizer_json_path()
    src_bytes = path.read_bytes()
    src_sha = hashlib.sha256(src_bytes).hexdigest()
    assert src_sha == vocab.TOKENIZER_SHA256
    src = json.loads(src_bytes)
    oracle = Tokenizer.from_file(str(path))

    streams = []
    for line in raw_bytes.decode().splitlines():
        row = json.loads(line)
        if row.get("what") != "arm" or row.get("arm") not in ARMS:
            continue
        ids = stream_ids(row)
        streams.append(
            {
                "case": row["case"],
                "arm": row["arm"],
                "ids": ids,
                "text": oracle.decode(ids, skip_special_tokens=False),
            }
        )
    assert len(streams) == 24, len(streams)
    stream_id_set = sorted({i for s in streams for i in s["ids"]})
    assert len(stream_id_set) == 276, len(stream_id_set)

    full = src["model"]["vocab"]
    by_id = {i: p for p, i in full.items()}
    specials = [a for a in src["added_tokens"] if a["special"]]
    assert len(specials) == 24
    byte_pieces = [f"<0x{b:02X}>" for b in range(256)]
    keep: set[int] = set(stream_id_set)
    keep |= {a["id"] for a in specials}
    keep |= {full[p] for p in byte_pieces}
    keep |= {full[p] for p in EXTRA_PIECES}
    sub_vocab = {by_id[i]: i for i in sorted(keep)}

    model = {k: v for k, v in src["model"].items() if k not in ("vocab", "merges")}
    model["vocab"] = sub_vocab
    model["merges"] = []
    tokenizer = {k: src[k] for k in src if k != "model"}
    tokenizer["model"] = model

    fixture = {
        "provenance": {
            "task": "T-0088a",
            "model": vocab.MODEL_ID,
            "revision": vocab.MODEL_REVISION,
            "source_file": "tokenizer.json (local HF cache, path rule of local_astra.decoding.vocab.snapshot_dir)",
            "source_sha256": src_sha,
            "source_vocab_size": len(full),
            "recorded_streams": {
                "path": RAW_REL,
                "sha256": RAW_SHA256,
                "commit": "724c0c4",
                "arms": list(ARMS),
                "n_streams": len(streams),
                "distinct_ids": len(stream_id_set),
            },
            "oracle": f"tokenizers {tokenizers_version}, decode(ids, skip_special_tokens=False)",
            "generator": "tests/fixtures/llm/build_vocab_subset_cc8d3a0c.py",
            "subset": "24 specials + 256 byte-fallback pieces + every stream id + EXTRA_PIECES; merges emptied",
        },
        "extra_pieces": list(EXTRA_PIECES),
        "tokenizer": tokenizer,
        "streams": streams,
    }
    OUT.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUT} ({len(sub_vocab)} pieces, {len(streams)} streams)")


if __name__ == "__main__":
    main()
