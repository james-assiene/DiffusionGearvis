#!/usr/bin/env python3
"""T-0091 criterion 1 -- write the precompiled native T4 payloads the live repair sends. CPU only, no network.

    uv run --all-extras --group cd python scripts/build_toolcall_payloads.py [--out PATH]

For each v1 tool (``toolcall/tools_v1.json``, file order) this compiles the automaton exactly as the eval did for
the cell ``native/T4/named/perstep_mar+seed`` (``twopass._Run.automaton``: ``compile_toolcall(tools, scope=<tool>,
thought="T4", whitespace=<TwoPassConfig default>, call_format="native")``) and encodes it with
``artifact.encode_automaton``, the string ``twopass`` put on the wire. The output holds:

- ``manifest``: the tools_v1 sha, the tokenizer sha, the model revision, the compiler version, the canvas length
  (CL) and the cell's coordinates (format, scope, thought, whitespace);
- ``tools``: per v1 tool, ``automaton``, ``automaton_sha256`` (sha256 of its ASCII bytes), ``n_states`` and
  ``bucket``.

This script is the file's only writer and ``tests/cd/unit/test_toolcall_live_parity.py`` is its oracle. The output
is deterministic (sorted keys, no timestamp). Exit 0: written. Exit 2: refused (a bucket above the live gate of
128); nothing is written on a refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from local_astra.decoding import artifact, grammars, vocab
from local_astra.decoding import toolcall as _tc
from local_astra.eval.cd import twopass
from local_astra.toolcall import tools as _tools

EXIT_OK = 0
EXIT_REFUSED = 2

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "local_astra" / "toolcall" / "payloads" / "native_T4.json"
THOUGHT = "T4"
CALL_FORMAT = "native"
LIVE_MAX_BUCKET = 128


def build() -> dict[str, Any]:
    ts = _tools.load()
    whitespace = twopass.TwoPassConfig.__dataclass_fields__["whitespace"].default
    entries: dict[str, dict[str, Any]] = {}
    for name in ts.names:
        a = _tc.compile_toolcall(
            ts, scope=name, thought=THOUGHT, whitespace=whitespace, call_format=CALL_FORMAT
        )
        payload = artifact.encode_automaton(a)
        entries[name] = {
            "automaton": payload,
            "automaton_sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
            "n_states": int(a.n_states),
            "bucket": int(a.n_states_bucket),
        }
    return {
        "manifest": {
            "task": "T-0091",
            "cell": "native/T4/named/perstep_mar+seed",
            "call_format": CALL_FORMAT,
            "scope": "named",
            "thought": THOUGHT,
            "whitespace": whitespace,
            "tools_sha256": ts.version,
            "tokenizer_sha256": vocab.TOKENIZER_SHA256,
            "model_revision": vocab.MODEL_REVISION,
            "compiler_version": grammars.compiler_version(),
            "canvas_length": twopass.CANVAS_LENGTH,
            "generator": "scripts/build_toolcall_payloads.py",
        },
        "tools": entries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="output path (default: %(default)s)"
    )
    args = ap.parse_args(argv)
    doc = build()
    over = {n: e["bucket"] for n, e in doc["tools"].items() if e["bucket"] > LIVE_MAX_BUCKET}
    if over:
        print(f"refused: buckets above the live gate {LIVE_MAX_BUCKET}: {over}", file=sys.stderr)
        return EXIT_REFUSED
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    for n, e in doc["tools"].items():
        print(
            f"{n}: n_states={e['n_states']} bucket={e['bucket']} sha={e['automaton_sha256'][:12]}"
        )
    print(f"wrote {out}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
