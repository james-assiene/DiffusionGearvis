"""Build ``t0087_streams.jsonl`` (T-0088) from T-0087's recorded probe streams.

One-off build step (stdlib only)::

    uv run python tests/fixtures/llm/build_t0087_streams.py --raw <main checkout>/logs/t0087-probe-724c0c4/raw.jsonl

Never imported by a test. ``logs/`` is not tracked, so the fixture this writes is the only committed copy of the
arm-(b)/(e) wire.

What it writes, one JSON object per line:

- line 1: ``{"provenance": {...}}`` (source path, its sha256, the probe commit, the arms, the rewrite rule);
- ``t0087_q10_tokenize.json``: q10's ten ``/tokenize`` prompts and the engine's ``tokens`` (gpu row 4's oracle);
- then one line per stream, in ``raw.jsonl`` order: ``case``, ``arm``, ``status``, ``request`` (the body's keys in
  order, its scalar fields verbatim, and ``messages``/``tools`` replaced by ``{len, sha256}``) and ``lines``: the
  stream's ``data:`` lines, **verbatim**, except the role chunk, whose ``prompt_token_ids`` is replaced by
  ``{len, sha256}`` (sha256 of the compact JSON list) and re-serialised the way the engine serialises
  (``separators=(",", ":")``, ``ensure_ascii=False``; the builder asserts that this round-trips every recorded line
  byte for byte, so the rewrite touches nothing else). ``prompt_text`` was ``null`` on every recorded stream and is
  kept as ``null``: there is nothing to digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parent / "t0087_streams.jsonl"
Q10_OUT = Path(__file__).resolve().parent / "t0087_q10_tokenize.json"
RAW_SHA256 = "f871d037fde4414abd53f8ce6011b3d7efb76cbc79bf9817bd805bbe58a848b0"
RAW_REL = "logs/t0087-probe-724c0c4/raw.jsonl"
COMMIT = "724c0c4"
ARMS = ("b", "e")


def _compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _digest(obj: Any) -> dict[str, Any]:
    return {"len": len(obj), "sha256": hashlib.sha256(_compact(obj).encode()).hexdigest()}


def _request(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"keys": list(body)}
    for key, value in body.items():
        out[key] = _digest(value) if key in ("messages", "tools") else value
    return out


def _lines(lines: list[str]) -> list[str]:
    data = [ln for ln in lines if ln.startswith("data: ")]
    assert all(not ln or ln.startswith("data: ") for ln in lines), (
        "a non-data SSE field was recorded"
    )
    out: list[str] = []
    for ln in data:
        payload = ln[len("data: ") :]
        if payload == "[DONE]":
            out.append(ln)
            continue
        chunk = json.loads(payload)
        assert "data: " + _compact(chunk) == ln, "the engine's serialisation does not round-trip"
        if "prompt_token_ids" in chunk:
            chunk["prompt_token_ids"] = _digest(chunk["prompt_token_ids"])
            assert chunk.get("prompt_text") is None, (
                "prompt_text was recorded non-null: digest it too"
            )
            out.append("data: " + _compact(chunk))
        else:
            out.append(ln)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    args = ap.parse_args()
    raw = args.raw.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RAW_SHA256, "not T-0087's raw.jsonl"
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    streams = [r for r in rows if r.get("what") == "arm" and r.get("arm") in ARMS]
    assert len(streams) == 24, len(streams)
    out = [
        {
            "provenance": {
                "task": "T-0088",
                "source": RAW_REL,
                "source_sha256": RAW_SHA256,
                "commit": COMMIT,
                "arms": list(ARMS),
                "n_streams": len(streams),
                "rewrite": "role chunk prompt_token_ids -> {len, sha256 of compact JSON}; prompt_text null kept; "
                "request messages/tools -> {len, sha256}; every other data: line verbatim",
                "generator": "tests/fixtures/llm/build_t0087_streams.py",
                "doc": "docs/cd/toolcall-live-probe.md",
            }
        }
    ]
    for r in streams:
        assert r["status"] == 200 and r["error"] is None
        out.append(
            {
                "case": r["case"],
                "arm": r["arm"],
                "status": r["status"],
                "request": _request(r["request"]),
                "lines": _lines(r["lines"]),
            }
        )
    OUT.write_text("".join(_compact(o) + "\n" for o in out))
    q10 = [r for r in rows if r.get("what") == "q10"]
    assert len(q10) == 10 and all(r["status"] == 200 for r in q10)
    Q10_OUT.write_text(
        json.dumps(
            {
                "provenance": {
                    "source": RAW_REL,
                    "source_sha256": RAW_SHA256,
                    "commit": COMMIT,
                    "what": "q10",
                },
                "rows": [
                    {
                        "case": r["case"],
                        "prompt": r["request"]["prompt"],
                        "tokens": r["payload"]["tokens"],
                    }
                    for r in q10
                ],
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"wrote {OUT} ({len(out) - 1} streams)")


if __name__ == "__main__":
    main()
