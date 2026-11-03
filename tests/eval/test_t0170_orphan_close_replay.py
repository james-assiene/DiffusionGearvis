"""T-0170 criteria 5 and 6 on the real record (eval, CPU).

- c5 (client): the fixture ``t0170_orphan_close_cases.jsonl`` (provenance on its first line: seq 92 of the T-0125
  search smoke at b372edaa, ``<channel|>`` re-inserted at the splice) is fed as a step-2 continuation through the real
  ``VllmClient`` with the **pinned** ``HostDetokenizer`` (skipped, with the reason, when it is not in the local HF
  cache). The ids are the canvas spelled piece by piece from the pinned ``tokenizer.json`` vocabulary (one piece per
  character, byte fallback otherwise; ``<channel|>`` as id 101): not the engine's tokenisation, but the same decode.
- c6 (scan): ``scripts/replay_implicit_thought.py --json`` over the incident's session directory lists exactly the
  seq-92 record under splices and nothing as implicit (skipped when ``logs/`` is absent).

Expected values are the fixture's and the task file's, never the client's output.
Regime: CPU, no GPU, no network; the logs are only read; n = 1.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.fakes import vllm_sse as V
from tests.unit import _llm_id_support as S

#: ``gpu`` only gates selection (T-0046: a path-only ``pytest tests/eval`` selects nothing), as T-0142b's eval; CPU.
#: Run: ``uv run pytest tests/eval/test_t0170_orphan_close_replay.py -q -m gpu -rA``
pytestmark = [pytest.mark.gpu, pytest.mark.eval]

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "eval" / "t0170_orphan_close_cases.jsonl"
SCRIPT = REPO / "scripts" / "replay_implicit_thought.py"
LOG_DIR = "t0125-search-smoke-b372eda/phase1/sessions"


def rows() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = [json.loads(x) for x in FIXTURE.read_text().splitlines() if x.strip()]
    return lines[0]["provenance"], lines[1:]


def pinned() -> Any:
    from local_astra.toolcall.detok import HostDetokenizer

    try:
        return HostDetokenizer.load()
    except (OSError, ValueError) as exc:
        pytest.skip(f"the pinned tokenizer.json is not in the local HF cache: {exc}")


def pinned_ids(text: str) -> list[int]:
    from local_astra.toolcall import detok

    vocab = json.loads(detok.pinned_tokenizer_path().read_text())["model"]["vocab"]
    out: list[int] = []
    head, sep, tail = text.partition("<channel|>")
    for part, close in ((head, bool(sep)), (tail, False)):
        for ch in part:
            piece = "▁" if ch == " " else ch
            if piece in vocab:
                out.append(vocab[piece])
            else:
                out.extend(vocab[f"<0x{b:02X}>"] for b in ch.encode())
        if close:
            out.append(S.CHANNEL_CLOSE)
    return out


def test_fixture_provenance() -> None:
    prov, cases = rows()
    assert (prov["seq"], prov["request_id"], prov["sha"]) == (92, "t1-gen-s2", "b372edaa")
    (case,) = cases
    assert case["canvas"] == case["thought"] + "<channel|>" + case["answer"]


def test_seq92_split_by_the_pinned_detokenizer() -> None:
    """Criterion 5 (client): thought commits concatenate to the three reasoning sentences, all ``implicit_thought``;
    the answer is exactly the sentence."""
    det = pinned()
    _prov, (case,) = rows()
    ids = [*pinned_ids(case["canvas"]), case["stop_id"]]
    assert det.decode(ids[:-1]) == case["canvas"], "premise: the ids decode to the fixture canvas"
    chunks = V.partition(ids, 64)
    wire = V.Wire(streams=[V.build_lines(V.StreamSpec(chunks=chunks), det)])
    prompt = S.continuation_prompt(
        '<|tool_call>call:google_search{query:<|"|>height of the Eiffel Tower<|"|>}<tool_call|>'
        "<|tool_response>response:google_search{results:[]}<tool_response|>"
    )

    async def go() -> list[Any]:
        from local_astra.conductor.llm_client import VllmClient

        cl = VllmClient(transport=wire.transport(), detokenizer=det)
        try:
            return [c async for c in cl.generate(prompt, "t1-gen-s2")]
        finally:
            await cl.aclose()

    got = asyncio.run(go())
    assert S.texts(got, "answer") == case["answer"], S.texts(got, "answer")
    thoughts = [c for c in got if c.segment == "thought"]
    assert "".join(c.text for c in thoughts) == case["thought"]
    assert thoughts and all(getattr(c, "implicit_thought", False) is True for c in thoughts), (
        "T-0170 criterion 1: every thought commit of the implicit thought is marked"
    )


def _log_dir() -> Path:
    cands = [REPO / "logs" / LOG_DIR]
    proc = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        cands.append(Path(proc.stdout.strip()).parent / "logs" / LOG_DIR)
    for c in cands:
        if c.is_dir():
            return c
    pytest.skip(f"the incident logs are absent (tried {[str(c) for c in cands]})")


def test_scan_reports_exactly_seq92() -> None:
    """Criterion 6 over the b372edaa log: exactly the seq-92 record under splices, nothing implicit."""
    d = _log_dir()
    if not SCRIPT.exists():
        raise AssertionError(
            "T-0170 criterion 6: scripts/replay_implicit_thought.py does not exist yet"
        )
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(d), "--json"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(proc.stdout)
    got = [
        (Path(str(s["path"])).name, s["seq"], s["request_id"], s["attempt_id"])
        for s in doc["splices"]
    ]
    assert got == [
        ("20261027-112444-1064873.jsonl", 92, "t1-gen-s2", "11cb09b2dc51-t1-gen-s2-3")
    ], got
    assert doc["implicit"] == []
