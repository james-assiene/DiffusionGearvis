"""T-0070 criterion 2: prompts byte-identical to diffgemma-fa's builders, sent as one user message with
thinking off, ``max_tokens`` 256 and none of temperature / seed / response_format.

Oracle: ``tests/fixtures/cd/prompts/prompts.json``, produced by executing diffgemma-fa's own
``build_prompt`` functions at 27c7d0d (MANIFEST there). Regime: CPU, fake server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.cd import _eval_support as es
from tests.cd.fakes.vllm_server import Reply


@pytest.mark.parametrize(
    ("task", "cut"), [("bfcl", "test"), ("bfcl", "dev"), ("sudoku", "test"), ("sudoku", "dev")]
)
def test_build_prompt_is_byte_identical_to_diffgemma_fa(task: str, cut: str) -> None:
    expected = es.prompt_fixture()[task][cut]
    recs = es.cuts().load(task, cut)
    # Ruling E4: Sudoku dev ids are cut-qualified (dev:sudoku4_<k>); compare the unqualified id.
    assert [es.unqualified(r.id) for r in recs] == [e["id"] for e in expected]
    bad = [
        e["id"]
        for rec, e in zip(recs, expected, strict=True)
        if es.prompts().build_prompt(task, rec) != e["prompt"]
    ]
    assert bad == []


@pytest.mark.parametrize("arm", ["unconstrained", "end_projection"])
def test_wire_request_shape(tmp_path: Path, arm: str) -> None:
    """Every request: one user message whose content is the fixture prompt, thinking off, max_tokens 256,
    no temperature, seed or response_format (sampling stays the served model's stock)."""
    script = es.sudoku_script("test", lambda i, rid, body: "solved")
    with es.fake(script, tmp_path) as srv:
        res = es.run(srv, tmp_path, task="sudoku", cut="test", arm=arm, limit=5)
        bodies = list(srv.requests)
    assert res.exit_code == 0, res.stopped
    expected = [e["prompt"] for e in es.prompt_fixture()["sudoku"]["test"][:5]]
    assert [b["messages"] for b in bodies] == [[{"role": "user", "content": p}] for p in expected]
    for b in bodies:
        assert b["chat_template_kwargs"] == {"enable_thinking": False}
        assert b["max_tokens"] == 256
        for forbidden in ("temperature", "seed", "response_format"):
            assert forbidden not in b, forbidden
        xargs = b.get("vllm_xargs") or {}
        for forbidden in ("temperature", "seed", "response_format"):
            assert forbidden not in xargs, forbidden


def test_bfcl_wire_prompts_match_fixture(tmp_path: Path) -> None:
    fixture = es.prompt_fixture()["bfcl"]["test"][:6]
    with es.fake(lambda i, body: Reply(text="<channel|>{}"), tmp_path) as srv:
        res = es.run(srv, tmp_path, task="bfcl", cut="test", arm="unconstrained", limit=6)
        bodies = list(srv.requests)
    assert res.exit_code == 0, res.stopped
    assert [b["messages"][-1]["content"] for b in bodies] == [e["prompt"] for e in fixture]
    assert all(len(b["messages"]) == 1 for b in bodies)
