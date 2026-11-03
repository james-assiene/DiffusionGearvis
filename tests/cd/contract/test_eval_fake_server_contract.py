"""Contract rows for the behaviours ``tests/cd/fakes/vllm_server.py`` adds (T-0070; tests.md: a behaviour
added to a fake needs a real-service row). Parametrised ``backend = fake | real``; ``real`` is marked
``gpu`` and needs the served stack with the T-0067 plugin (``ASTRA_CD_REAL_URL``, e.g.
``http://127.0.0.1:8000/v1``, and ``ASTRA_CD_ENGINE_LOG``, the engine's log file). Run via /launch.

Pinned behaviours:
1. thinking-off responses carry no special piece in ``content``, and with ``return_token_ids`` the choice
   carries the generated ids whose detokenised ``answer_region`` equals that of ``<|channel>`` +
   reasoning + ``<channel|>`` + content (the join rule of criterion 5 needs exactly this);
2. a constrained request built by ``CdRequest.to_xargs()`` yields ``ASTRA_CD <json>`` engine-log lines
   that validate as ``CdCanvasRecord`` and echo the request's ``record_tag``, within 5 s of the response,
   and the last of them (highest ``canvas_idx``) carries ``stop_pos`` or ``error`` (E9: flushed on
   release, so a run's final record is complete);
3. (T-0076 criterion 1) a ``perstep_mar_qargmax`` request (constrained SC) yields records naming
   ``arm = "perstep_mar_qargmax"`` with ``mean_accepted_per_step`` filled; a qargmax request the parser refuses
   (``entropy_bound`` 0) yields one ``bad_value`` error record that still names the mode;
4. (T-0071b C9) ``GET /health`` answers 200 on a live engine (a health probe may use it; the driver's uses
   ``GET /v1/models``, pinned by row 1's server).

The fake's dead-engine mode (``die_after``/``die()``: connections dropped) models a crashed server and has no
real-service row: killing the 5090 engine is not a contract test.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.cd import _compile_support as cs
from tests.cd import _eval_support as es
from tests.cd.fakes.vllm_server import SPECIAL_PIECES, Reply


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.gpu)])
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[tuple[str, Path]]:
    if request.param == "real":
        url, log = os.environ.get("ASTRA_CD_REAL_URL"), os.environ.get("ASTRA_CD_ENGINE_LOG")
        if not url or not log:
            pytest.fail(
                "backend=real needs ASTRA_CD_REAL_URL and ASTRA_CD_ENGINE_LOG (run via /launch)"
            )
        yield url, Path(log)
        return
    sol = cs.sudoku_cut("test")[0][2]
    with es.fake(lambda i, b: Reply(text=es.solved_text(sol)), tmp_path) as srv:
        yield srv.url, srv.engine_log


def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        url + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        out: dict[str, Any] = json.loads(resp.read())
    return out


def _body(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    prompt = es.prompt_fixture()["sudoku"]["test"][0]["prompt"]
    body: dict[str, Any] = {
        "model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 256,
        "return_token_ids": True,
    }
    if extra:
        body["vllm_xargs"] = extra
    return body


def test_served_split_and_token_ids_support_the_answer_region_join(
    backend: tuple[str, Path],
) -> None:
    url, _ = backend
    out = _post(url, _body())
    choice = out["choices"][0]
    content = choice["message"]["content"] or ""
    assert not [p for p in SPECIAL_PIECES if p in content]
    ids = choice["token_ids"]
    text = cs.hf_tokenizer().decode(ids, skip_special_tokens=False)
    served = "<|channel>" + (choice["message"].get("reasoning") or "") + "<channel|>" + content
    assert cs.answer_region_oracle(text) == cs.answer_region_oracle(served)


def test_constrained_request_logs_valid_records_with_the_tag(backend: tuple[str, Path]) -> None:
    from local_astra.contracts.cd import CdCanvasRecord, CdRequest
    from tests.cd import _wire_support as ws

    url, log = backend
    puzzle = cs.sudoku_cut("test")[0][1]
    payload = ws.encode(cs.compiled_sudoku(puzzle))
    tag = f"contract-{time.time_ns()}"
    req = CdRequest(
        mode="end_projection", automaton=payload, automaton_sha256=ws.sha(payload), record_tag=tag
    )
    _post(url, _body(req.to_xargs()))
    deadline = time.monotonic() + 5.0
    found: list[dict[str, Any]] = []
    while time.monotonic() < deadline and not found:
        for line in log.read_text().splitlines():
            if "ASTRA_CD " in line:
                rec = json.loads(line.split("ASTRA_CD ", 1)[1])
                if rec.get("record_tag") == tag:
                    found.append(rec)
        time.sleep(0.05)
    assert found, f"no ASTRA_CD record with record_tag={tag} in {log}"
    for rec in found:
        CdCanvasRecord.model_validate(rec)
        assert rec["arm"] == "end_projection"
    last = max(found, key=lambda r: r["canvas_idx"])
    # E9: the final canvas's record (flushed on release) ends the request: it carries a stop or an error
    assert last["stop_pos"] is not None or last["error"] is not None, last


def _records_for(log: Path, tag: str) -> list[dict[str, Any]]:
    deadline = time.monotonic() + 5.0
    found: list[dict[str, Any]] = []
    while time.monotonic() < deadline and not found:
        for line in log.read_text().splitlines():
            if "ASTRA_CD " in line:
                rec = json.loads(line.split("ASTRA_CD ", 1)[1])
                if rec.get("record_tag") == tag:
                    found.append(rec)
        time.sleep(0.05)
    return found


def test_qargmax_request_records_name_the_new_arm(backend: tuple[str, Path]) -> None:
    from local_astra.contracts.cd import CdCanvasRecord, CdRequest
    from tests.cd import _wire_support as ws

    url, log = backend
    puzzle = cs.sudoku_cut("test")[0][1]
    payload = ws.encode(cs.compiled_sudoku(puzzle))
    tag = f"contract-qa-{time.time_ns()}"
    req = CdRequest(
        mode="perstep_mar_qargmax",
        self_conditioning="constrained",
        automaton=payload,
        automaton_sha256=ws.sha(payload),
        record_tag=tag,
    )
    _post(url, _body(req.to_xargs()))
    found = _records_for(log, tag)
    assert found, f"no ASTRA_CD record with record_tag={tag} in {log}"
    for rec in found:
        CdCanvasRecord.model_validate(rec)
        assert rec["arm"] == "perstep_mar_qargmax", rec
        assert rec["error"] is None, rec
        assert rec["mean_accepted_per_step"] is not None and rec["steps"] > 0, rec
        assert rec["shadow_argmax_ids"] is None and rec["map_equals_argmax"] is None, rec


def test_a_refused_qargmax_request_error_record_names_the_mode(backend: tuple[str, Path]) -> None:
    from local_astra.contracts.cd import CdCanvasRecord
    from tests.cd import _wire_support as ws

    url, log = backend
    puzzle = cs.sudoku_cut("test")[0][1]
    payload = ws.encode(cs.compiled_sudoku(puzzle))
    tag = f"contract-qa-bad-{time.time_ns()}"
    x = ws.raw_xargs(payload, mode="perstep_mar_qargmax", record_tag=tag, entropy_bound=0)
    _post(url, _body(x))
    found = _records_for(log, tag)
    assert len(found) == 1, f"want exactly one error record for {tag}, got {found}"
    CdCanvasRecord.model_validate(found[0])
    assert found[0]["error"] == "bad_value", found[0]
    assert found[0]["arm"] == "perstep_mar_qargmax", found[0]


def test_health_answers_200_on_a_live_engine(backend: tuple[str, Path]) -> None:
    url, _ = backend
    root = url.rstrip("/").removesuffix("/v1")
    with urllib.request.urlopen(root + "/health", timeout=10) as resp:
        assert resp.status == 200
