"""T-0070 criterion 6 (tests.md; diffgemma-fa RESULTS [B]/[B2]): every record of a cut stays in ``n``, and
for BFCL in ``arg_total`` = 291, whatever the outcome -- HTTP error, timeout, ``join_error``, ``TooLarge``,
loud CD error. A fake that fails one third of the records proves it.

Expected counts are by construction of the script (hand-counted): of 250 Sudoku test records, those with
index % 3 == 0 (84 of them) fail; the other 166 are served a correct, accepted solution. Failures that
would trip a hard stop are kept out of their windows (join errors only after record 20; CD errors of a
non-stopping class). Regime: CPU, fake server.
"""

from __future__ import annotations

from pathlib import Path

from tests.cd import _compile_support as cs
from tests.cd import _eval_support as es
from tests.cd.fakes.vllm_server import Reply


def _sudoku_decide(i: int, rid: str, body: object) -> Reply | str:
    if i % 3 != 0:
        return "solved"
    kind = (i // 3) % 4
    sol = es.sudoku_by_id("test")[rid][1]
    if kind == 0:
        return Reply(http_status=500)
    if kind == 1:
        return Reply(text=es.solved_text(sol), hang_s=1.5)  # beyond request_timeout_s
    if kind == 2:
        return Reply(error="bucket_too_large")  # a loud CD error outside hard stop (b)'s classes
    if i >= 21:
        puzzle = es.sudoku_by_id("test")[rid][0]
        return Reply(
            text=es.solved_text(sol), committed_text=es.wrong_text(puzzle, sol)
        )  # join_error
    return Reply(http_status=503)


def test_sudoku_constrained_one_third_failing_keeps_n_250(tmp_path: Path) -> None:
    cs.warm_sudoku("test")
    with es.fake(es.sudoku_script("test", _sudoku_decide), tmp_path) as srv:
        res = es.run(
            srv, tmp_path, task="sudoku", cut="test", arm="end_projection", request_timeout_s=0.5
        )
    assert res.exit_code == 0, res.stopped
    regime, rows, _ = es.read_artifact(res.artifact)
    assert regime["n"] == 250
    assert len(rows) == 250
    assert [r["id"] for r in rows] == [r.id for r in es.cuts().load("sudoku", "test")]
    failed = [r for i, r in enumerate(rows) if i % 3 == 0]
    assert len(failed) == 84
    assert all(r["error"] for r in failed)
    assert not any(r["scores"]["solved"] for r in failed)
    assert sum(bool(r["scores"]["solved"]) for r in rows) == 166
    kinds = {r["error"] for r in failed}
    assert "join_error" in kinds and "bucket_too_large" in kinds and len(kinds) >= 4, kinds


def test_bfcl_unconstrained_one_third_failing_keeps_arg_total_291(tmp_path: Path) -> None:
    preserved = {r["id"]: r["text"] for r in cs.preserved("exp_e5_grammar130_map")["rows"]}
    resolve = es.BfclResolver("test")

    def script(i: int, body: dict) -> Reply:  # type: ignore[type-arg]
        rid = resolve(body)
        if i % 3 == 0:
            return Reply(http_status=500) if i % 2 == 0 else Reply(text="{}", hang_s=1.5)
        return Reply(text=preserved[rid])

    with es.fake(script, tmp_path) as srv:
        res = es.run(
            srv, tmp_path, task="bfcl", cut="test", arm="unconstrained", request_timeout_s=0.5
        )
    assert res.exit_code == 0, res.stopped
    regime, rows, _ = es.read_artifact(res.artifact)
    assert regime["n"] == 130 and len(rows) == 130
    assert sum(int(r["scores"]["arg_total"]) for r in rows) == 291
    failed = [r for i, r in enumerate(rows) if i % 3 == 0]
    assert len(failed) == 44
    assert all(r["error"] for r in failed)
    assert sum(int(r["scores"]["arg_correct"]) for r in failed) == 0
    assert not any(r["scores"]["exact_call"] for r in failed)
    assert sum(int(r["scores"]["arg_correct"]) for r in rows) <= 193


def test_too_large_records_stay_in_n_and_are_never_sent(tmp_path: Path) -> None:
    """``max_states`` below every Sudoku automaton (32 states): each record is a counted ``too_large``
    failure; nothing reaches the server."""
    cs.warm_sudoku("test")
    with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), tmp_path) as srv:
        res = es.run(
            srv, tmp_path, task="sudoku", cut="test", arm="perstep_mf", limit=12, max_states=16
        )
        sent = len(srv.requests)
    regime, rows, _ = es.read_artifact(res.artifact)
    assert sent == 0
    assert regime["n"] == 12 and len(rows) == 12
    assert {r["error"] for r in rows} == {"too_large"}
    assert not any(r["scores"]["solved"] for r in rows)
