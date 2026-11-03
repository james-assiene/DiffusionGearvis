"""T-0070 criterion 5: constraint satisfaction by host replay of the committed ids, joined to the
response on ``record_tag`` and ordered by ``canvas_idx``; the join check compares ``answer_region`` of
the detokenised committed ids with that of what was served (reasoning + content).

The fake serves the gemma4 reasoning split (header -> ``reasoning``, special pieces dropped from
``content``), interleaves decoy records of other tags, writes multi-canvas records in reverse order, and
can delay the records past the HTTP response (T-0067: a record may lag its commit). Expected CS comes
from the scripted texts: ``solved``/``wrong`` grids keep the givens (accepted), ``rejected`` overwrites a
given. Regime: CPU, fake server, Sudoku test cut (bucket 64 automata).
"""

from __future__ import annotations

from pathlib import Path

from tests.cd import _compile_support as cs
from tests.cd import _eval_support as es
from tests.cd.fakes.vllm_server import Reply


def _run(tmp_path: Path, decide, n: int = 4, arm: str = "end_projection"):  # type: ignore[no-untyped-def]
    cs.warm_sudoku("test")
    with es.fake(es.sudoku_script("test", decide), tmp_path) as srv:
        res = es.run(srv, tmp_path, task="sudoku", cut="test", arm=arm, limit=n)
        records = list(srv.records)
    _, rows, _ = es.read_artifact(res.artifact)
    return res, rows, records


def test_legit_records_join_and_replay_accepts(tmp_path: Path) -> None:
    res, rows, records = _run(tmp_path, lambda i, rid, b: "solved" if i % 2 == 0 else "wrong")
    assert res.exit_code == 0, res.stopped
    assert [r["error"] for r in rows] == [None] * 4
    assert [r["cs"] for r in rows] == [True] * 4
    assert [r["scores"]["solved"] for r in rows] == [True, False, True, False]
    by_tag = {rec["record_tag"]: rec["committed_ids"] for rec in records}
    assert sorted(map(tuple, (r["committed_ids"] for r in rows))) == sorted(
        map(tuple, by_tag.values())
    )


def test_nonempty_header_name_still_joins(tmp_path: Path) -> None:
    """The header name ("```" here, as in the preserved map arm) is served as ``reasoning``; the join
    must still match (a comparison that keeps the header on one side only would never match)."""
    by_id = es.sudoku_by_id("test")

    def decide(i: int, rid: str, body: object) -> Reply:
        return Reply(text="<|channel>```\n<channel|>" + es.grid_text(by_id[rid][1]))

    res, rows, _ = _run(tmp_path, decide)
    assert res.exit_code == 0, res.stopped
    assert [r["error"] for r in rows] == [None] * 4
    assert all(r["cs"] and r["scores"]["solved"] for r in rows)


def test_answer_region_mismatch_is_join_error_and_counts_as_failure(tmp_path: Path) -> None:
    """Records whose committed text differs from the served answer (a wrong join) -> ``join_error``;
    the row fails even though the served content is a correct solution. Placed after record 20 so hard
    stop (c) does not fire."""
    by_id = es.sudoku_by_id("test")

    def decide(i: int, rid: str, body: object) -> Reply | str:
        if i == 21:
            puzzle, sol = by_id[rid]
            return Reply(text=es.solved_text(sol), committed_text=es.wrong_text(puzzle, sol))
        return "solved"

    res, rows, _ = _run(tmp_path, decide, n=23)
    assert res.exit_code == 0, res.stopped
    assert rows[21]["error"] == "join_error"
    assert rows[21]["scores"]["solved"] is False
    assert rows[21]["cs"] is not True
    assert [r["error"] for i, r in enumerate(rows) if i != 21] == [None] * 22


def test_missing_records_are_join_error(tmp_path: Path) -> None:
    def decide(i: int, rid: str, body: object) -> Reply | str:
        if i == 22:
            return Reply(text=es.solved_text(es.sudoku_by_id("test")[rid][1]), omit_records=True)
        return "solved"

    res, rows, _ = _run(tmp_path, decide, n=23)
    assert res.exit_code == 0, res.stopped
    assert rows[22]["error"] == "join_error"
    assert rows[22]["scores"]["solved"] is False


def test_records_lagging_the_response_still_join(tmp_path: Path) -> None:
    def decide(i: int, rid: str, body: object) -> Reply:
        return Reply(text=es.solved_text(es.sudoku_by_id("test")[rid][1]), record_delay_s=0.3)

    res, rows, _ = _run(tmp_path, decide, n=3)
    assert res.exit_code == 0, res.stopped
    assert [r["error"] for r in rows] == [None] * 3
    assert all(r["cs"] for r in rows)


def test_multi_canvas_records_are_ordered_by_canvas_idx(tmp_path: Path) -> None:
    """Two canvases per request, written to the log in reverse order: the row's committed ids are the
    canvases concatenated in canvas order, i.e. exactly encode(text) + [EOS]."""
    by_id = es.sudoku_by_id("test")

    def decide(i: int, rid: str, body: object) -> Reply:
        return Reply(text=es.solved_text(by_id[rid][1]), canvases=2)

    res, rows, _ = _run(tmp_path, decide, n=3)
    assert res.exit_code == 0, res.stopped
    for row in rows:
        want = [*cs.encode(es.solved_text(by_id[row["id"]][1])), 1]
        assert list(row["committed_ids"]) == want
        assert row["cs"] is True
        assert len(row["steps"]) == 2


def test_cs_comes_from_replay_not_from_the_plugin(tmp_path: Path) -> None:
    """The plugin says in_final=False on an accepted canvas: replay/plugin disagreement is hard stop (a)
    -- the harness must not simply copy the plugin's verdict into ``cs``."""
    by_id = es.sudoku_by_id("test")

    def decide(i: int, rid: str, body: object) -> Reply | str:
        if i == 1:
            return Reply(text=es.solved_text(by_id[rid][1]), in_final=False)
        return "solved"

    res, _, _ = _run(tmp_path, decide, n=4)
    assert res.exit_code != 0
    assert (res.stopped or "").startswith("a:")
    assert es.stopped_marker(res.artifact).startswith("a:")


# ------------------------------------------------------------------ E9: record flush (review 2026-09-30)


def test_e9_held_records_join_through_flush_requests(tmp_path: Path) -> None:
    """The fake holds each request's records until the next request arrives, as an engine that logs
    the final canvas late would. The run's last record can only arrive through the harness's own
    flush request(s): every row, the last included, must join cleanly."""
    cs.warm_sudoku("test")
    by_id = es.sudoku_by_id("test")

    def decide(i: int, rid: str, b: object) -> Reply:
        return Reply(text=es.solved_text(by_id[rid][1]), hold_records=True)

    with es.fake(es.sudoku_script("test", decide), tmp_path) as srv:
        res = es.run(
            srv,
            tmp_path,
            task="sudoku",
            cut="test",
            arm="end_projection",
            limit=3,
            record_wait_s=0.3,
        )
        flushes = list(srv.flushes)
        n_requests = len(srv.requests)
    assert res.exit_code == 0, res.stopped
    regime, rows, _ = es.read_artifact(res.artifact)
    assert [r["error"] for r in rows] == [None] * 3
    assert all(r["cs"] is True for r in rows)
    assert flushes, "the last record can only arrive through a flush request"
    for f in flushes:
        assert not [k for k in (f.get("vllm_xargs") or {}) if k.startswith("astra_cd_")]
        assert f["max_tokens"] == 1
    # flushes are never rows and never in n
    assert n_requests == 3
    assert regime["n"] == 3 and len(rows) == 3
    assert [r["id"] for r in rows] == [r.id for r in es.cuts().load("sudoku", "test")[:3]]


def test_e9_flushes_do_not_enter_unconstrained_denominators(tmp_path: Path) -> None:
    """A run whose records all arrive on time sends no flush that could leak into a row."""
    with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), tmp_path) as srv:
        res = es.run(srv, tmp_path, task="sudoku", cut="test", arm="unconstrained", limit=3)
        flushes = list(srv.flushes)
    assert res.exit_code == 0, res.stopped
    regime, rows, _ = es.read_artifact(res.artifact)
    assert regime["n"] == 3 and len(rows) == 3
    assert flushes == []


# ------------------------------------------------------------------ completeness (records.py)


def test_join_waits_for_the_last_canvas(tmp_path: Path) -> None:
    """Canvas 0 has no stop; canvas 1 (with the EOS) reaches the log 0.3 s later. The row must join
    cleanly with both canvases -- neither ``join_error`` nor an early join on canvas 0 alone."""
    by_id = es.sudoku_by_id("test")

    def decide(i: int, rid: str, b: object) -> Reply:
        return Reply(text=es.solved_text(by_id[rid][1]), canvases=2, last_canvas_delay_s=0.3)

    res, rows, records = _run(tmp_path, decide, n=2)
    assert res.exit_code == 0, res.stopped
    first = [r for r in records if r["canvas_idx"] == 0]
    assert first and all(r["stop_pos"] is None for r in first), "premise: canvas 0 carries no stop"
    for row in rows:
        want = [*cs.encode(es.solved_text(by_id[row["id"]][1])), 1]
        assert row["error"] is None
        assert list(row["committed_ids"]) == want
        assert len(row["steps"]) == 2
        assert row["cs"] is True


# ------------------------------------------------------------------ E3: the rebuilt served text


def test_e3_literal_channel_close_in_content_joins_only_with_the_rebuilt_text(
    tmp_path: Path,
) -> None:
    """A JSON string may spell ``<channel|>`` with ordinary tokens (not id 101); ``skip_special_tokens``
    keeps it in ``content``. ``answer_region`` of the committed text and of ``<|channel>`` + reasoning +
    ``<channel|>`` + content both cut at it; ``answer_region(content)`` alone splits *at* it and keeps
    the tail -- so a content-only join reports ``join_error`` on a legitimate record (E3 is not a no-op).
    """
    from tests.cd import _wire_support as ws  # noqa: F401  (plugin parser used by the fake)

    spelled = [*cs.encode("<"), *cs.encode("channel"), *cs.encode("|>")]
    assert 101 not in spelled, "premise: the literal is spelled with ordinary tokens"
    ids1 = [*cs.encode(es.HEADER + '{"repos": "a'), *spelled, *cs.encode('b"}'), 1]
    content1 = '{"repos": "a<channel|>b"}'
    oracle = cs.answer_region_oracle
    detok = cs.hf_tokenizer().decode(ids1, skip_special_tokens=False)
    assert oracle(detok) == oracle("<|channel><channel|>" + content1) == '{"repos": "a'
    assert oracle(content1) != oracle(detok), "premise: content alone gives a different region"
    rid1 = "live_simple_1-1-0"
    walker = cs.walker(cs.grammars().compile_bfcl(cs.bfcl_function(rid1), whitespace="json"))
    assert walker.accepts(ids1), "premise: the grammar accepts the spelled literal inside a string"

    resolve = es.BfclResolver("test")
    preserved = {r["id"]: r["text"] for r in cs.preserved("exp_e5_grammar130_map")["rows"]}

    def script(i: int, body: dict) -> Reply:  # type: ignore[type-arg]
        rid = resolve(body)
        if rid == rid1:
            return Reply(committed_ids=ids1, served_reasoning="", served_content=content1)
        return Reply(text=preserved[rid])

    with es.fake(script, tmp_path) as srv:
        res = es.run(srv, tmp_path, task="bfcl", cut="test", arm="end_projection", limit=2)
    assert res.exit_code == 0, res.stopped
    _, rows, _ = es.read_artifact(res.artifact)
    row = next(r for r in rows if r["id"] == rid1)
    assert row["error"] is None
    assert row["cs"] is True


# ------------------------------------------------------------------ record structure (re-review)


def _structural(tmp_path: Path, **fault: bool):  # type: ignore[no-untyped-def]
    """Record 21 of 21 (after stop (c)'s window): two canvases whose ids are complete and whose text
    matches, so only the canvas numbering is wrong."""
    by_id = es.sudoku_by_id("test")

    def decide(i: int, rid: str, b: object) -> Reply | str:
        if i == 20:
            return Reply(text=es.solved_text(by_id[rid][1]), canvases=2, **fault)
        return "solved"

    return _run(tmp_path, decide, n=21)


def test_duplicated_canvas_idx_is_join_error(tmp_path: Path) -> None:
    """The last canvas's record appears twice, byte-identical. The text join still matches (first record
    of an index wins), so only the duplicate check can catch it."""
    res, rows, _ = _structural(tmp_path, duplicate_last_canvas=True)
    assert res.exit_code == 0, res.stopped
    assert rows[20]["error"] == "join_error"
    assert rows[20]["scores"]["solved"] is False
    assert [r["error"] for r in rows[:20]] == [None] * 20


def test_gap_in_canvas_idx_is_join_error(tmp_path: Path) -> None:
    """Canvases numbered 0 and 2 with every id present: text matches, numbering has a gap."""
    res, rows, _ = _structural(tmp_path, gap_before_last_canvas=True)
    assert res.exit_code == 0, res.stopped
    assert rows[20]["error"] == "join_error"
    assert [r["error"] for r in rows[:20]] == [None] * 20


def test_e9_records_held_across_two_requests_join_on_the_second_flush(tmp_path: Path) -> None:
    """The last request's records reach the log only after two further requests: the harness's second
    flush (E9: up to 2) is what brings them. One flush is not enough."""
    cs.warm_sudoku("test")
    by_id = es.sudoku_by_id("test")

    def decide(i: int, rid: str, b: object) -> Reply | str:
        if i == 2:
            return Reply(text=es.solved_text(by_id[rid][1]), hold_records=True, hold_for_requests=2)
        return "solved"

    with es.fake(es.sudoku_script("test", decide), tmp_path) as srv:
        res = es.run(
            srv,
            tmp_path,
            task="sudoku",
            cut="test",
            arm="end_projection",
            limit=3,
            record_wait_s=0.3,
        )
        flushes = len(srv.flushes)
    assert res.exit_code == 0, res.stopped
    regime, rows, _ = es.read_artifact(res.artifact)
    assert [r["error"] for r in rows] == [None, None, None]
    assert flushes >= 2
    assert regime["n"] == 3 and len(rows) == 3
