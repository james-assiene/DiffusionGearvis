"""T-0070 criterion 8: coded hard stops. Every stop has a fire row and a benign non-fire row on the fake
server; each fire writes a partial artifact marked ``stopped: <letter>: ...`` and exits non-zero.
Thresholds are exercised on both sides of the boundary (tests.md: a guard needs a too-wide row too).

Adversarial rows named by the task: a single replay rejection on record 129 of 130, and a header rate of
0 % that is reported and does not stop the run. Regime: CPU, fake server, Sudoku test cut unless stated.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.cd import _compile_support as cs
from tests.cd import _eval_support as es
from tests.cd.fakes.vllm_server import STEP_CAP, Reply

#: Loud CD error codes of hard stop (b)'s classes (Z0, empty delta*, infeasible). T-0067 has not fixed
#: the runtime codes yet; these are the test-author's proposal (see the report).
STOP_B_CODES = ("z0", "empty_delta", "infeasible")


def _sudoku(tmp_path: Path, decide, *, arm: str = "end_projection", **cfg):  # type: ignore[no-untyped-def]
    cs.warm_sudoku("test")
    with es.fake(es.sudoku_script("test", decide), tmp_path) as srv:
        res = es.run(srv, tmp_path, task="sudoku", cut="test", arm=arm, **cfg)
        sent = len(srv.requests)
    return res, sent


def _fired(res, letter: str) -> None:  # type: ignore[no-untyped-def]
    assert res.exit_code != 0
    assert (res.stopped or "").startswith(f"{letter}:"), res.stopped
    marker = es.stopped_marker(res.artifact)
    assert marker is not None and marker.startswith(f"{letter}:"), marker


def _clean(res) -> None:  # type: ignore[no-untyped-def]
    assert res.exit_code == 0, res.stopped
    assert res.stopped is None
    assert es.stopped_marker(res.artifact) is None


# ------------------------------------------------------------------------------------------ (a)


def test_a_fires_on_a_single_replay_rejection_at_record_129_of_130(tmp_path: Path) -> None:
    res, _ = _sudoku(tmp_path, lambda i, rid, b: "rejected" if i == 128 else "solved", limit=130)
    _fired(res, "a")
    _, rows, _ = es.read_artifact(res.artifact)
    assert len(rows) >= 128
    rej = [r for r in rows if r["id"] == "sudoku4_128"]
    assert rej and rej[0]["cs"] is False


def test_a_fires_on_rejection_even_when_the_plugin_agrees(tmp_path: Path) -> None:
    """Replay rejects and the plugin says in_final=False: no disagreement, but a terminated constrained
    output outside the language is itself stop (a)."""
    res, _ = _sudoku(tmp_path, lambda i, rid, b: "rejected" if i == 3 else "solved", limit=6)
    _fired(res, "a")


def test_a_fires_on_plugin_disagreement_with_an_accepted_replay(tmp_path: Path) -> None:
    by_id = es.sudoku_by_id("test")
    res, _ = _sudoku(
        tmp_path,
        lambda i, rid, b: (
            Reply(text=es.solved_text(by_id[rid][1]), in_final=False) if i == 2 else "solved"
        ),
        limit=6,
    )
    _fired(res, "a")


def test_a_does_not_fire_on_130_accepted_records(tmp_path: Path) -> None:
    res, sent = _sudoku(tmp_path, lambda i, rid, b: "solved" if i % 2 else "wrong", limit=130)
    _clean(res)
    assert sent == 130


# ------------------------------------------------------------------------------------------ (b)


@pytest.mark.parametrize("code", STOP_B_CODES)
def test_b_fires_on_a_loud_cd_error_of_the_stop_classes(tmp_path: Path, code: str) -> None:
    res, _ = _sudoku(
        tmp_path, lambda i, rid, b: Reply(error=code) if i == 5 else "solved", limit=10
    )
    _fired(res, "b")


def test_b_does_not_fire_on_another_loud_cd_error(tmp_path: Path) -> None:
    res, _ = _sudoku(
        tmp_path,
        lambda i, rid, b: Reply(error="bucket_too_large") if i == 5 else "solved",
        limit=10,
    )
    _clean(res)
    _, rows, _ = es.read_artifact(res.artifact)
    assert rows[5]["error"] == "bucket_too_large" and rows[5]["scores"]["solved"] is False


# ------------------------------------------------------------------------------------------ (c)


def _join_error_at(k: int):  # type: ignore[no-untyped-def]
    by_id = es.sudoku_by_id("test")

    def decide(i: int, rid: str, b: object) -> Reply | str:
        if i == k:
            puzzle, sol = by_id[rid]
            return Reply(text=es.solved_text(sol), committed_text=es.wrong_text(puzzle, sol))
        return "solved"

    return decide


def test_c_fires_on_a_join_error_in_record_20(tmp_path: Path) -> None:
    res, sent = _sudoku(tmp_path, _join_error_at(19), limit=30)
    _fired(res, "c")
    assert sent < 30


def test_c_does_not_fire_on_a_join_error_in_record_21(tmp_path: Path) -> None:
    res, sent = _sudoku(tmp_path, _join_error_at(20), limit=30)
    _clean(res)
    assert sent == 30


# ------------------------------------------------------------------------------------------ (d)


def test_d_fires_when_unconstrained_sudoku_solves_14_of_30(tmp_path: Path) -> None:
    res, _ = _sudoku(
        tmp_path, lambda i, rid, b: "solved" if i < 14 else "wrong", arm="unconstrained", limit=40
    )
    _fired(res, "d")
    _, rows, _ = es.read_artifact(res.artifact)
    assert len(rows) == 30, "the stack check runs exactly after record 30"


def test_d_does_not_fire_at_exactly_15_of_30(tmp_path: Path) -> None:
    res, sent = _sudoku(
        tmp_path,
        lambda i, rid, b: "solved" if (i < 15 or i >= 30) else "wrong",
        arm="unconstrained",
        limit=40,
    )
    _clean(res)
    assert sent == 40


def test_d_is_for_the_unconstrained_arm_only(tmp_path: Path) -> None:
    """A constrained arm solving 0/30 (accepted, wrong) is a result, not a broken stack."""
    res, _ = _sudoku(tmp_path, lambda i, rid, b: "wrong", arm="end_projection", limit=31)
    _clean(res)


def _bfcl(tmp_path: Path, valid_upto: int, limit: int):  # type: ignore[no-untyped-def]
    preserved = {r["id"]: r["text"] for r in cs.preserved("exp_e5_grammar130_map")["rows"]}
    resolve = es.BfclResolver("test")

    def script(i: int, body: dict) -> Reply:  # type: ignore[type-arg]
        rid = resolve(body)
        return Reply(text=preserved[rid] if i < valid_upto else "<channel|>I cannot help.")

    with es.fake(script, tmp_path) as srv:
        return es.run(srv, tmp_path, task="bfcl", cut="test", arm="unconstrained", limit=limit)


def test_d_fires_when_unconstrained_bfcl_has_no_schema_valid_output(tmp_path: Path) -> None:
    _fired(_bfcl(tmp_path, 0, 35), "d")


def test_d_does_not_fire_on_the_preserved_map_outputs(tmp_path: Path) -> None:
    """The first 30 preserved map-arm objects (130/130 schema valid in JAX) served unconstrained."""
    _clean(_bfcl(tmp_path, 35, 35))


# T-0078 (human decision 2026-10-07): BFCL's (d) counts outputs that *parsed* (the row's ``parse``
# field, the runner summary's ``parsed``), not schema-valid ones; same 30-record window, same 0.40
# floor. 0.40 * 30 = 12, so 12/30 parsed is the lowest count that does not fire and 11/30 fires.

NO_PARSE = "<channel|>I cannot help."
WRONG_TYPE: dict[str, object] = {
    "string": 12345,
    "integer": "not a number",
    "float": "not a number",
    "number": "not a number",
    "boolean": "yes",
    "array": "not a list",
    "tuple": "not a list",
    "dict": "not an object",
    "object": "not an object",
}


def _preserved_object(text: str) -> dict[str, object]:
    """The argument object of a preserved map-arm text, parsed here with ``json`` (independent of the
    scorer's extractor); ``{}`` when the preserved text is not a complete object."""
    body = text.split("<channel|>", 1)[1] if "<channel|>" in text else text
    body = body.replace("<turn|>", "").strip().strip("`").strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return dict(obj) if isinstance(obj, dict) else {}


def _schema_invalid_text(rid: str, preserved_text: str) -> str:
    """A parseable JSON object that breaks ``rid``'s schema, built from the preserved output.

    b2's shape first: an enum'd string argument set to a value outside its enum (b2 emitted
    ``"unit": "shell"`` where ``unit`` is ``seconds|milliseconds``). The vendored validator does not set
    ``additionalProperties: false``, so a *new* key alone is schema valid and cannot be the perturbation;
    without an enum'd argument, a required (else any) argument gets a value of the wrong type."""
    obj = _preserved_object(preserved_text)
    props = cs.bfcl_function(rid)["parameters"].get("properties") or {}
    required = cs.bfcl_function(rid)["parameters"].get("required") or []
    for name, spec in props.items():
        if spec.get("type") == "string" and spec.get("enum") and "shell" not in spec["enum"]:
            obj[name] = "shell"
            return "<channel|>" + json.dumps(obj)
    for name in [*required, *props]:
        bad = WRONG_TYPE.get(str(props.get(name, {}).get("type")))
        if bad is not None:
            obj[name] = bad
            return "<channel|>" + json.dumps(obj)
    raise AssertionError(f"{rid}: no argument to perturb ({props})")


def _bfcl_mix(tmp_path: Path, kind, limit: int = 35):  # type: ignore[no-untyped-def]
    """Serve unconstrained BFCL where record ``i`` is ``kind(i)``: ``"valid"`` (the preserved map
    output), ``"invalid"`` (parses, schema invalid) or ``"none"`` (does not parse). Returns the result
    and the artifact's first 30 rows."""
    preserved = {r["id"]: r["text"] for r in cs.preserved("exp_e5_grammar130_map")["rows"]}
    resolve = es.BfclResolver("test")

    def script(i: int, body: dict) -> Reply:  # type: ignore[type-arg]
        rid = resolve(body)
        k = kind(i)
        if k == "none":
            return Reply(text=NO_PARSE)
        return Reply(
            text=preserved[rid] if k == "valid" else _schema_invalid_text(rid, preserved[rid])
        )

    with es.fake(script, tmp_path) as srv:
        res = es.run(srv, tmp_path, task="bfcl", cut="test", arm="unconstrained", limit=limit)
    _, rows, _ = es.read_artifact(res.artifact)
    return res, rows[:30]


def _counts(rows) -> tuple[int, int]:  # type: ignore[no-untyped-def]
    """(parsed, schema valid) over ``rows``, as the scorer recorded them."""
    return (
        sum(bool(r["parse"]) for r in rows),
        sum(bool((r.get("scores") or {}).get("schema_valid")) for r in rows),
    )


def test_d_bfcl_premise_invalid_texts_parse_but_are_schema_invalid() -> None:
    """The guard-too-wide rows below are only meaningful if the scorer really marks every
    perturbed text parsed and not schema valid, and the unperturbed preserved text schema valid."""
    recs = {r.id: r for r in es.cuts().load("bfcl", "test")}
    rows = cs.preserved("exp_e5_grammar130_map")["rows"][:35]
    for r in rows:
        bad = es.cuts().score("bfcl", recs[r["id"]], _schema_invalid_text(r["id"], r["text"]))
        assert bad["parsed"] is True and bad["schema_valid"] is False, (r["id"], bad)
    assert es.cuts().score("bfcl", recs[rows[0]["id"]], NO_PARSE)["parsed"] is False


def test_d_does_not_fire_when_bfcl_outputs_parse_but_are_schema_invalid(tmp_path: Path) -> None:
    """Guard too wide (the case that stopped b2): 30/30 parse, 0/30 schema valid -> no stop."""
    res, rows = _bfcl_mix(tmp_path, lambda i: "invalid")
    assert _counts(rows) == (30, 0), "premise: every row parsed, none schema valid"
    _clean(res)
    _, all_rows, _ = es.read_artifact(res.artifact)
    assert len(all_rows) == 35


def test_d_does_not_fire_on_b2s_mix_29_parsed_11_schema_valid(tmp_path: Path) -> None:
    """b2's numbers (PROGRESS 2026-10-07): 29/30 parsed, 11/30 schema valid -> no stop."""
    res, rows = _bfcl_mix(
        tmp_path, lambda i: "valid" if i < 11 else ("none" if i == 29 else "invalid")
    )
    assert _counts(rows) == (29, 11), "premise"
    _clean(res)


def test_d_does_not_fire_at_12_of_30_parsed(tmp_path: Path) -> None:
    """Boundary, not firing: 12/30 = 0.40 parsed, all 12 schema invalid (so a schema-valid count,
    0/30, would fire)."""
    res, rows = _bfcl_mix(tmp_path, lambda i: "invalid" if (i < 12 or i >= 30) else "none")
    assert _counts(rows) == (12, 0), "premise"
    _clean(res)
    _, all_rows, _ = es.read_artifact(res.artifact)
    assert len(all_rows) == 35


def test_d_fires_at_11_of_30_parsed_naming_parsed(tmp_path: Path) -> None:
    """Boundary, firing: 11/30 parsed (all schema valid, so the parse count is what fires), stopped
    after exactly 30 records, message per T-0078 criterion 1."""
    res, rows = _bfcl_mix(tmp_path, lambda i: "valid" if (i < 11 or i >= 30) else "none")
    assert _counts(rows) == (11, 11), "premise"
    _fired(res, "d")
    _, all_rows, _ = es.read_artifact(res.artifact)
    assert len(all_rows) == 30, "the stack check runs exactly after record 30"
    _d_message(res.stopped, 11)
    _d_message(es.stopped_marker(res.artifact), 11)


def test_d_fires_at_11_of_30_parsed_when_all_are_schema_invalid(tmp_path: Path) -> None:
    """Guard absent: 11/30 parse and they are schema *invalid* -- the parse count alone fires."""
    res, rows = _bfcl_mix(tmp_path, lambda i: "invalid" if i < 11 else "none")
    assert _counts(rows) == (11, 0), "premise"
    _fired(res, "d")
    _d_message(res.stopped, 11)


def test_d_message_when_bfcl_has_no_parseable_output(tmp_path: Path) -> None:
    res, rows = _bfcl_mix(tmp_path, lambda i: "none")
    assert _counts(rows) == (0, 0), "premise"
    _fired(res, "d")
    _d_message(res.stopped, 0)


def _d_message(msg: str | None, k: int) -> None:
    """``d: unconstrained parsed k/30 = r < 0.4 after 30 records (stack broken?)`` (criterion 1);
    ``r`` is k/30 to the precision printed."""
    head = f"d: unconstrained parsed {k}/30 = "
    tail = " < 0.4 after 30 records (stack broken?)"
    assert msg is not None and msg.startswith(head) and msg.endswith(tail), msg
    r = msg[len(head) : len(msg) - len(tail)]
    digits = len(r.split(".", 1)[1]) if "." in r else 0
    assert abs(float(r) - k / 30) <= 0.5 * 10**-digits + 1e-12, msg
    assert "schema" not in msg, msg


def test_d_sudoku_message_still_names_solved(tmp_path: Path) -> None:
    """Sudoku's (d) is unchanged by T-0078: it counts solved against 0.5."""
    res, _ = _sudoku(
        tmp_path, lambda i, rid, b: "solved" if i < 14 else "wrong", arm="unconstrained", limit=40
    )
    _fired(res, "d")
    assert (res.stopped or "").startswith("d: unconstrained solved 14/30 = "), res.stopped
    assert "< 0.5 after 30 records" in (res.stopped or ""), res.stopped


# ------------------------------------------------------------------------------------------ (e)


def _capped(n_capped: int):  # type: ignore[no-untyped-def]
    by_id = es.sudoku_by_id("test")

    def decide(i: int, rid: str, b: object) -> Reply:
        return Reply(text=es.solved_text(by_id[rid][1]), steps=STEP_CAP if i < n_capped else 20)

    return decide


def test_e_fires_when_11_of_20_canvases_hit_the_step_cap(tmp_path: Path) -> None:
    res, _ = _sudoku(tmp_path, _capped(11), arm="perstep_mf", limit=25)
    _fired(res, "e")
    _, rows, _ = es.read_artifact(res.artifact)
    assert len(rows) == 20, "the step-cap check runs from record 20"


def test_e_does_not_fire_before_20_records(tmp_path: Path) -> None:
    """11 of 19 capped, run ends at 19: the check never runs (a check from record 19 would fire)."""
    res, sent = _sudoku(tmp_path, _capped(11), arm="perstep_mf", limit=19)
    _clean(res)
    assert sent == 19


def test_e_does_not_fire_at_exactly_half(tmp_path: Path) -> None:
    res, sent = _sudoku(tmp_path, _capped(10), arm="perstep_mf", limit=25)
    _clean(res)
    assert sent == 25


# ------------------------------------------------------------------------------------------ (f)


def _header_rate(tmp_path: Path, n_header: int) -> tuple[object, float | None, str]:
    by_id = es.sudoku_by_id("test")

    def decide(i: int, rid: str, b: object) -> Reply:
        grid = es.grid_text(by_id[rid][1])
        # the preserved unconstrained texts open with a bare `<channel|>`, never with `<|channel>`
        return Reply(text=(es.HEADER + grid) if i < n_header else ("<channel|>" + grid))

    res, _ = _sudoku(tmp_path, decide, arm="unconstrained", limit=24)
    regime, _, other = es.read_artifact(res.artifact)
    rate = next(
        (x["header_prefix_rate"] for x in [regime, *other] if "header_prefix_rate" in x), None
    )
    md = es.report().build([res.artifact])
    return res, rate, md


def test_f_zero_percent_header_rate_is_reported_and_does_not_stop(tmp_path: Path) -> None:
    res, rate, md = _header_rate(tmp_path, 0)
    _clean(res)
    assert rate == 0.0
    assert "header" in md.lower()


def test_f_nonzero_header_rate_is_reported_not_stopped(tmp_path: Path) -> None:
    """5 of the first 20 open with the header -> 0.25, still reported and not a stop."""
    res, rate, _ = _header_rate(tmp_path, 5)
    _clean(res)
    assert rate == pytest.approx(0.25)


# ------------------------------------------------------------------------------------------ (g)


def test_g_fires_over_the_wall_clock_budget(tmp_path: Path) -> None:
    by_id = es.sudoku_by_id("test")
    res, sent = _sudoku(
        tmp_path,
        lambda i, rid, b: Reply(text=es.solved_text(by_id[rid][1]), hang_s=0.3),
        limit=20,
        budget_s=1.0,
    )
    _fired(res, "g")
    assert sent < 20


def test_g_does_not_fire_within_budget(tmp_path: Path) -> None:
    by_id = es.sudoku_by_id("test")
    res, sent = _sudoku(
        tmp_path,
        lambda i, rid, b: Reply(text=es.solved_text(by_id[rid][1]), hang_s=0.05),
        limit=8,
        budget_s=60.0,
    )
    _clean(res)
    assert sent == 8


# ------------------------------------------------------------------------------------------ (h)


def _h(tmp_path: Path, *, dirty: bool, commit: str | None):  # type: ignore[no-untyped-def]
    repo, head = es.clean_repo(tmp_path)
    if dirty:
        (repo / "README").write_text("edited\n")
    return _sudoku(
        tmp_path, lambda i, rid, b: "solved", limit=3, repo=repo, expected_commit=commit or head
    )


def test_h_refuses_a_dirty_tree(tmp_path: Path) -> None:
    res, sent = _h(tmp_path, dirty=True, commit=None)
    _fired(res, "h")
    assert sent == 0


def test_h_refuses_a_commit_mismatch(tmp_path: Path) -> None:
    res, sent = _h(tmp_path, dirty=False, commit="0" * 40)
    _fired(res, "h")
    assert sent == 0


def test_h_starts_on_a_clean_tree_at_the_expected_commit(tmp_path: Path) -> None:
    res, sent = _h(tmp_path, dirty=False, commit=None)
    _clean(res)
    assert sent == 3
    regime, _, _ = es.read_artifact(res.artifact)
    head = subprocess.run(
        ["git", "-C", str(tmp_path / "repo"), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert regime["commit"] == head


# ------------------------------------------------------------------------------------------ E11 report


def test_stopped_artifact_table_row_says_stopped(tmp_path: Path) -> None:
    res, _ = _sudoku(
        tmp_path, lambda i, rid, b: "solved" if i < 14 else "wrong", arm="unconstrained", limit=40
    )
    _fired(res, "d")
    lines = es.table_lines(es.report().build([res.artifact]), "unconstrained")
    assert lines and all("stopped" in line.lower() for line in lines), lines


def test_complete_artifact_table_row_does_not_say_stopped(tmp_path: Path) -> None:
    res, _ = _sudoku(tmp_path, lambda i, rid, b: "solved", arm="unconstrained", limit=5)
    _clean(res)
    lines = es.table_lines(es.report().build([res.artifact]), "unconstrained")
    assert lines and not any("stopped" in line.lower() for line in lines), lines
