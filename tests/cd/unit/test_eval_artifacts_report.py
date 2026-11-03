"""T-0070 criterion 9: ``artifacts/cd/<ts>-<task>-<cut>-<arm>.jsonl`` -- a regime line, then one row per
record -- and a report whose markdown tables are built from artifacts only.

Expected report figures are Wilson 95 % intervals hand-checked against diffgemma-fa RESULTS' published
intervals (same formula, 3 d.p.): 10/20 -> 0.500 [0.299, 0.701]; 15/20 -> 0.750 [0.531, 0.888];
9/20 -> 0.450 [0.258, 0.658]; 14/20 -> 0.700 [0.481, 0.855]. Regime: CPU, fake server.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from tests.cd import _compile_support as cs
from tests.cd import _eval_support as es
from tests.cd.fakes.vllm_server import Reply

ROW_KEYS = {
    "id",
    "text",
    "committed_ids",
    "parse",
    "scores",
    "cs",
    "steps",
    "wall_ms",
    "bucket",
    "error",
    "knobs",
}
EP_KEYS = {"shadow_argmax_ids", "argmax_in_language", "map_equals_argmax"}


def test_artifact_name_regime_and_rows_end_projection(tmp_path: Path) -> None:
    cs.warm_sudoku("test")
    by_id = es.sudoku_by_id("test")

    def decide(i: int, rid: str, b: object) -> Reply:
        return Reply(
            text=es.solved_text(by_id[rid][1]),
            canvases=2,
            steps=17,
            shadow_argmax_ids=[7, 7, 1],
            argmax_in_language=False,
            map_equals_argmax=False,
        )

    with es.fake(es.sudoku_script("test", decide), tmp_path) as srv:
        res = es.run(srv, tmp_path, task="sudoku", cut="test", arm="end_projection", limit=3)
    assert res.exit_code == 0, res.stopped
    path = Path(res.artifact)
    assert path.parent == tmp_path / "artifacts"
    assert re.fullmatch(r".+-sudoku-test-end_projection\.jsonl", path.name), path.name
    assert not path.name.startswith("-")
    regime, rows, _ = es.read_artifact(path)
    for key in (
        "commit",
        "versions",
        "arm",
        "knobs",
        "task",
        "cut",
        "n",
        "max_tokens",
        "calibration_sha",
    ):
        assert key in regime, key
    assert (
        regime["task"] == "sudoku" and regime["cut"] == "test" and regime["arm"] == "end_projection"
    )
    assert regime["n"] == 3 and regime["max_tokens"] == 256
    assert regime.get("enable_thinking", regime.get("thinking")) is False
    assert regime["calibration_sha"] is None
    assert regime["versions"], "the regime names the versions it ran against"
    assert len(rows) == 3
    for row in rows:
        assert set(row) >= ROW_KEYS | EP_KEYS, sorted(ROW_KEYS | EP_KEYS - set(row))
        assert row["steps"] == [17, 17]  # steps per canvas
        assert row["bucket"] == 64
        assert row["wall_ms"] > 0
        assert row["argmax_in_language"] == [False, False] or row["argmax_in_language"] is False
        assert row["parse"] is True and row["cs"] is True and row["error"] is None


def test_regime_records_the_calibration_sha_for_mar(tmp_path: Path) -> None:
    cs.warm_sudoku("test")
    calib = es.freeze_calibration(tmp_path)
    with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), tmp_path) as srv:
        res = es.run(
            srv, tmp_path, task="sudoku", cut="test", arm="perstep_mar", limit=2, calibration=calib
        )
    assert res.exit_code == 0, res.stopped
    regime, rows, _ = es.read_artifact(res.artifact)
    assert regime["calibration_sha"] == hashlib.sha256(calib.read_bytes()).hexdigest()
    assert all(r.get("shadow_argmax_ids") is None for r in rows)


def _report_run(tmp_path: Path) -> Path:
    """Unconstrained, n = 20: 10 solved, 5 wrong grids, 5 with no grid (unparsed)."""

    def decide(i: int, rid: str, b: object) -> Reply | str:
        if i < 10:
            return "solved"
        if i < 15:
            return "wrong"
        return Reply(text="<channel|>Sorry, I cannot solve this.")

    with es.fake(es.sudoku_script("test", decide), tmp_path) as srv:
        res = es.run(srv, tmp_path, task="sudoku", cut="test", arm="unconstrained", limit=20)
    assert res.exit_code == 0, res.stopped  # (d) looks after 30 records; n = 20 never reaches it
    return Path(res.artifact)


def test_report_puts_parse_rate_beside_solve_rate_with_wilson(tmp_path: Path) -> None:
    md = es.report().build([_report_run(tmp_path)])
    assert "0.500 [0.299, 0.701]" in md  # solved 10/20
    assert "0.750 [0.531, 0.888]" in md  # parsed 15/20
    solved_line = next(line for line in md.splitlines() if "0.500 [0.299, 0.701]" in line)
    assert "0.750 [0.531, 0.888]" in solved_line, (
        "never a solve rate without the parse rate on the same row"
    )


def test_report_is_built_from_the_artifact_alone(tmp_path: Path) -> None:
    """Copy the artifact elsewhere, flip one solved row and one parsed row: the report follows the file."""
    src = _report_run(tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    dst = other / src.name
    shutil.copy(src, dst)
    assert es.report().build([dst]) == es.report().build([src])
    lines = dst.read_text().splitlines()
    out, flipped = [lines[0]], 0
    for line in lines[1:]:
        row = json.loads(line)
        if "id" in row and row["scores"].get("solved") and not flipped:
            row["scores"]["solved"] = False
            flipped = 1
        out.append(json.dumps(row))
    dst.write_text("\n".join(out) + "\n")
    src.unlink()  # nothing may be read from the original run
    md = es.report().build([dst])
    assert "0.450 [0.258, 0.658]" in md  # solved 9/20
    assert "0.500 [0.299, 0.701]" not in md


# ------------------------------------------------------------------ E2: both normalisations, labelled


def _bfcl_three(tmp_path: Path, arm: str, texts: dict[str, str]) -> Path:
    resolve = es.BfclResolver("test")

    def script(i: int, body: dict) -> Reply:  # type: ignore[type-arg]
        return Reply(text=texts[resolve(body)])

    with es.fake(script, tmp_path) as srv:
        res = es.run(srv, tmp_path, task="bfcl", cut="test", arm=arm, limit=3)
    assert res.exit_code == 0, res.stopped
    return Path(res.artifact)


def test_e2_unconstrained_row_scored_both_ways_and_report_prints_both(tmp_path: Path) -> None:
    """Record 3 of the cut (live_simple_2-2-0) is served ``"Comfort"`` for enum ``comfort``: schema-valid
    under the grammar's normalisation, invalid under the published one; the report prints the
    unconstrained arm twice, one line labelled as the published normalisation."""
    texts = {
        r["id"]: r["text"]
        for r in cs.preserved("eval_bfcl_live_simple_unconstrained_sample")["rows"]
    }
    path = _bfcl_three(tmp_path, "unconstrained", texts)
    _, rows, _ = es.read_artifact(path)
    row = next(r for r in rows if r["id"] == "live_simple_2-2-0")
    assert row["scores"]["schema_valid"] is True
    assert row["scores_published"]["schema_valid"] is False
    lines = es.table_lines(es.report().build([path]), "unconstrained")
    assert len(lines) >= 2, lines
    assert any("published" in line.lower() for line in lines), lines
    assert len(set(lines)) == len(lines)


def test_e2_constrained_row_scored_with_the_grammar_normalisation(tmp_path: Path) -> None:
    """The preserved map arm's own text for live_simple_2-2-0 writes ``"Comfort"`` and is accepted by the
    ci_enums grammar; the constrained row must not be scored schema-invalid for it."""
    texts = {r["id"]: r["text"] for r in cs.preserved("exp_e5_grammar130_map")["rows"]}
    assert '"Comfort"' in texts["live_simple_2-2-0"]
    path = _bfcl_three(tmp_path, "end_projection", texts)
    _, rows, _ = es.read_artifact(path)
    row = next(r for r in rows if r["id"] == "live_simple_2-2-0")
    assert row["error"] is None and row["cs"] is True
    assert row["scores"]["schema_valid"] is True
    if "scores_published" in row:
        assert row["scores_published"]["schema_valid"] is False
