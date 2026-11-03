"""T-0082a criteria 9, 10 and 12: ``twopass.rescore`` (the offline re-score of stored pass-1 rows under the
stripping router), the CLI ``scripts/cd_toolcall_rescore.py``, the report's stripped-close line, and the scope guard.

The artifact is hand-built here (never the gitignored T-0080c evidence; criterion 11 on the real artifacts is the
verifier's). Six pass-1 rows on real test-split cases of ``tests/eval/toolcall/cases.jsonl`` (ground truth from the
file), each stored as the T-0080c driver wrote it -- the *unstripped* router fields, no ``draft_close`` -- with
``token_ids = encode(text) + [106]`` (pinned tokenizer) and a served ``content`` that deliberately disagrees with
``text`` (rescore must read ``text``):

====  =========  ==================================================  ===========================================
row   case       answer after the thought                            under the strip (hand-computed)
====  =========  ==================================================  ===========================================
A     search-11  ``<cc>`` exact search + ``</cc>``                   stripped ``</cc>``; ok, valid, exact
B     search-12  ``<cc>`` exact search                               unchanged; ok, valid, exact
C     click-11   ``<cc>`` click x 1500 + ``</cc>``                   stripped ``</cc>``; ok, schema-invalid
D     click-12   ``<cc>`` click (100, 100) outside bbox + ``</cc>\\n``  stripped ``</cc>\\n``; ok, valid, not exact
E     search-13  ``<cc>`` search truncated in a string: ``"mars</cc>``  unchanged (``</cc>`` not after ``}``)
F     chat-09    plain answer, no tag                                route none, unchanged
====  =========  ==================================================  ===========================================

Hence ``draft_only`` (tool cases 5, all routed): stored schema 1/5, exact 1/5 (B only); re-scored schema 3/5
(A, B, D), exact 2/5 (A, B); 3 drafts stripped. Pass-2 rows: two cells, ``T1/named/perstep_mar`` and the seed cell
``T1/named/perstep_mar+seed``, one row per routed case.

API pins where the task is silent (reported to the orchestrator): ``rescore(rows, tools, cases)`` takes the
artifact's pass rows (``read_artifact``'s second element), ``tools`` a ``ToolSet`` and ``cases`` the case dicts of
``cases.jsonl`` (a list, as the driver reads them), and returns a list of rows in input order. The CLI is run as a
subprocess (``sys.executable``, cwd = the repo, like ``test_cd_reselect.py``) with ``--artifact`` relative to
``--evidence-root``; the in-process rows load it by path and call ``main(argv) -> int`` (the ``cd_reselect.py``
pattern the task cites).

Regime: CPU, no GPU, no network; pinned tokenizer ``cc8d3a0c...`` (model rev 2ea8372); ``tools_v1.json`` sha
``bbfc53cc...``; cases sha ``460cd1c3...`` (MANIFEST).
"""

from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import io
import json
import os
import re
import socket
import stat
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.cd import _twopass_support as ts

REPO = ts.REPO
TOOL = REPO / "scripts" / "cd_toolcall_rescore.py"
TURN_CLOSE = 106
BASE_CELL = "T1/named/perstep_mar"
SEED_CELL = "T1/named/perstep_mar+seed"
CELLS = [BASE_CELL, SEED_CELL]
ARTIFACT_REL = "logs/t0080c-test/20261009T000000.000000Z-toolcall-test.jsonl"
SYSTEM_PROMPT_SHA = "fa5dace3bd1c9c89476c922c8e3c7bb4e8030d8a8fffc8ffd349982c903bbad3"

A_CALL = '{"name": "google_search", "arguments": {"query": "hiking zermatt"}}'
B_CALL = '{"name": "google_search", "arguments": {"query": "steam deck price"}}'
C_CALL = '{"name": "click_at", "arguments": {"x": 1500, "y": 470}}'
D_CALL = '{"name": "click_at", "arguments": {"x": 100, "y": 100}}'
E_DRAFT = '{"name": "google_search", "arguments": {"query": "mars</cc>'

#: case -> (spoken prefix, tag text + draft as the model wrote it, or None for no tag)
ANSWERS: dict[str, tuple[str, str | None]] = {
    "search-11": ("Searching. ", A_CALL + "</cc>"),
    "search-12": ("Looking it up. ", B_CALL),
    "click-11": ("Clicking. ", C_CALL + "</cc>"),
    "click-12": ("Clicking it. ", D_CALL + "</cc>\n"),
    "search-13": ("Searching. ", E_DRAFT),
    "chat-09": ("It is four.", None),
}
P1_IDS = list(ANSWERS)
ROUTED = [c for c in P1_IDS if ANSWERS[c][1] is not None]


def _call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": arguments}


#: The stored (T-0080c router: text after the tag) fields, written by hand.
#: case -> (draft_name, draft_call_text, draft_parse, draft_call, draft_schema_valid, draft_exact)
STORED: dict[str, tuple[Any, ...]] = {
    "search-11": ("google_search", A_CALL + "</cc>", "not JSON", None, False, False),
    "search-12": (
        "google_search",
        B_CALL,
        "ok",
        _call("google_search", query="steam deck price"),
        True,
        True,
    ),
    "click-11": ("click_at", C_CALL + "</cc>", "not JSON", None, False, False),
    "click-12": ("click_at", D_CALL + "</cc>\n", "not JSON", None, False, False),
    "search-13": ("google_search", E_DRAFT, "not JSON", None, False, False),
    "chat-09": (None, "", None, None, None, False),
}
#: The re-scored fields (criterion 10), written by hand.
#: case -> (draft_call_text, draft_close, draft_parse, draft_call, draft_schema_valid, draft_exact)
RESCORED: dict[str, tuple[Any, ...]] = {
    "search-11": (
        A_CALL,
        "</cc>",
        "ok",
        _call("google_search", query="hiking zermatt"),
        True,
        True,
    ),
    "search-12": (
        B_CALL,
        "",
        "ok",
        _call("google_search", query="steam deck price"),
        True,
        True,
    ),
    "click-11": (C_CALL, "</cc>", "ok", _call("click_at", x=1500, y=470), False, False),
    "click-12": (D_CALL, "</cc>\n", "ok", _call("click_at", x=100, y=100), True, False),
    "search-13": (E_DRAFT, "", "not JSON", None, False, False),
    "chat-09": ("", "", None, None, None, False),
}
STRIPPED = {"search-11", "click-11", "click-12"}


# --------------------------------------------------------------------------- the hand-built artifact


def encode(text: str) -> list[int]:
    from local_astra.decoding import vocab

    return [int(t) for t in vocab.encode(text)]


def _text(case: str) -> str:
    spoken, tagged = ANSWERS[case]
    head = f"<|channel>thought\nThe user asked for {case}.<channel|>"
    return head + spoken + ("" if tagged is None else "<cc>" + tagged)


def _case(case: str) -> Mapping[str, Any]:
    return ts.by_id()[case]


def p1_row(i: int, case: str) -> dict[str, Any]:
    text = _text(case)
    spoken, tagged = ANSWERS[case]
    name, draft, parse, call, valid, exact = STORED[case]
    c = _case(case)
    return {
        "pass": 1,
        "case": case,
        "kind": c["kind"],
        "tool": c.get("tool"),
        "error": None,
        "wall_ms": 900.0 + 10 * i,
        "prefix_hit_tokens": 4000,
        "prefix_query_tokens": 4100,
        "token_ids": [*encode(text), TURN_CLOSE],
        "text": text,
        # the served (parsed) content: deliberately not the answer -- rescore reads ``text``
        "content": "served content without any tag",
        "route": {
            "kind": "none" if tagged is None else "toolcall",
            "tag": None if tagged is None else "cc",
            "draft_name": name,
            "spoken_prefix": spoken if tagged is not None else text.split("<channel|>", 1)[1],
        },
        "draft_call_text": draft,
        "draft_parse": parse,
        "draft_call": call,
        "draft_schema_valid": valid,
        "draft_exact": exact,
    }


def p2_row(i: int, case: str, cell: str) -> dict[str, Any]:
    name, args = ts.gt_call(_case(case))
    row: dict[str, Any] = {
        "pass": 2,
        "case": case,
        "cell": cell,
        "thought": "T1",
        "scope": "named",
        "arm": cell.split("/")[2],
        "requests": 1,
        "text": "<|channel>thought\nok<channel|><cc>" + ts.call_text(name, args),
        "parse": "ok",
        "call": {"name": name, "arguments": ts.ordered_args(name, args)},
        "schema_valid": True,
        "exact": True,
        "cs": True,
        "error": None,
        "wall_ms": 300.0 + i,
        "steps": [7],
        "prefix_hit_tokens": 3900,
        "prefix_query_tokens": 4000,
        "close_101_pos": 3,
    }
    if cell == SEED_CELL:
        row.update(seed=True, seed_reason=None, seed_ids=[100, 101, 1], seed_trimmed=0)
        row["committed_ids"] = [[100, 101, 1]]
    return row


def pass_rows() -> list[dict[str, Any]]:
    rows = [p1_row(i, c) for i, c in enumerate(P1_IDS)]
    rows += [p2_row(i, c, cell) for cell in CELLS for i, c in enumerate(ROUTED)]
    return rows


def regime() -> dict[str, Any]:
    return {
        "stage": "test",
        "cells": list(CELLS),
        "n_cases": len(P1_IDS),
        "commit": "0dc91342a8c8bf33eb0a0a940bae123448873735",
        "system_prompt_sha256": SYSTEM_PROMPT_SHA,
        "tools_sha256": hashlib.sha256(
            (REPO / "local_astra/toolcall/tools_v1.json").read_bytes()
        ).hexdigest(),
        "cases_sha256": hashlib.sha256(ts.CASES_FILE.read_bytes()).hexdigest(),
        "stopped": None,
    }


def write_artifact(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    stopped: str | None = None,
    regime_extra: Mapping[str, Any] | None = None,
) -> str:
    """``stopped`` goes on the trailer; ``regime_extra`` overrides regime-line keys (``stopped`` included)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{**regime(), **(regime_extra or {})}, *rows, {"kind": "trailer", "stopped": stopped}]
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases() -> list[dict[str, Any]]:
    return [dict(c) for c in ts.all_cases()]


def rescore(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tp = ts.twopass()
    fn = getattr(tp, "rescore", None)
    if fn is None:
        raise AssertionError("T-0082a not implemented: twopass.rescore is missing")
    return [dict(r) for r in fn(list(rows), ts.toolset(), cases())]


def test_premises_of_the_fixture() -> None:
    """The hand-built rows are what the table says: real test-split cases, ``text`` round-trips through the
    pinned tokenizer and is the host decode of ``token_ids`` up to the stop, and the stored fields are what the
    *unstripped* contract (T-0080a criterion 2) gives -- checked here with the scorer and the regex prefix reading
    by hand, not with the router under test."""
    from local_astra.eval.cd import replay
    from local_astra.toolcall import score

    for case in P1_IDS:
        assert _case(case)["split"] == "test", case
        row = p1_row(0, case)
        assert replay.detokenize(row["token_ids"][:-1]) == row["text"], case
    for case in ROUTED:
        draft = STORED[case][1]
        assert _text(case).endswith("<cc>" + draft), case
        parsed = score.parse_call(draft)
        if STORED[case][2] == "ok":
            assert parsed == STORED[case][3]
        else:
            assert parsed == score.ParseFailure("not JSON"), case
    for case, (draft, close, parse, call, valid, exact) in RESCORED.items():
        if case not in ROUTED:
            continue
        assert draft + close == STORED[case][1], case
        parsed = score.parse_call(draft)
        assert (parse == "ok") is (not isinstance(parsed, score.ParseFailure)), case
        if parse == "ok":
            assert parsed == call
            assert score.schema_valid(parsed, ts.toolset()) is valid, case
            assert score.exact(parsed, _case(case)["gt"]) is exact, case


# =========================================================================== criterion 10: rescore


def test_rescore_recomputes_the_pass1_draft_fields_by_hand() -> None:
    """Criterion 10: each pass-1 row's ``route``, ``draft_call_text``, ``draft_close``, ``draft_parse``,
    ``draft_call``, ``draft_schema_valid`` and ``draft_exact`` are recomputed from its stored ``text`` (the table
    in the module docstring); every other field is kept."""
    rows = pass_rows()
    out = rescore(copy.deepcopy(rows))
    assert len(out) == len(rows)
    p1 = [r for r in out if r.get("pass") == 1]
    assert [r["case"] for r in p1] == P1_IDS
    recomputed = {
        "route",
        "draft_call_text",
        "draft_close",
        "draft_parse",
        "draft_call",
        "draft_schema_valid",
        "draft_exact",
    }
    for before, after in zip((r for r in rows if r["pass"] == 1), p1, strict=True):
        case = before["case"]
        draft, close, parse, call, valid, exact = RESCORED[case]
        got = (
            after.get("draft_call_text"),
            after.get("draft_close"),
            after.get("draft_parse"),
            after.get("draft_call"),
            after.get("draft_schema_valid"),
            after.get("draft_exact"),
        )
        assert got == (draft, close, parse, call, valid, exact), case
        assert after["route"] == before["route"], case  # same kind/tag/spoken/name (else a refusal)
        assert {k: v for k, v in after.items() if k not in recomputed} == {
            k: v for k, v in before.items() if k not in recomputed
        }, case


def test_rescore_counts_of_draft_only_are_the_hand_counts() -> None:
    """Criterion 10 on the 6-row artifact: ``draft_only`` derived from the stored rows is schema 1/5 and exact 1/5;
    from the re-scored rows schema 3/5 (A, B, D) and exact 2/5 (A, B); 3 rows carry a non-empty ``draft_close``."""
    tp = ts.twopass()
    rows = pass_rows()

    def counts(rs: Sequence[Mapping[str, Any]]) -> tuple[int, int, int, int]:
        d = tp.derive(rs, "draft_only")
        called = [r for r in d if r["called"]]
        tool = [r for r in d if r["tool_case"]]
        return (
            sum(r["schema_valid"] is True for r in called),
            len(called),
            sum(bool(r["exact"]) for r in tool),
            len(tool),
        )

    assert counts(rows) == (1, 5, 1, 5)
    out = rescore(copy.deepcopy(rows))
    assert counts(out) == (3, 5, 2, 5)
    assert {r["case"] for r in out if r.get("pass") == 1 and r.get("draft_close")} == STRIPPED


def test_rescore_returns_pass2_rows_untouched() -> None:
    """Criterion 10: pass-2 rows come back ``==`` the input (both cells, including the seed cell's)."""
    rows = pass_rows()
    want = [copy.deepcopy(r) for r in rows if r["pass"] == 2]
    out = rescore(copy.deepcopy(rows))
    assert [r for r in out if r.get("pass") == 2] == want


def test_rescore_reads_text_not_the_served_content() -> None:
    """Criterion 10 / implementation notes: the served ``content`` (here tag-free) is never routed -- a rescore
    that read it would route every row ``none``."""
    out = rescore(pass_rows())
    kinds = [r["route"]["kind"] for r in out if r.get("pass") == 1]
    assert kinds == ["toolcall"] * 5 + ["none"]


def test_rescore_refuses_a_text_that_is_not_the_decode_of_its_ids() -> None:
    """Criterion 10 (refusal): a stored ``text`` that differs from ``_decode(token_ids)`` raises ``ValueError`` --
    here row A's text without its ``</cc>`` (the ids still hold it), the edit a careless re-score would make."""
    rows = pass_rows()
    rows[0]["text"] = rows[0]["text"][: -len("</cc>")]
    with pytest.raises(ValueError):
        rescore(rows)
    rows2 = pass_rows()
    rows2[5]["text"] = rows2[5]["text"] + " "
    with pytest.raises(ValueError):
        rescore(rows2)


def test_rescore_refuses_an_unknown_case_id() -> None:
    """Review F3 (criterion 10): a pass-1 row whose case is not among ``cases`` has no ground truth -> ``ValueError``
    (not a ``KeyError``, not a silent score)."""
    tp = ts.twopass()
    rows = pass_rows()
    known = [c for c in cases() if c["id"] != "click-12"]
    with pytest.raises(ValueError):
        tp.rescore(rows, ts.toolset(), known)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("draft_name", "scroll"),
        ("draft_name", None),
        ("kind", "none"),
        ("tag", "native"),
        ("spoken_prefix", "Searching."),
    ],
)
def test_rescore_refuses_a_changed_route(field: str, value: Any) -> None:
    """Criterion 10 (refusal): if re-routing the stored text changes ``kind``, ``tag``, ``spoken_prefix`` or
    ``draft_name``, rescore raises ``ValueError``. The stored route of row A is edited; re-routing gives the
    original. (The happy-path rows above are the other direction: an unedited artifact never refuses.)"""
    rows = pass_rows()
    rows[0]["route"][field] = value
    with pytest.raises(ValueError):
        rescore(rows)


# =========================================================================== criterion 9: the report line


def _rescored_artifact(path: Path, ids: Sequence[str]) -> Path:
    """An artifact as the driver writes it after the strip: pass-1 rows with the re-scored fields (hand values,
    ``draft_close`` included), pass-2 rows of the base cell only."""
    rows: list[dict[str, Any]] = []
    for i, case in enumerate(ids):
        r = p1_row(i, case)
        draft, close, parse, call, valid, exact = RESCORED[case]
        r.update(
            draft_call_text=draft,
            draft_close=close,
            draft_parse=parse,
            draft_call=call,
            draft_schema_valid=valid,
            draft_exact=exact,
        )
        rows.append(r)
    rows += [p2_row(i, c, BASE_CELL) for i, c in enumerate(ids) if ANSWERS[c][1] is not None]
    path.parent.mkdir(parents=True, exist_ok=True)
    reg = {**regime(), "cells": [BASE_CELL], "n_cases": len(ids)}
    lines = [reg, *rows, {"kind": "trailer", "stopped": None}]
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    return path


def _close_lines(md: str) -> list[str]:
    return [ln for ln in md.splitlines() if "close" in ln.lower() or "</cc>" in ln]


def _strip_closes(path: Path, out: Path) -> Path:
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    for x in lines:
        if x.get("pass") == 1:
            x["draft_close"] = ""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    return out


def test_report_shows_drafts_with_a_stripped_close_per_artifact(tmp_path: Path) -> None:
    """Criterion 9: ``report()`` and ``render_markdown`` give one line per artifact, "drafts with a stripped close,
    k / routed": 3/5 for the six-case artifact (A, C, D of A..E routed), 1/2 for a second artifact of search-11 and
    search-12 (+ chat-09). The same artifacts with every ``draft_close`` emptied give 0/5 and 0/2, and the report
    dict differs from theirs (the figure is in ``report()``, not only in the markdown)."""
    tp = ts.twopass()
    a1 = _rescored_artifact(tmp_path / "a" / "one.jsonl", P1_IDS)
    a2 = _rescored_artifact(tmp_path / "a" / "two.jsonl", ["search-11", "search-12", "chat-09"])
    b1 = _strip_closes(a1, tmp_path / "b" / "one.jsonl")
    b2 = _strip_closes(a2, tmp_path / "b" / "two.jsonl")
    rep_a = tp.report([a1, a2])
    rep_b = tp.report([b1, b2])
    md_a, md_b = tp.render_markdown(rep_a), tp.render_markdown(rep_b)
    lines_a, lines_b = _close_lines(md_a), _close_lines(md_b)
    assert len(lines_a) == 2, lines_a
    assert re.search(r"\b3\s*/\s*5\b", lines_a[0]) and re.search(r"\b1\s*/\s*2\b", lines_a[1]), (
        lines_a
    )
    assert len(lines_b) == 2, lines_b
    assert re.search(r"\b0\s*/\s*5\b", lines_b[0]) and re.search(r"\b0\s*/\s*2\b", lines_b[1]), (
        lines_b
    )
    strip = {"artifacts"}
    assert {k: v for k, v in rep_a.items() if k not in strip} != {
        k: v for k, v in rep_b.items() if k not in strip
    }


# =========================================================================== criterion 10: the CLI


def _evidence(tmp_path: Path, *, stopped: str | None = None) -> tuple[Path, str]:
    ev = tmp_path / "evidence"
    sha = write_artifact(ev / ARTIFACT_REL, pass_rows(), stopped=stopped)
    return ev, sha


def _argv(ev: Path, sha: str, out: Path) -> list[str]:
    return [
        "--evidence-root",
        str(ev),
        "--artifact",
        ARTIFACT_REL,
        "--sha256",
        sha,
        "--out-dir",
        str(out),
    ]


def _run(ev: Path, sha: str, out: Path) -> subprocess.CompletedProcess[str]:
    assert TOOL.is_file(), f"T-0082a not implemented: {TOOL.relative_to(REPO)} does not exist"
    cmd = [sys.executable, str(TOOL), *_argv(ev, sha, out)]
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _refused(proc: subprocess.CompletedProcess[str], out: Path, before: Sequence[str] = ()) -> None:
    assert proc.returncode == 2, (
        f"expected a refusal (exit 2), got {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}"
    )
    written = sorted(str(p.relative_to(out)) for p in out.rglob("*")) if out.exists() else []
    assert written == sorted(before), f"a refusal wrote {written}"


def _read_only(root: Path) -> Iterator[None]:
    """Make every file and directory under ``root`` read-only for the duration; restore after."""
    paths = [root, *root.rglob("*")]
    modes = {p: p.stat().st_mode for p in paths}
    for p in paths:
        p.chmod(modes[p] & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        yield
    finally:
        for p in reversed(paths):
            p.chmod(modes[p])


def _keys_and_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield str(k)
            yield from _keys_and_strings(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from _keys_and_strings(v)
    elif isinstance(obj, str):
        yield obj


def test_cli_writes_the_rescore_report(tmp_path: Path) -> None:
    """Criterion 10: on a read-only evidence tree, exit 0 and exactly ``report-rescore.json`` and ``.md`` in the
    new out-dir. The report has ``draft_only`` and ``cd_on_invalid:`` / ``cd_always:`` for the unseeded cell; the
    seed cell is listed as stale and has no policy of its own. The markdown's ``draft_only`` line shows the
    re-scored 3/5 schema-valid and 2/5 exact (not the stored 1/5), and the counterfactual caveat (implementation
    notes). Nothing under the evidence root changes."""
    ev, sha = _evidence(tmp_path)
    out = tmp_path / "out" / "rescore"
    before = _snapshot(ev)
    ro = _read_only(ev)
    next(ro)
    try:
        proc = _run(ev, sha, out)
    finally:
        next(ro, None)
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    assert sorted(p.name for p in out.iterdir()) == ["report-rescore.json", "report-rescore.md"]
    assert _snapshot(ev) == before
    rep = json.loads((out / "report-rescore.json").read_text())
    strings = set(_keys_and_strings(rep))
    for policy in ("draft_only", f"cd_on_invalid:{BASE_CELL}", f"cd_always:{BASE_CELL}"):
        assert policy in strings, policy
    for policy in (f"cd_on_invalid:{SEED_CELL}", f"cd_always:{SEED_CELL}"):
        assert policy not in strings, f"{policy} is in the policy table"
    assert SEED_CELL in strings
    assert any("stale" in s.lower() for s in strings), "the seed cell is not flagged stale"
    # Review F1: the headline counters, against the fixture's hand values (module docstring table).
    changes = rep["changes"]
    want = {
        "stripped": {"search-11", "click-11", "click-12"},
        "newly_schema_valid": {"search-11", "click-12"},
        "newly_exact": {"search-11"},
        # Nothing can be lost, by construction: a stored draft that was schema-valid or exact parsed, so it does
        # not end in ``</cc>`` (criterion 4: the strip never changes a parseable draft) and is re-scored unchanged.
        "lost_schema_valid": set(),
        "lost_exact": set(),
    }
    for key, cases_ in want.items():
        assert set(changes[key]["cases"]) == cases_, key
        assert changes[key]["n"] == len(cases_), key
    assert rep["stripped_close"] == {"k": 3, "n": 5}
    assert rep["routed"] == 5
    md = (out / "report-rescore.md").read_text()
    assert "stale" in md.lower() and SEED_CELL in md
    assert "counterfactual" in md.lower()
    draft_lines = [
        ln for ln in md.splitlines() if "draft_only" in ln and "/5" in ln.replace(" ", "")
    ]
    assert any(
        re.search(r"\b3\s*/\s*5\b", ln) and re.search(r"\b2\s*/\s*5\b", ln) for ln in draft_lines
    ), draft_lines


def test_cli_refuses_a_sha_mismatch(tmp_path: Path) -> None:
    """Criterion 10: ``--sha256`` that is not the artifact's -> exit 2, nothing written (out-dir not created)."""
    ev, sha = _evidence(tmp_path)
    out = tmp_path / "out" / "rescore"
    wrong = hashlib.sha256(b"not the artifact").hexdigest()
    assert wrong != sha
    _refused(_run(ev, wrong, out), out)
    assert not out.exists()


def test_cli_refuses_an_existing_out_dir(tmp_path: Path) -> None:
    """Criterion 10: an out-dir that already exists (empty) -> exit 2, nothing written into it."""
    ev, sha = _evidence(tmp_path)
    out = tmp_path / "out" / "rescore"
    out.mkdir(parents=True)
    _refused(_run(ev, sha, out), out)


def test_cli_refuses_a_stopped_artifact(tmp_path: Path) -> None:
    """Criterion 10: an artifact whose trailer has ``stopped`` set -> exit 2, nothing written."""
    ev, sha = _evidence(
        tmp_path, stopped="engine: http_error (case search-13, cell T1/named/perstep_mar)"
    )
    out = tmp_path / "out" / "rescore"
    _refused(_run(ev, sha, out), out)
    assert not out.exists()


def test_cli_refuses_when_rescore_refuses(tmp_path: Path) -> None:
    """Ruling A6 (criterion 10): a ``rescore`` ``ValueError`` reached through the CLI -- here row A's stored
    ``text`` without its ``</cc>`` while its ids still hold it (text != decode(ids)) -- is a refusal: exit 2, a
    ``REFUSED:`` message, nothing written. The sha passed is the tampered artifact's own, so only the re-score
    can refuse."""
    rows = pass_rows()
    rows[0]["text"] = rows[0]["text"][: -len("</cc>")]
    ev = tmp_path / "evidence"
    sha = write_artifact(ev / ARTIFACT_REL, rows)
    out = tmp_path / "out" / "rescore"
    proc = _run(ev, sha, out)
    _refused(proc, out)
    assert not out.exists()
    assert "REFUSED:" in proc.stdout + proc.stderr, (proc.stdout[-1000:], proc.stderr[-1000:])


@pytest.mark.parametrize(
    "override",
    [
        {"cases_sha256": hashlib.sha256(b"another eval set").hexdigest()},
        {"tools_sha256": hashlib.sha256(b"another tools file").hexdigest()},
        {"stopped": "engine: http_error (case search-12, cell T1/named/perstep_mar)"},
    ],
    ids=[
        "cases_sha_not_the_artifacts",
        "tools_sha_not_the_artifacts",
        "stopped_on_the_regime_line",
    ],
)
def test_cli_refuses_a_regime_that_does_not_match(tmp_path: Path, override: dict[str, Any]) -> None:
    """Review F2/F3 (criterion 10, implementation notes): the artifact's regime names a cases sha256 or a
    ``tools_v1.json`` sha256 other than the files the re-score would read, or carries ``stopped`` on the regime line
    itself (not only the trailer) -> exit 2, ``REFUSED:``, nothing written."""
    ev = tmp_path / "evidence"
    sha = write_artifact(ev / ARTIFACT_REL, pass_rows(), regime_extra=override)
    assert json.loads((ev / ARTIFACT_REL).read_text().splitlines()[0]).items() >= override.items()
    out = tmp_path / "out" / "rescore"
    proc = _run(ev, sha, out)
    _refused(proc, out)
    assert not out.exists()
    assert "REFUSED:" in proc.stdout + proc.stderr


def test_cli_refuses_cases_not_at_the_manifest_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review F2: ground truth is read at the MANIFEST-pinned sha. A ``cases.jsonl`` whose bytes differ from
    ``MANIFEST.json``'s ``cases_sha256`` -- even when the artifact's regime names those very bytes -- -> exit 2,
    ``REFUSED:``, nothing written. In-process: the CLI's ``SET_DIR`` (a module constant; pin noted) points at a
    copy of the eval set with one ground truth edited."""
    cli = _load_cli()
    set_dir = tmp_path / "toolcall"
    set_dir.mkdir()
    (set_dir / "MANIFEST.json").write_bytes((ts.SET_DIR / "MANIFEST.json").read_bytes())
    edited = ts.CASES_FILE.read_text().replace('"hiking", "zermatt"', '"hiking", "matterhorn"', 1)
    assert edited != ts.CASES_FILE.read_text(), "fixture: the edit did not apply"
    (set_dir / "cases.jsonl").write_text(edited)
    monkeypatch.setattr(cli, "SET_DIR", set_dir)
    ev = tmp_path / "evidence"
    sha = write_artifact(
        ev / ARTIFACT_REL,
        pass_rows(),
        regime_extra={"cases_sha256": hashlib.sha256(edited.encode()).hexdigest()},
    )
    out = tmp_path / "out" / "rescore"
    rc = cli.main(_argv(ev, sha, out))
    assert rc == 2
    assert not out.exists()
    captured = capsys.readouterr()
    assert "REFUSED:" in captured.out + captured.err


def _load_cli() -> Any:
    assert TOOL.is_file(), f"T-0082a not implemented: {TOOL.relative_to(REPO)} does not exist"
    name = "_t0082a_cd_toolcall_rescore"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        spec.loader.exec_module(m)
    finally:
        sys.modules.pop(name, None)
    assert hasattr(m, "main"), (
        "scripts/cd_toolcall_rescore.py has no main(argv) (cd_reselect.py pattern)"
    )
    return m


_WRITE_MODE = re.compile(r"[wax+]")


def test_cli_never_selects_never_connects_and_opens_inputs_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criteria 10 and 12: in-process, ``twopass.select`` raises if called, every socket connect raises, and every
    ``open`` of a path under the evidence root or the eval set is recorded -- the run exits 0, and no input was
    opened in a writing mode."""
    tp = ts.twopass()
    ev, sha = _evidence(tmp_path)
    out = tmp_path / "out" / "rescore"
    opened: list[tuple[str, str]] = []
    real_open = builtins.open
    watched = (str(ev.resolve()), str(ts.SET_DIR.resolve()))

    def recording_open(file: Any, mode: str = "r", *a: Any, **kw: Any) -> Any:
        if isinstance(file, str | os.PathLike):
            p = str(Path(os.fspath(file)).resolve())
            if p.startswith(watched):
                opened.append((p, mode))
        return real_open(file, mode, *a, **kw)

    def no_select(*a: Any, **kw: Any) -> Any:
        raise AssertionError("criterion 10: select must not be called by the re-score")

    def no_connect(*a: Any, **kw: Any) -> Any:
        raise AssertionError("criterion 12: no network")

    monkeypatch.setattr(tp, "select", no_select)
    monkeypatch.setattr(socket.socket, "connect", no_connect)
    monkeypatch.setattr(builtins, "open", recording_open)
    monkeypatch.setattr(io, "open", recording_open)
    cli = _load_cli()
    rc = cli.main(_argv(ev, sha, out))
    assert rc == 0, rc
    assert (out / "report-rescore.json").is_file()
    assert any(p.endswith(".jsonl") and p.startswith(str(ev.resolve())) for p, _ in opened), (
        "fixture: the artifact was never seen opened"
    )
    assert [(p, m) for p, m in opened if _WRITE_MODE.search(m)] == []


# =========================================================================== criterion 12: scope guard

UNCHANGED = [
    "local_astra/toolcall/score.py",
    "local_astra/toolcall/tools.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/decoding",
    "local_astra/contracts",
    "local_astra/conductor",
    "plugin",
    "local_astra/bench/thresholds.py",
    "scripts/locked/t0080_toolcall.sh",
    "docs/cd/toolcall-v1.md",
]


#: T-0082a's merge commit a8757f9; BASE = ``git merge-base a8757f9^1 a8757f9^2``, TIP = ``a8757f9^2`` (T-0094 criterion
#: 2). Not the declared fork point ec9621d: ec9621d..64a7a00 is a docs commit on main. Over this closed range no
#: later-task exemption is needed (T-0084a, T-0086 came later).
T0082A_MERGE = "a8757f9"
T0082A_BASE = "64a7a00"
T0082A_TIP = "73a87c9"


@pytest.mark.parametrize("rel", UNCHANGED)
def test_not_owned_paths_are_unchanged(rel: str) -> None:
    """Criterion 12 (negative) over T-0082a's closed range (T-0094): nothing under ``rel`` differs between BASE and
    TIP. The untracked-file half of the old guard is retired: it only meant something while the branch was open."""
    gr.prove_range(T0082A_BASE, T0082A_TIP, T0082A_MERGE)
    changed = gr.changed(T0082A_BASE, T0082A_TIP, rel)
    assert changed == [], changed


def test_the_t0082a_range_guard_has_power() -> None:
    """T-0094 criterion 4 (adversarial): over ``BASE..6a8600b`` the predicate over ``UNCHANGED`` reports
    ``conductor/executor.py`` (T-0089/T-0092)."""
    gr.require_commits(T0082A_BASE, gr.LATER)
    assert "local_astra/conductor/executor.py" in gr.changed(T0082A_BASE, gr.LATER, *UNCHANGED)


def test_system_prompt_is_unchanged() -> None:
    """Criterion 12: ``twopass.SYSTEM_PROMPT`` keeps its sha256 (``fa5dace3...``, docs/cd/toolcall-v1.md)."""
    tp = ts.twopass()
    assert hashlib.sha256(tp.SYSTEM_PROMPT.encode("utf-8")).hexdigest() == SYSTEM_PROMPT_SHA


# =========================================================================== T-0085 criterion 5: stored evidence

from tests.cd import _native_artifacts as _na  # noqa: E402
from tests.unit._toolcall_support import (  # noqa: E402
    CLICK01_NATIVE_IDS,
    CLICK01_NATIVE_TEXT,
    CLICK01_SPOKEN,
    CLICK01_STORED_SPOKEN,
)


def _stored_rows(key: str) -> list[dict[str, Any]]:
    _regime, rows, _other = ts.twopass().read_artifact(_na.stored(key))
    return [dict(r) for r in rows]


@pytest.mark.parametrize("key", list(_na.STORED))
def test_t0085_rescore_of_the_stored_cc_artifacts_keeps_every_route(key: str) -> None:
    """Criterion 5: on T-0080c dev/test and T-0082c dev/test, ``rescore`` refuses nothing and gives route fields
    identical to the stored ones (their spoken prefixes hold 0 channel markers: the premise)."""
    rows = _stored_rows(key)
    p1 = [r for r in rows if r.get("pass") == 1]
    assert p1
    assert not any("<channel|>" in str((r.get("route") or {}).get("spoken_prefix")) for r in p1), (
        "premise"
    )
    got = [r for r in rescore(rows) if r.get("pass") == 1]
    assert [r["route"] for r in got] == [r["route"] for r in p1]


@pytest.mark.parametrize(
    ("key", "want"),
    [
        (
            "t0080c-test",
            {"stripped": 33, "valid": 25, "exact": 23, "schema": (52, 76), "ex": (47, 80)},
        ),
        ("t0080c-dev", {"stripped": 10}),
    ],
)
def test_t0085_t0082a_criterion_11_counts_are_unchanged(key: str, want: dict[str, Any]) -> None:
    """Criterion 5: T-0082a criterion 11's counts (review B's, by hand) on the T-0080c artifacts: 33 drafts stripped,
    25 newly schema-valid, 23 newly exact, ``draft_only'`` schema 52/76 and exact 47/80 on test; 10 stripped on dev."""
    rows = _stored_rows(key)
    p1 = [r for r in rows if r.get("pass") == 1]
    got = rescore(rows)
    q1 = [r for r in got if r.get("pass") == 1]
    assert sum(1 for r in q1 if r.get("draft_close")) == want["stripped"]
    if "valid" not in want:
        return
    pairs = list(zip(p1, q1, strict=True))
    assert (
        sum(1 for a, b in pairs if b.get("draft_schema_valid") and not a.get("draft_schema_valid"))
        == 25
    )
    assert sum(1 for a, b in pairs if b.get("draft_exact") and not a.get("draft_exact")) == 23
    tool = [x for x in ts.twopass().derive(got, "draft_only") if x["tool_case"]]
    called = [x for x in tool if x["called"]]
    assert (sum(bool(x["schema_valid"]) for x in called), len(called)) == want["schema"]
    assert (sum(bool(x["exact"]) for x in tool), len(tool)) == want["ex"]


def _click01_row(spoken: str) -> dict[str, Any]:
    """The click-01 native pass-1 row as the 9aee66d driver stored it (text and ids verbatim), with ``spoken``."""
    return {
        "pass": 1,
        "case": "click-01",
        "format": "native",
        "error": None,
        "token_ids": list(CLICK01_NATIVE_IDS),
        "text": CLICK01_NATIVE_TEXT,
        "route": {"kind": "toolcall", "tag": "native", "draft_name": None, "spoken_prefix": spoken},
    }


def test_t0085_rescore_refuses_the_click01_row_naming_spoken_prefix() -> None:
    """Criterion 5: the row as stored at 9aee66d (``spoken_prefix`` with the stray close) re-routes to a different
    ``spoken_prefix`` -> ``ValueError`` naming it."""
    with pytest.raises(ValueError, match="spoken_prefix"):
        rescore([_click01_row(CLICK01_STORED_SPOKEN)])


def test_t0085_rescore_accepts_the_click01_row_with_the_clean_prefix() -> None:
    """Criterion 5, the other direction: the same row with the template-clean prefix re-scores without refusal."""
    (row,) = rescore([_click01_row(CLICK01_SPOKEN)])
    assert row["route"]["spoken_prefix"] == CLICK01_SPOKEN
