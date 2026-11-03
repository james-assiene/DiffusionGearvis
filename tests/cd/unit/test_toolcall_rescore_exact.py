"""T-0086 part B (criteria 6-10): ``scripts/cd_toolcall_rescore_exact.py``, the offline re-score of stored two-pass
tool-call artifacts under the plausible-query ``exact`` rule.

Expectations come from hand construction, from the frozen 4c7dfdc ``exact`` (``tests/unit/_frozen_score_4c7dfdc.py``)
and, for the stored artifacts, from the artifacts' own flags -- never from the new implementation.

**The hand-built artifact** (dev stage, mixed formats, real dev cases of ``tests/eval/toolcall/cases.jsonl``):

- Tool cases ``search-01`` (keywords weather, lisbon), ``search-02`` (british museum), ``click-01``, ``click-02``; no-tool
  ``chat-01``. n = 4 tool cases per policy.
- Cells: cc ``T1/named/{end_projection, perstep_mar, perstep_mar_qargmax, perstep_mar+seed}``; native
  ``T4/named/{same four arms}``; and only ``native/T1/named/perstep_mar`` and ``native/T2/named/perstep_mar`` for
  T1/T2 (so θ_n = T1's step-2 family is *not* run).
- Pass 1, cc: every tool case routed ``<cc>``, schema-valid and old-exact, stored with ``text`` / ``token_ids`` (pinned
  tokenizer) so ``twopass.rescore`` re-scores it; ``search-01``'s draft query is ``"I will search for the weather in
  Lisbon"`` (narration). Wall 1000 ms.
- Pass 1, native: every tool case routed, valid, exact, with the reference call; stored without ``text`` (rescore
  returns such rows as they are). Wall 800 ms.
- Pass 2: every cell on every tool case, valid, CS true, 5 steps, wall 500 ms (``native/T2/named/perstep_mar``:
  600 ms); the reference call except these queries (each keeps every keyword, so the old flag is True):

  ==================================  ==========  =================================================  ==================
  cell                                case        query                                              reason
  ==================================  ==========  =================================================  ==================
  T1/named/perstep_mar                search-02   ``british museum | opening hours``                 markup
  native/T4/named/end_projection      search-01   ``I will search for the weather in Lisbon ...``    narration
  native/T4/named/end_projection      search-02   ``british museum`` + U+200B                        control character
  native/T4/named/perstep_mar         search-01   the T-0084f shape (backtick, ``{``, ``\\[``)        markup
  native/T4/named/perstep_mar         search-02   ``british museum call:google_search``              tool syntax
  native/T4/named/perstep_mar_qargmax search-01   ``weather lisbon google_search``                   tool syntax
  ==================================  ==========  =================================================  ==================

Hence 7 flipped rows (pass 1: 1; pass 2: 6), reasons narration 2, markup 2, tool syntax 2, control character 1, and
the old/new exact table ``EXPECT`` below. The counterfactual ``select_format(dev)``, by hand:

- old: step 1 over ``cd_always:native/{T1,T2,T4}/named/perstep_mar`` -- all 4/4, T2 slower (600 > 1.05 x 500), T1 and
  T4 tie on wall and steps, so T4 by order; step 2 over T4's family: every policy eligible at 4/4; the lowest e2e p50
  is the native draft (800 ms) shared by ``draft_only:native`` and ``cd_on_invalid:native/T4/...``; kind order picks
  ``draft_only:native`` (winner native).
- new: T4's perstep_mar falls to 2/4 < 4/4 - 1/10, T2 is out on wall, so θ_n = T1; T1's family is not run, so step 2 is
  ``not computable (cells not run: native/T1/named/end_projection, native/T1/named/perstep_mar_qargmax,
  native/T1/named/perstep_mar+seed)``.

**Report contract (K-notes; the task names the content, not the keys -- reported to the orchestrator):**
``report-rescore-exact.json`` holds ``task`` = ``"T-0086"``, ``caveat``, ``policies[<policy>]["old"|"new"][<scope>]
["exact"] = {"k", "n"}`` with scope ``"all"`` or ``"google_search"`` (the ``_policy_table`` entry shape of
``scripts/cd_toolcall_rescore.py``), ``flipped`` = a list of ``{"pass", "case", "cell", "reason"}`` (``cell`` of a
pass-1 row is free), ``reasons`` = ``{reason: count}`` over the flipped rows, and ``selection`` with ``old`` and
``new`` entries. For pass 1, "old" means the stored row re-scored by ``twopass.rescore`` under the 4c7dfdc rule
(router held fixed): on the T-0080c artifacts, whose flags predate the T-0082a strip, the old ``draft_only`` is
T-0082a's re-scored 27/40 (dev) and 47/80 (test), not the stored 18/40 and 24/80.

Regime: CPU, no GPU, no network; pinned tokenizer ``cc8d3a0c...`` (rev 2ea8372); ``tools_v1.json`` ``bbfc53cc...``;
cases ``460cd1c3...``. Stored artifacts: n = 1 each, shas in ``STORED``.
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
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.cd import _native_artifacts as na
from tests.cd import _twopass_support as ts
from tests.unit import _frozen_score_4c7dfdc as old

REPO = ts.REPO
TOOL = REPO / "scripts" / "cd_toolcall_rescore_exact.py"
JSON_NAME = "report-rescore-exact.json"
MD_NAME = "report-rescore-exact.md"
LABEL = "counterfactual; not a decision"
TURN_CLOSE = 106
ARTIFACT_REL = "logs/t0086-mini/20261011T000000.000000Z-toolcall-dev.jsonl"

S1, S2, C1, C2, N1 = "search-01", "search-02", "click-01", "click-02", "chat-01"
TOOL_CASES = (S1, S2, C1, C2)
ARMS = ("end_projection", "perstep_mar", "perstep_mar_qargmax", "perstep_mar+seed")
CC_CELLS = [f"T1/named/{a}" for a in ARMS]
T4_CELLS = [f"native/T4/named/{a}" for a in ARMS]
STEP1_ONLY = ["native/T1/named/perstep_mar", "native/T2/named/perstep_mar"]
CELLS = [*CC_CELLS, *STEP1_ONLY, *T4_CELLS]
NOT_RUN_T1 = {
    "native/T1/named/end_projection",
    "native/T1/named/perstep_mar_qargmax",
    "native/T1/named/perstep_mar+seed",
}

T84F_SHAPE = (
    " weather in Lisbon. I should use the `google_search` tool for this.1callcall:google_search{query:\\[weather"
    " lisbon"
)
P1_CC_QUERY = {S1: "I will search for the weather in Lisbon", S2: "british museum opening hours"}
P2_QUERY = {
    ("T1/named/perstep_mar", S2): "british museum | opening hours",
    ("native/T4/named/end_projection", S1): "I will search for the weather in Lisbon tomorrow.",
    ("native/T4/named/end_projection", S2): "british museum​",
    ("native/T4/named/perstep_mar", S1): T84F_SHAPE,
    ("native/T4/named/perstep_mar", S2): "british museum call:google_search",
    ("native/T4/named/perstep_mar_qargmax", S1): "weather lisbon google_search",
}
FLIPPED = {
    (2, S2, "T1/named/perstep_mar", "markup"),
    (2, S1, "native/T4/named/end_projection", "narration"),
    (2, S2, "native/T4/named/end_projection", "control character"),
    (2, S1, "native/T4/named/perstep_mar", "markup"),
    (2, S2, "native/T4/named/perstep_mar", "tool syntax"),
    (2, S1, "native/T4/named/perstep_mar_qargmax", "tool syntax"),
}
FLIPPED_P1 = {(1, S1, "narration")}
REASONS = {"narration": 2, "markup": 2, "tool syntax": 2, "control character": 1}

#: policy -> (old exact k, new exact k, old google_search k, new google_search k); n = 4 (all) and 2 (google_search)
EXPECT = {
    "draft_only": (4, 3, 2, 1),
    "draft_only:native": (4, 4, 2, 2),
    "cd_always:T1/named/end_projection": (4, 4, 2, 2),
    "cd_always:T1/named/perstep_mar": (4, 3, 2, 1),
    "cd_on_invalid:T1/named/perstep_mar": (
        4,
        3,
        2,
        1,
    ),  # every draft is valid: the cc draft decides
    "cd_always:T1/named/perstep_mar+seed": (4, 4, 2, 2),
    "cd_always:native/T1/named/perstep_mar": (4, 4, 2, 2),
    "cd_always:native/T2/named/perstep_mar": (4, 4, 2, 2),
    "cd_always:native/T4/named/end_projection": (4, 2, 2, 0),
    "cd_always:native/T4/named/perstep_mar": (4, 2, 2, 0),
    "cd_always:native/T4/named/perstep_mar_qargmax": (4, 3, 2, 1),
    "cd_always:native/T4/named/perstep_mar+seed": (4, 4, 2, 2),
    "cd_on_invalid:native/T4/named/perstep_mar": (4, 4, 2, 2),
}


# --------------------------------------------------------------------------- the hand-built artifact


def _case(cid: str) -> Mapping[str, Any]:
    return ts.by_id()[cid]


def _ref(cid: str) -> dict[str, Any]:
    return copy.deepcopy(dict(_case(cid)["reference_call"]))


def _search(q: str) -> dict[str, Any]:
    return {"name": "google_search", "arguments": {"query": q}}


def encode(text: str) -> list[int]:
    from local_astra.decoding import vocab

    return [int(t) for t in vocab.encode(text)]


def _p1_cc(i: int, cid: str) -> dict[str, Any]:
    c = _case(cid)
    head = f"<|channel>thought\nThe user asked for {cid}.<channel|>"
    if c["kind"] != "tool":
        spoken = "It is four."
        text = head + spoken
        route = {"kind": "none", "tag": None, "draft_name": None, "spoken_prefix": spoken}
        draft: dict[str, Any] = {
            "draft_call_text": "",
            "draft_close": "",
            "draft_parse": None,
            "draft_call": None,
            "draft_schema_valid": None,
            "draft_exact": False,
        }
    else:
        dc = _search(P1_CC_QUERY[cid]) if cid in P1_CC_QUERY else _ref(cid)
        body = json.dumps(dc)
        spoken = "On it. "
        text = head + spoken + "<cc>" + body
        route = {"kind": "toolcall", "tag": "cc", "draft_name": dc["name"], "spoken_prefix": spoken}
        draft = {
            "draft_call_text": body,
            "draft_close": "",
            "draft_parse": "ok",
            "draft_call": dc,
            "draft_schema_valid": True,
            "draft_exact": old.exact(dc, c["gt"]),
        }
    return {
        "pass": 1,
        "format": "cc",
        "case": cid,
        "kind": c["kind"],
        "tool": c.get("tool"),
        "error": None,
        "wall_ms": 1000.0,
        "prefix_hit_tokens": 0,
        "prefix_query_tokens": 1000,
        "token_ids": [*encode(text), TURN_CLOSE],
        "text": text,
        "content": "served content (rescore reads text)",
        "route": route,
        **draft,
    }


def _p1_native(cid: str) -> dict[str, Any]:
    c = _case(cid)
    if c["kind"] != "tool":
        return na.p1(cid, "native", kind="no_tool", routed=False, wall=800)
    ref = _ref(cid)
    row = na.p1(
        cid, "native", tool=c["tool"], valid=True, exact=True, draft_call=ref, parse="ok", wall=800
    )
    row["route"]["draft_name"] = ref["name"]
    return row


def _p2(cid: str, cell: str) -> dict[str, Any]:
    q = P2_QUERY.get((cell, cid))
    out = _search(q) if q is not None else _ref(cid)
    wall = 600.0 if cell == "native/T2/named/perstep_mar" else 500.0
    row = na.p2(cid, cell, out_call=out, exact=old.exact(out, _case(cid)["gt"]), wall=wall)
    return row


def pass_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, cid in enumerate([*TOOL_CASES, N1]):
        rows += [_p1_cc(i, cid), _p1_native(cid)]
        if cid in TOOL_CASES:
            rows += [_p2(cid, cell) for cell in CELLS]
    return rows


def regime(stage: str = "dev", cells: Sequence[str] = CELLS) -> dict[str, Any]:
    r = na.regime(stage, cells)
    r.update(
        tools_sha256=hashlib.sha256(
            (REPO / "local_astra/toolcall/tools_v1.json").read_bytes()
        ).hexdigest(),
        cases_sha256=hashlib.sha256(ts.CASES_FILE.read_bytes()).hexdigest(),
        stopped=None,
    )
    return r


def write_artifact(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    cells: Sequence[str] = CELLS,
    stage: str = "dev",
    regime_extra: Mapping[str, Any] | None = None,
    stopped: str | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {**regime(stage, cells), **(regime_extra or {})}
    lines = [head, *rows, {"kind": "trailer", "stopped": stopped}]
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(tmp_path: Path, rows: Sequence[Mapping[str, Any]] | None = None) -> tuple[Path, str]:
    ev = tmp_path / "evidence"
    sha = write_artifact(ev / ARTIFACT_REL, pass_rows() if rows is None else rows)
    return ev, sha


# --------------------------------------------------------------------------- the CLI


def _load_cli() -> Any:
    assert TOOL.is_file(), f"T-0086 not implemented: {TOOL.relative_to(REPO)} does not exist"
    name = "_t0086_cd_toolcall_rescore_exact"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        spec.loader.exec_module(m)
    finally:
        sys.modules.pop(name, None)
    assert callable(getattr(m, "main", None)), f"{TOOL.name} has no main(argv) -> int (criterion 6)"
    return m


def _argv(ev: Path, rel: str, sha: str, out: Path) -> list[str]:
    return ["--evidence-root", str(ev), "--artifact", rel, "--sha256", sha, "--out-dir", str(out)]


def _run_ok(ev: Path, sha: str, out: Path, rel: str = ARTIFACT_REL) -> tuple[dict[str, Any], str]:
    rc = _load_cli().main(_argv(ev, rel, sha, out))
    assert rc == 0, f"exit {rc}"
    assert sorted(p.name for p in out.iterdir()) == sorted([JSON_NAME, MD_NAME])
    return json.loads((out / JSON_NAME).read_text()), (out / MD_NAME).read_text()


def _run_refused(
    ev: Path, sha: str, out: Path, capsys: pytest.CaptureFixture[str], rel: str = ARTIFACT_REL
) -> str:
    rc = _load_cli().main(_argv(ev, rel, sha, out))
    cap = capsys.readouterr()
    text = cap.out + cap.err
    assert rc == 2, (rc, text[-2000:])
    assert "REFUSED:" in text, text[-2000:]
    assert not out.exists() or list(out.iterdir()) == [], "a refusal wrote something"
    return text


def _exact(rep: Mapping[str, Any], policy: str, side: str, scope: str) -> tuple[int, int]:
    pol = rep.get("policies")
    assert isinstance(pol, Mapping) and policy in pol, f"policy {policy} missing from the report"
    try:
        e = pol[policy][side][scope]["exact"]
        return int(e["k"]), int(e["n"])
    except (KeyError, TypeError) as exc:  # the K-note shape
        raise AssertionError(
            f"policies[{policy}][{side}][{scope}]['exact'] not found: {exc!r}"
        ) from None


# --------------------------------------------------------------------------- fixture premises


def test_premises_of_the_hand_built_artifact() -> None:
    """The table in the docstring, checked with the frozen oracle and the stored rows only: every stored flag is
    the 4c7dfdc verdict; each override keeps every keyword; ``text`` round-trips through the pinned tokenizer;
    and ``twopass.rescore`` (current router, not under test) accepts the pass-1 rows without a refusal."""
    from local_astra.eval.cd import replay

    rows = pass_rows()
    for r in rows:
        if r["pass"] == 2:
            assert r["exact"] is True, (r["case"], r["cell"])
        elif r["pass"] == 1 and r["kind"] == "tool":
            assert r["draft_exact"] is True, (r["case"], r["format"])
        if "text" in r:
            assert replay.detokenize(r["token_ids"][:-1]) == r["text"], r["case"]
    assert {(c, k) for (c, k) in P2_QUERY} == {(cell, case) for (_p, case, cell, _r) in FLIPPED}
    out = ts.twopass().rescore(copy.deepcopy(rows), ts.toolset(), [dict(c) for c in ts.all_cases()])
    for a, b in zip(rows, out, strict=True):
        if a["pass"] == 1:
            assert a["route"] == b["route"], a["case"]


# --------------------------------------------------------------------------- criteria 6 and 8: the report


def test_report_tables_old_against_new(tmp_path: Path) -> None:
    """Criterion 8: every ``derive`` policy, old against new exact, overall and google_search (``EXPECT``)."""
    ev, sha = _evidence(tmp_path)
    rep, _md = _run_ok(ev, sha, tmp_path / "out")
    assert rep.get("task") == "T-0086"
    got = {}
    for p in EXPECT:
        ok, n = _exact(rep, p, "old", "all")
        nk, n2 = _exact(rep, p, "new", "all")
        og, gn = _exact(rep, p, "old", "google_search")
        ng, gn2 = _exact(rep, p, "new", "google_search")
        assert (n, n2, gn, gn2) == (4, 4, 2, 2), p
        got[p] = (ok, nk, og, ng)
    assert got == EXPECT


def test_flipped_rows_per_cell_and_pass_and_the_reason_histogram(tmp_path: Path) -> None:
    """Criterion 8: the flipped rows (pass, case, cell, reason) and the ``query_problem`` reason histogram over them."""
    ev, sha = _evidence(tmp_path)
    rep, _md = _run_ok(ev, sha, tmp_path / "out")
    flipped = rep.get("flipped")
    assert isinstance(flipped, list), "report has no 'flipped' list"
    p2 = {(f["pass"], f["case"], f["cell"], f["reason"]) for f in flipped if f["pass"] == 2}
    p1 = {(f["pass"], f["case"], f["reason"]) for f in flipped if f["pass"] == 1}
    assert len(flipped) == len(FLIPPED) + len(FLIPPED_P1)
    assert p2 == FLIPPED
    assert p1 == FLIPPED_P1
    per_cell = Counter(f["cell"] for f in flipped if f["pass"] == 2)
    assert per_cell == Counter(cell for (_p, _c, cell, _r) in FLIPPED)
    assert rep.get("reasons") == REASONS


def test_markdown_shows_old_and_new_per_policy_and_the_caveat(tmp_path: Path) -> None:
    """Criterion 8: the markdown table carries old and new exact per policy, and the caveat (counterfactual on data
    already seen)."""
    ev, sha = _evidence(tmp_path)
    rep, md = _run_ok(ev, sha, tmp_path / "out")
    caveat = rep.get("caveat")
    assert isinstance(caveat, str) and "counterfactual" in caveat.lower()
    assert caveat in md
    line = [
        ln
        for ln in md.splitlines()
        if "cd_always:native/T4/named/perstep_mar " in ln + " " and "|" in ln
    ]
    assert any(
        re.search(r"\b4\s*/\s*4\b", ln) and re.search(r"\b2\s*/\s*4\b", ln) for ln in line
    ), line


def test_counterfactual_selection_old_beside_new_with_the_not_computable_path(
    tmp_path: Path,
) -> None:
    """Criterion 8: the artifact's own rule (``select_format``, dev) on old and new flags, labelled; the new θ_n is
    T1 and its family was not run, so the new line is ``not computable (cells not run: ...)`` naming exactly the
    three missing T1 cells. Hand computation in the module docstring."""
    ev, sha = _evidence(tmp_path)
    rep, md = _run_ok(ev, sha, tmp_path / "out")
    assert LABEL in md
    sel = rep.get("selection")
    assert isinstance(sel, Mapping) and "old" in sel and "new" in sel, sel
    old_s, new_s = json.dumps(sel["old"]), json.dumps(sel["new"])
    assert "draft_only:native" in old_s and '"T4"' in old_s, old_s
    assert '"T1"' in new_s, new_s
    assert "draft_only:native" not in new_s, new_s
    for text in (new_s, md):
        m = re.search(r"not computable \(cells not run: ([^)]*)\)", text)
        assert m, text[-1500:]
        named = set(re.findall(r"native/T\d/named/[\w+]+", m.group(1)))
        assert named == NOT_RUN_T1, named


def test_cc_only_artifact_uses_select_and_is_labelled(tmp_path: Path) -> None:
    """Criterion 8: a cc-only artifact (the mixed one's cc rows and cells) gets the ``select`` counterfactual, with
    the label; no native policy appears."""
    rows = [r for r in pass_rows() if r.get("format") == "cc"]
    ev = tmp_path / "evidence"
    sha = write_artifact(ev / ARTIFACT_REL, rows, cells=CC_CELLS)
    rep, md = _run_ok(ev, sha, tmp_path / "out")
    assert LABEL in md
    assert not [p for p in rep["policies"] if "native" in p]
    assert _exact(rep, "cd_always:T1/named/perstep_mar", "new", "all") == (3, 4)


STAGE1_CELLS = ["T1/named/perstep_mar", "T2/named/perstep_mar", "T4/named/perstep_mar"]


def _cc_rows(
    cells: Sequence[str], overrides: Mapping[tuple[str, str], str]
) -> list[dict[str, Any]]:
    """cc pass 1 as in the mixed artifact (``search-01``'s draft narrated); pass 2 on ``cells``, wall 500, reference
    calls except ``overrides[(cell, case)]`` (a query keeping every keyword)."""
    rows: list[dict[str, Any]] = []
    for i, cid in enumerate([*TOOL_CASES, N1]):
        rows.append(_p1_cc(i, cid))
        for cell in cells if cid in TOOL_CASES else []:
            q = overrides.get((cell, cid))
            out = _search(q) if q is not None else _ref(cid)
            rows.append(na.p2(cid, cell, out_call=out, exact=old.exact(out, _case(cid)["gt"])))
    return rows


def test_cc_only_dev_artifact_counterfactual_is_select_stage1(tmp_path: Path) -> None:
    """Ruling S3: a cc-only dev artifact gets ``select("stage1")``. Cells T1/T2/T4 ``named/perstep_mar``, every
    pass 2 valid, CS true, wall 500, 5 steps; T4's two search calls narrated / markup (both keep the keywords).
    By hand: old -- all 4/4, walls and steps tie, so T4 by order; new -- T4 falls to 2/4 < 4/4 - 1/10, T1 and
    T2 tie, so T2 by order (T4, T2, T1)."""
    over = {
        ("T4/named/perstep_mar", S1): "I will search for the weather in Lisbon tomorrow.",
        ("T4/named/perstep_mar", S2): "british museum `opening hours`",
    }
    ev = tmp_path / "evidence"
    sha = write_artifact(ev / ARTIFACT_REL, _cc_rows(STAGE1_CELLS, over), cells=STAGE1_CELLS)
    rep, md = _run_ok(ev, sha, tmp_path / "out")
    assert LABEL in md
    sel = rep["selection"]
    assert "stage1" in json.dumps(sel), sel
    old_s, new_s = json.dumps(sel["old"]), json.dumps(sel["new"])
    assert "T4/named/perstep_mar" in old_s and "T2/named" not in old_s, old_s
    assert "T2/named/perstep_mar" in new_s and "T4/named" not in new_s, new_s


def test_cc_only_test_artifact_counterfactual_is_select_stage2(tmp_path: Path) -> None:
    """Ruling S3: a cc-only test artifact gets ``select("stage2")``. Cells T1 ``named/{end_projection, perstep_mar,
    perstep_mar_qargmax}``; every draft valid (pass-1 wall 1000), every pass 2 valid, CS true, wall 500, 5 steps;
    ``perstep_mar``'s search-02 call is markup. By hand: old -- every policy 4/4; the lowest e2e p50 (1000) is shared
    by ``draft_only`` and the ``cd_on_invalid`` policies, ties through p90 and steps (0), kind order gives
    ``draft_only``. New -- ``search-01``'s draft is narration, so ``draft_only`` and every ``cd_on_invalid`` fall to
    3/4 < 4/4 - 1/20, as does ``cd_always:T1/named/perstep_mar``; ``cd_always`` over ``end_projection`` and
    ``perstep_mar_qargmax`` tie at 1500 / 5 steps, and arm order gives ``cd_always:T1/named/end_projection``.
    (The cases are dev-split ids; the stage is the regime's.)"""
    cells = ["T1/named/end_projection", "T1/named/perstep_mar", "T1/named/perstep_mar_qargmax"]
    over = {("T1/named/perstep_mar", S2): "british museum | opening hours"}
    ev = tmp_path / "evidence"
    sha = write_artifact(ev / ARTIFACT_REL, _cc_rows(cells, over), cells=cells, stage="test")
    rep, md = _run_ok(ev, sha, tmp_path / "out")
    assert LABEL in md
    sel = rep["selection"]
    assert "stage2" in json.dumps(sel), sel
    old_s, new_s = json.dumps(sel["old"]), json.dumps(sel["new"])
    assert '"draft_only"' in old_s and "cd_always" not in old_s, old_s
    assert "cd_always:T1/named/end_projection" in new_s and '"draft_only"' not in new_s, new_s


def test_mixed_test_artifact_uses_the_recorded_theta_for_old_and_new(tmp_path: Path) -> None:
    """Ruling S3: a mixed test artifact runs step 2 with the recorded θ_n = T4 on both lines (T4's family is run).
    By hand (as the dev computation's step 2, tolerance 1/20): the native drafts are 4/4 in both, the native draft
    wall (800) is the lowest e2e p50, so ``draft_only:native`` on both lines."""
    ev = tmp_path / "evidence"
    sha = write_artifact(ev / ARTIFACT_REL, pass_rows(), stage="test")
    rep, md = _run_ok(ev, sha, tmp_path / "out")
    assert LABEL in md
    for side in ("old", "new"):
        s = json.dumps(rep["selection"][side])
        assert '"T4"' in s and "draft_only:native" in s and "not computable" not in s, (side, s)


# --------------------------------------------------------------------------- criterion 7: refusals


def _tamper(rows: list[dict[str, Any]], cid: str, cell: str, **kw: Any) -> list[dict[str, Any]]:
    hit = [r for r in rows if r["pass"] == 2 and r["case"] == cid and r["cell"] == cell]
    assert len(hit) == 1
    hit[0].update(kw)
    return rows


def test_refuses_a_tampered_click_at_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Criterion 7: a non-google_search pass-2 row whose stored ``exact`` is not its recomputed one (a click far
    outside click-01's bbox stored True) -> exit 2, ``REFUSED:`` naming the case, cell and pass; nothing written."""
    cell = "native/T4/named/perstep_mar+seed"
    outside = {"name": "click_at", "arguments": {"x": 10, "y": 10}}
    assert old.exact(outside, _case(C1)["gt"]) is False, (
        "fixture: the click must be outside the bbox"
    )
    rows = _tamper(pass_rows(), C1, cell, call=outside)  # its stored ``exact`` stays True
    ev, sha = _evidence(tmp_path, rows)
    text = _run_refused(ev, sha, tmp_path / "out", capsys)
    assert C1 in text and cell in text and re.search(r"pass\W{0,3}2", text), text[-2000:]


def test_refuses_a_clean_google_row_stored_false(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Criterion 7: a google_search pass-2 row with a clean query and every keyword, stored ``exact`` False, cannot
    be "stored and plausible" -> refusal naming case, cell and pass."""
    cell = "native/T4/named/perstep_mar_qargmax"
    rows = _tamper(pass_rows(), S2, cell, exact=False)
    ev, sha = _evidence(tmp_path, rows)
    text = _run_refused(ev, sha, tmp_path / "out", capsys)
    assert S2 in text and cell in text and re.search(r"pass\W{0,3}2", text), text[-2000:]


def test_refuses_a_sha_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Criterion 6: ``--sha256`` not the artifact's -> exit 2, nothing written."""
    ev, _sha = _evidence(tmp_path)
    wrong = hashlib.sha256(b"not the artifact").hexdigest()
    _run_refused(ev, wrong, tmp_path / "out", capsys)
    assert not (tmp_path / "out").exists()


def test_refuses_an_existing_out_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Criterion 6: the out-dir must be new -> exit 2, nothing written into the existing one."""
    ev, sha = _evidence(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _run_refused(ev, sha, out, capsys)
    assert list(out.iterdir()) == []


def test_refuses_when_rescore_refuses(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Criterion 6: ``twopass.rescore``'s ``ValueError`` (a stored cc ``text`` that is not the decode of its
    ``token_ids``) -> exit 2, ``REFUSED:``, nothing written."""
    rows = pass_rows()
    first = next(r for r in rows if "text" in r and r["kind"] == "tool")
    first["text"] = first["text"] + " "
    ev, sha = _evidence(tmp_path, rows)
    _run_refused(ev, sha, tmp_path / "out", capsys)
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- criteria 6 and 10: inputs and outputs

_WRITE_MODE = re.compile(r"[wax+]")


def test_inputs_are_opened_read_only_no_network_no_selection_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criteria 6 and 10: every ``open`` under the evidence root or the eval set is read-only; no socket connects;
    the out-dir holds exactly the two report files; the evidence root and the artifact are byte-unchanged."""
    ev, sha = _evidence(tmp_path)
    before = {p: p.read_bytes() for p in ev.rglob("*") if p.is_file()}
    opened: list[tuple[str, str]] = []
    real_open = builtins.open
    watched = (str(ev.resolve()), str(ts.SET_DIR.resolve()))

    def recording_open(file: Any, mode: str = "r", *a: Any, **kw: Any) -> Any:
        if isinstance(file, str | os.PathLike):
            p = str(Path(os.fspath(file)).resolve())
            if p.startswith(watched):
                opened.append((p, mode))
        return real_open(file, mode, *a, **kw)

    def no_connect(*a: Any, **kw: Any) -> Any:
        raise AssertionError("criterion 10: no network")

    cli = _load_cli()
    monkeypatch.setattr(socket.socket, "connect", no_connect)
    monkeypatch.setattr(builtins, "open", recording_open)
    monkeypatch.setattr(io, "open", recording_open)
    out = tmp_path / "out"
    rc = cli.main(_argv(ev, ARTIFACT_REL, sha, out))
    monkeypatch.undo()
    assert rc == 0, rc
    assert any(p.endswith(".jsonl") for p, _ in opened), (
        "fixture: the artifact was never seen opened"
    )
    assert [(p, m) for p, m in opened if _WRITE_MODE.search(m)] == []
    assert sorted(p.name for p in out.iterdir()) == sorted([JSON_NAME, MD_NAME])
    assert {p: p.read_bytes() for p in ev.rglob("*") if p.is_file()} == before


# --------------------------------------------------------------------------- the stored artifacts (criterion 9)

#: key -> (path under the main checkout, sha256, {draft policy: re-scored exact k / n under the old rule})
STORED: dict[str, tuple[str, str, dict[str, tuple[int, int]]]] = {
    # T-0080c's flags predate the T-0082a strip; 27/40 and 47/80 are the strip-rescored counts (T-0082a
    # criterion 11 for test; T-0082c's replay of the same pass 1 stores 27/40 and 47/80).
    "t0080c-dev": (
        "logs/t0080-toolcall-dev-0dc9134/20261009T134249.641182Z-toolcall-dev.jsonl",
        "b0fd71cccdc5b8fb3704407f1ff58bf5fc74911ee0d6e6924c9c8e8c740f1d82",
        {"draft_only": (27, 40)},
    ),
    "t0080c-test": (
        "logs/t0080-toolcall-test-0dc9134/20261009T152205.274601Z-toolcall-test.jsonl",
        "1c69a9c2fcb0465db2399d65f470ffd3b702af63097788e5789d4220b77b6ce1",
        {"draft_only": (47, 80)},
    ),
    "t0082c-dev": (
        "logs/t0082-replay-dev-48313a9/20261010T111949.137517Z-toolcall-dev.jsonl",
        "e5148afdb7f0fa1f5834c9966b18b84e4b72a7dbe097ec6a7544c632152f82dd",
        {"draft_only": (27, 40)},
    ),
    "t0082c-test": (
        "logs/t0082-replay-test-48313a9/20261010T113830.525698Z-toolcall-test.jsonl",
        "41628bdb3b0095acfc8901d5de21214dfa6804f3ae6329a87ce557a53d840a22",
        {"draft_only": (47, 80)},
    ),
    "t0084e-dev": (
        "logs/t0084-native-dev-794161d/20261011T022440.808068Z-toolcall-dev.jsonl",
        "06a71fc7413828ba5ff2843accf0811b5ce8303014ebf0177e99904060bd5c7f",
        {"draft_only": (31, 40), "draft_only:native": (39, 40)},
    ),
    "t0084f-test": (
        "logs/t0084-native-test-794161d/20261011T031521.606364Z-toolcall-test.jsonl",
        "14728634eee2a8621e4bf3fbaf3c84c52ccc224e7c6b0383c82a4dcac0595101",
        {"draft_only": (54, 80), "draft_only:native": (74, 80)},
    ),
}


def _stored_cell_counts(path: Path) -> dict[str, int]:
    """Old ``cd_always:<cell>`` exact k from the artifact's own flags: routed tool-case pass-1 rows of the cell's format
    whose pass-2 row in that cell is stored exact."""
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    p1 = [r for r in lines if r.get("pass") == 1]
    p2 = {(r["case"], r["cell"]): r for r in lines if r.get("pass") == 2}
    out: dict[str, int] = {}
    for cell in lines[0]["cells"]:
        fmt = "native" if cell.startswith("native/") else "cc"
        k = 0
        for r in p1:
            route = r.get("route") or {}
            if (
                r.get("format", "cc") != fmt
                or r.get("kind") != "tool"
                or route.get("kind") != "toolcall"
            ):
                continue
            q = p2.get((r["case"], cell))
            k += bool(q and q.get("exact"))
        out[f"cd_always:{cell}"] = k
    return out


@pytest.mark.parametrize("key", sorted(STORED))
def test_stored_artifacts_rescore_without_refusal(key: str, tmp_path: Path) -> None:
    """Criterion 9 (CPU, read-only): each stored artifact re-scores with exit 0; pass-1 draft exact counts are
    unchanged (old == new == the known count); every ``cd_always`` old count is the artifact's own pass-2 flags; new
    <= old for every policy; the artifact is byte-unchanged. n = 1 artifact each."""
    rel, sha, drafts = STORED[key]
    main = na._main_checkout()  # the stored evidence lives in the main checkout's logs/
    path = main / rel
    assert path.is_file(), f"the stored {key} artifact is not at {path}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    rep, _md = _run_ok(main, sha, tmp_path / "out", rel=rel)
    for p, (k, n) in drafts.items():
        assert _exact(rep, p, "old", "all") == (k, n), p
        assert _exact(rep, p, "new", "all") == (k, n), p
    for p, k in _stored_cell_counts(path).items():
        assert _exact(rep, p, "old", "all")[0] == k, p
    for p in rep["policies"]:
        assert _exact(rep, p, "new", "all")[0] <= _exact(rep, p, "old", "all")[0], p
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


def test_the_9aee66d_partial_is_never_an_input() -> None:
    """Criterion 10: no stored input of this module is the 9aee66d partial."""
    assert not [rel for rel, _s, _d in STORED.values() if "9aee66d" in rel]


# --------------------------------------------------------------------------- criterion 10: unchanged files

#: T-0086's merge commit 3b3a0f1; BASE = ``git merge-base 3b3a0f1^1 3b3a0f1^2``, TIP = ``3b3a0f1^2`` (T-0094b
#: criterion 2). Not the declared fork point 4c7dfdc (4c7dfdc..db7b480 is a docs commit on main). Over this closed range
#: the guard stays meaningful after later tasks legitimately edit the paths (T-0089/T-0092 edited the conductor).
T0086_MERGE = "3b3a0f1"
T0086_BASE = "db7b480"
T0086_TIP = "41df3a4"
T86_UNCHANGED = (
    "tests/eval/toolcall/cases.jsonl",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/router.py",
    "local_astra/toolcall/native.py",
    "local_astra/toolcall/tools.py",
    "local_astra/eval/cd/twopass.py",
    "scripts/cd_toolcall_rescore.py",
    "local_astra/decoding",
    "plugin",
    "local_astra/conductor",
    "local_astra/contracts",
    "local_astra/bench/thresholds.py",
    "scripts/locked",
)
#: The one runtime file T-0086 owns among the toolcall package (criterion 4's ``score.query_problem``): the allowance.
T86_OWNED_SCORE = "local_astra/toolcall/score.py"
#: A prior commit that changed a ``docs/cd/toolcall-v1*`` file (T-0082c's results in ``toolcall-v1.1.md``), parent and
#: commit: the power pair of the docs half (T-0094b ruling P2).
V1_DOCS_PRIOR_PARENT = "3c09ccea4"
V1_DOCS_PRIOR = "45322a7"


def _t86_v1_docs(base: str, tip: str) -> list[str]:
    """Criterion 10's docs half: the ``docs/cd/toolcall-v1*`` paths that differ between two commits."""
    return [p for p in gr.changed(base, tip, "docs/cd") if p.startswith("docs/cd/toolcall-v1")]


def test_t0086_changes_nothing_it_does_not_own() -> None:
    """Criterion 10 over T-0086's closed range (T-0094b): between BASE and TIP nothing under the unchanged paths
    differs, no ``docs/cd/toolcall-v1*.md`` changed, and ``cases.jsonl`` is ``460cd1c3...``. The untracked-file half
    of the old guard is retired: it only meant something while the branch was open. ``score.py`` (owned) may change."""
    gr.prove_range(T0086_BASE, T0086_TIP, T0086_MERGE)
    assert gr.changed(T0086_BASE, T0086_TIP, *T86_UNCHANGED) == []
    assert _t86_v1_docs(T0086_BASE, T0086_TIP) == []
    assert not any(
        p == T86_OWNED_SCORE or T86_OWNED_SCORE.startswith(p + "/") for p in T86_UNCHANGED
    ), "the allowance: score.py is T-0086's own file and must not be fenced"
    assert hashlib.sha256(ts.CASES_FILE.read_bytes()).hexdigest().startswith("460cd1c3")


def test_the_t0086_range_guard_has_power() -> None:
    """T-0094b criterion 5 (adversarial): over ``BASE..6a8600b`` the predicate over ``T86_UNCHANGED`` reports the
    conductor's ``executor.py`` (T-0089/T-0092), reached through the tuple; a predicate reporting nothing fails."""
    gr.require_commits(T0086_BASE, gr.LATER)
    conductor = next(p for p in T86_UNCHANGED if p.endswith("/conductor"))
    assert f"{conductor}/executor.py" in gr.changed(T0086_BASE, gr.LATER, *T86_UNCHANGED)


def test_the_t0086_docs_half_has_power() -> None:
    """T-0094b criterion 5 (adversarial), docs half: over the prior commit 45322a7 (its parent to it), the same docs
    predicate reports ``toolcall-v1.1.md``; a predicate reporting nothing fails."""
    gr.require_commits(V1_DOCS_PRIOR_PARENT, V1_DOCS_PRIOR)
    assert gr.is_ancestor(V1_DOCS_PRIOR, T0086_BASE)
    assert _t86_v1_docs(V1_DOCS_PRIOR_PARENT, V1_DOCS_PRIOR) == ["docs/cd/toolcall-v1.1.md"]


def test_the_t0086_score_allowance_is_not_too_strict() -> None:
    """T-0094b criterion 6, too strict: with the allowance emptied (the toolcall package fenced whole) the predicate
    over T-0086's own range reports exactly the owned ``score.py``."""
    gr.prove_range(T0086_BASE, T0086_TIP, T0086_MERGE)
    fenced_whole = (*T86_UNCHANGED, "local_astra/toolcall")
    assert gr.changed(T0086_BASE, T0086_TIP, *fenced_whole) == [T86_OWNED_SCORE]


# =========================================================================== review F3 (rows added after the review)

T1_ARMS3 = ["T1/named/end_projection", "T1/named/perstep_mar", "T1/named/perstep_mar_qargmax"]
T1_OVER = {("T1/named/perstep_mar", S2): "british museum | opening hours"}


def test_cc_only_dev_replay_uses_select_stage2(tmp_path: Path) -> None:
    """Ruling S3 (review F3.1): a cc-only *dev* artifact whose regime has ``pass1_from`` (a T-0082c replay) gets
    ``select("stage2")``, not stage 1. Same rows as the cc-only test-stage row, so the hand computation there holds:
    old ``draft_only``, new ``cd_always:T1/named/end_projection``. Under stage 1 the only candidate would be
    ``T1/named/perstep_mar`` on both lines."""
    ev = tmp_path / "evidence"
    sha = write_artifact(
        ev / ARTIFACT_REL,
        _cc_rows(T1_ARMS3, T1_OVER),
        cells=T1_ARMS3,
        stage="dev",
        regime_extra={"pass1_from": "logs/t0080-toolcall-dev-0dc9134/x-toolcall-dev.jsonl"},
    )
    rep, _md = _run_ok(ev, sha, tmp_path / "out")
    sel = rep["selection"]
    assert "stage2" in json.dumps(sel), sel
    old_s, new_s = json.dumps(sel["old"]), json.dumps(sel["new"])
    assert '"draft_only"' in old_s and "cd_always" not in old_s, old_s
    assert "cd_always:T1/named/end_projection" in new_s, new_s


@pytest.mark.parametrize(
    "override",
    [
        {"cases_sha256": hashlib.sha256(b"another eval set").hexdigest()},
        {"tools_sha256": hashlib.sha256(b"another tools file").hexdigest()},
    ],
    ids=["cases_sha_not_the_artifacts", "tools_sha_not_the_artifacts"],
)
def test_refuses_ground_truth_that_is_not_the_artifacts(
    tmp_path: Path, override: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """Criterion 6 (review F3.2): the artifact's regime names a cases or tools sha256 other than the files read ->
    exit 2, ``REFUSED:``, nothing written."""
    ev = tmp_path / "evidence"
    sha = write_artifact(ev / ARTIFACT_REL, pass_rows(), regime_extra=override)
    _run_refused(ev, sha, tmp_path / "out", capsys)
    assert not (tmp_path / "out").exists()


def test_refuses_cases_not_at_the_manifest_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Criterion 6 (review F3.2): ground truth is read at the MANIFEST-pinned sha. A ``cases.jsonl`` that differs
    from ``MANIFEST.json``'s ``cases_sha256`` -- even when the artifact names those very bytes -> refusal. In-process
    with the CLI's ``SET_DIR`` (module constant, as in ``scripts/cd_toolcall_rescore.py``) on an edited copy."""
    cli = _load_cli()
    set_dir = tmp_path / "toolcall"
    set_dir.mkdir()
    (set_dir / "MANIFEST.json").write_bytes((ts.SET_DIR / "MANIFEST.json").read_bytes())
    # a case outside the fixture, so only the MANIFEST check can refuse (not criterion 7)
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
    out = tmp_path / "out"
    rc = cli.main(_argv(ev, ARTIFACT_REL, sha, out))
    cap = capsys.readouterr()
    assert rc == 2 and "REFUSED:" in cap.out + cap.err
    assert not out.exists()


def test_flipped_per_cell_aggregate_in_json_and_markdown(tmp_path: Path) -> None:
    """Criterion 8 (review F3.3): ``flipped_per_cell`` holds, per pass-2 cell, the hand count from ``FLIPPED`` (T1
    perstep_mar 1, T4 end_projection 2, T4 perstep_mar 2, T4 qargmax 1) and one pass-1 entry (cc) of 1; the markdown
    per-cell table has a ``| <key> | <n> |`` line for each."""
    ev, sha = _evidence(tmp_path)
    rep, md = _run_ok(ev, sha, tmp_path / "out")
    agg = rep.get("flipped_per_cell")
    assert isinstance(agg, Mapping), "report has no 'flipped_per_cell'"
    want = {
        "T1/named/perstep_mar": 1,
        "native/T4/named/end_projection": 2,
        "native/T4/named/perstep_mar": 2,
        "native/T4/named/perstep_mar_qargmax": 1,
    }
    assert {k: v for k, v in agg.items() if k in CELLS} == want
    rest = {k: v for k, v in agg.items() if k not in CELLS}
    assert list(rest.values()) == [1] and "1" in next(iter(rest)) and "cc" in next(iter(rest)), rest
    for k, v in agg.items():
        assert re.search(rf"^\|\s*{re.escape(k)}\s*\|\s*{v}\s*\|\s*$", md, re.M), (k, v)


def test_mixed_test_artifact_carries_the_s3_note(tmp_path: Path) -> None:
    """Ruling S3 (review F3.4): on a mixed test artifact the selection states that θ_n is the recorded T4 and that
    the dev counterfactual θ_n is in the dev artifact's report (JSON and markdown)."""
    ev = tmp_path / "evidence"
    sha = write_artifact(ev / ARTIFACT_REL, pass_rows(), stage="test")
    rep, md = _run_ok(ev, sha, tmp_path / "out")
    note = rep["selection"].get("note")
    assert isinstance(note, str) and "T4" in note and "dev" in note and "report" in note, note
    assert note in md


def test_mixed_dev_artifact_has_no_s3_note(tmp_path: Path) -> None:
    """Ruling S3 (guard width): the dev artifact chooses θ_n itself, so it carries no recorded-θ_n note."""
    ev, sha = _evidence(tmp_path)
    rep, _md = _run_ok(ev, sha, tmp_path / "out")
    assert not rep["selection"].get("note")


def test_refuses_a_stopped_artifact(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Criterion 6 (review F3.5): a trailer with ``stopped`` set -> exit 2, nothing written."""
    ev = tmp_path / "evidence"
    sha = write_artifact(
        ev / ARTIFACT_REL, pass_rows(), stopped="engine: http_error (case search-01)"
    )
    _run_refused(ev, sha, tmp_path / "out", capsys)
    assert not (tmp_path / "out").exists()


def test_refuses_a_rescore_whose_draft_exact_is_not_the_recomputed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Criterion 7 (review F3.5): if ``twopass.rescore`` hands back a pass-1 row (one it re-scored from ``text``)
    whose ``draft_exact`` is not ``score.exact`` of its ``draft_call``, the CLI refuses. ``rescore`` is wrapped to
    flip ``search-01``'s cc ``draft_exact`` (narrated: recomputed False, returned True); the real one runs first."""
    tp = ts.twopass()
    real = tp.rescore

    def flipping(rows: Any, tools: Any, cases: Any) -> list[dict[str, Any]]:
        out = real(rows, tools, cases)
        hit = [r for r in out if r.get("pass") == 1 and r["case"] == S1 and "text" in r]
        assert len(hit) == 1 and hit[0]["draft_exact"] is False
        hit[0]["draft_exact"] = True
        return out

    monkeypatch.setattr(tp, "rescore", flipping)
    ev, sha = _evidence(tmp_path)
    _run_refused(ev, sha, tmp_path / "out", capsys)


def test_a_no_tool_case_with_a_call_is_not_exact_and_does_not_raise(tmp_path: Path) -> None:
    """Criterion 1 / 6 (review F3.5): a pass-2 row for the no-tool case (ground truth ``None``) carrying a call is
    scored not exact, never raises, and is not a flipped row."""
    rows = pass_rows()
    rows.append(
        na.p2(
            N1,
            "native/T4/named/perstep_mar",
            out_call=_search("weather lisbon"),
            exact=False,
        )
    )
    ev, sha = _evidence(tmp_path, rows)
    rep, _md = _run_ok(ev, sha, tmp_path / "out")
    assert not [f for f in rep["flipped"] if f["case"] == N1]
    assert len(rep["flipped"]) == len(FLIPPED) + len(FLIPPED_P1)
