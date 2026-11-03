"""T-0084d criteria 3 and 3a: ``twopass.report`` on mixed (cc + native) artifacts, from artifacts only (CPU, pure).

Criterion 3a (amendment 2026-10-10, from the T-0084c review): on a mixed artifact every existing section is computed
per format -- ``p1_by_case`` (so ``draft_agreement`` pairs a cell with its own format's draft), the seed report,
``n_cases``, route recall, the false-tag rate, ``pass1_wall_ms``, ``stripped_close``, McNemar against the format's own
``draft_only`` (``draft_only:native`` for native cells) and the dev stage-1 line -- and **cc-only artifacts give
byte-identical reports** (the dict, serialised, and ``render_markdown``) to ``main`` at the fork point.

Criterion 3 additions per format: route recall by the format's own tag, false tags, the ``wrong_tag`` and
``open_thought_call`` counts, the draft parse-failure histogram, the spoken-prefix share (non-empty after strip), the
``text_after_close`` count, pass-1 prompt tokens (definitions ruled in D3 of docs/tasks/T-0084d.md); plus the step-1/step-2 tables, a ``format winner: <cc|native>
(policy …)`` line, the cross-format McNemar, and "dev pilot; not a held-out result" on a dev artifact.

Layout pins (test-author proposal, for the orchestrator's ruling; the task names the sections, not the keys):

- A mixed report has ``by_format`` = ``{"cc": {...}, "native": {...}}``; each holds that format's sections under
  today's key names (``n_cases``, ``route_recall``, ``false_tag_rate``, ``pass1_wall_ms``, ``stripped_close``,
  ``draft_agreement``, ``seed``, ``mcnemar_exact``, ``selection``) and the criterion-3 additions (``wrong_tag``,
  ``open_thought_call``, ``draft_parse_failures``, ``spoken_prefix_share``, ``spoken_prefix_median_chars``, ``text_after_close``,
  ``pass1_prompt_tokens``). A top-level key, if kept, must not carry the pooled value. All reads go through ``sec``.
- ``format_selection`` = ``{"native_thought", "choice", "winner", "tables"}`` (the ``select_format`` result) and
  ``format_mcnemar`` (``{"all": {"n", "b", "c", "p", "p_bonferroni"}, ...}``, as ``_mcnemar``).
- A count is an int or a rate dict (``{"k", "n", ...}``); ``_k`` reads either.

Every expected number is hand-computed in ``tests/cd/_native_artifacts.mixed_rows``'s table; the pooled value each
row would get wrong is named beside it.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests import _git_range as gr
from tests.cd import _native_artifacts as na


def twopass() -> ModuleType:
    return importlib.import_module("local_astra.eval.cd.twopass")


def _report(
    tmp_path: Path, stage: str, cells: list[str], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    art = na.write_artifact(
        tmp_path / f"20261011T000000Z-toolcall-{stage}.jsonl", stage, cells, rows
    )
    return dict(twopass().report([art]))


def sec(rep: Mapping[str, Any], fmt: str, key: str) -> Any:
    by = rep.get("by_format")
    assert isinstance(by, Mapping), (
        f"T-0084d 3a not implemented: the mixed report has no per-format sections (by_format); keys {sorted(rep)}"
    )
    assert fmt in by, f"by_format has no {fmt!r}: {sorted(by)}"
    assert key in by[fmt], f"by_format[{fmt!r}] has no {key!r}: {sorted(by[fmt])}"
    return by[fmt][key]


def _fs(rep: Mapping[str, Any]) -> Mapping[str, Any]:
    fs = rep.get("format_selection")
    assert isinstance(fs, Mapping), (
        f"T-0084d criterion 3 not implemented: the report has no format_selection; keys {sorted(rep)}"
    )
    return fs


def _k(x: Any) -> int:
    if isinstance(x, Mapping):
        return int(x["k"])
    if isinstance(x, list) and len(x) == 1:  # stripped_close is per artifact today
        return _k(x[0])
    return int(x)


def _kn(x: Any) -> tuple[int, int]:
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    assert isinstance(x, Mapping), x
    return int(x["k"]), int(x["n"])


@pytest.fixture
def mixed(tmp_path: Path) -> dict[str, Any]:
    return _report(tmp_path, "dev", na.MIXED_CELLS, na.mixed_rows())


@pytest.fixture
def mixed_d(tmp_path: Path) -> dict[str, Any]:
    return _report(tmp_path, "dev", na.MIXED_CELLS, na.mixed_rows(wrong_tag_case=True))


# =========================================================================== 3a: byte identity on cc-only artifacts


def _reports(
    old_rev: str, new_rev: str, paths: list[Path]
) -> tuple[tuple[str, str], tuple[str, str]]:
    """The byte-identity predicate's inputs: ``twopass.report`` (JSON) and ``render_markdown`` of the blob at
    ``new_rev`` and of the blob at ``old_rev`` on ``paths`` (T-0094b: two fixed blobs, never the live file)."""
    new, old = na.main_twopass(new_rev), na.main_twopass(old_rev)
    a, b = new.report(paths), old.report(paths)
    return (
        (json.dumps(a, indent=2, sort_keys=False), new.render_markdown(a)),
        (json.dumps(b, indent=2, sort_keys=False), old.render_markdown(b)),
    )


def _twopass_blob(rev: str) -> str:
    """The bytes of ``twopass.py`` at ``rev`` (the differential premise compares content, not module identity)."""
    proc = gr._git("show", f"{rev}:local_astra/eval/cd/twopass.py")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _both(paths: list[Path]) -> tuple[tuple[str, str], tuple[str, str]]:
    """3a over T-0084d's own blobs: TIP (9abdb3b) against BASE = ``main`` at the fork point (9e84f21), from its merge
    2e89e87 (T-0094b criteria 3 and 4; was the live module against BASE)."""
    gr.prove_range(na.T0084D_BASE, na.T0084D_TIP, na.T0084D_MERGE)
    old_rev, new_rev = na.T0084D_BASE, na.T0084D_TIP
    assert _twopass_blob(old_rev) != _twopass_blob(new_rev), "premise: two blobs"
    return _reports(old_rev, new_rev, paths)


@pytest.mark.parametrize("key", list(na.STORED))
def test_cc_only_stored_artifacts_give_byte_identical_reports(key: str) -> None:
    """3a: on T-0080c's and T-0082c's stored artifacts (cc only, rows without ``format``) the report dict (JSON) and
    its markdown equal ``main``'s byte for byte. Green before the implementation by design: it guards the diff."""
    (js, md), (js0, md0) = _both([na.stored(key)])
    assert js == js0
    assert md == md0


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_cc_only_synthetic_artifacts_give_byte_identical_reports(
    tmp_path: Path, stage: str
) -> None:
    """3a: a hand-built cc-only artifact whose rows carry ``format: "cc"`` (T-0084c onwards), the four T1 cells: no
    per-format section, no format line, byte-identical to ``main``."""
    cells = na.cc_cells("T1")
    art = na.write_artifact(
        tmp_path / f"x-toolcall-{stage}.jsonl",
        stage,
        cells,
        na.full_rows(cells, walls={cells[0]: 10.0}),
    )
    (js, md), (js0, md0) = _both([art])
    assert js == js0
    assert md == md0
    assert "format winner" not in md


# =========================================================================== 3a: each pooled section, per format


def test_n_cases_per_format(mixed: dict[str, Any]) -> None:
    """Five cases per format (A, B, C, N1, N2); pooled it is 10."""
    assert sec(mixed, "cc", "n_cases") == 5
    assert sec(mixed, "native", "n_cases") == 5
    assert mixed.get("n_cases") != 10, "the top-level n_cases is the pooled count"


def test_route_recall_and_false_tags_per_format(mixed: dict[str, Any]) -> None:
    """cc: recall 2/3 (A, B), false tags 0/2. native: recall 3/3, false tags 1/2 (N1). Pooled: 5/6 and 1/4."""
    assert _kn(sec(mixed, "cc", "route_recall")) == (2, 3)
    assert _kn(sec(mixed, "native", "route_recall")) == (3, 3)
    assert _kn(sec(mixed, "cc", "false_tag_rate")) == (0, 2)
    assert _kn(sec(mixed, "native", "false_tag_rate")) == (1, 2)
    for key, pooled in (("route_recall", (5, 6)), ("false_tag_rate", (1, 4))):
        top = mixed.get(key)
        assert not (isinstance(top, Mapping) and "k" in top and (top["k"], top["n"]) == pooled), key


def test_pass1_wall_per_format(mixed: dict[str, Any]) -> None:
    """cc walls 100..500 -> p50 300, n 5; native 1000..1400 -> p50 1200, n 5. Pooled: n 10, p50 750."""
    cc, nat = sec(mixed, "cc", "pass1_wall_ms"), sec(mixed, "native", "pass1_wall_ms")
    assert (cc["n"], cc["p50"]) == (5, 300.0)
    assert (nat["n"], nat["p50"]) == (5, 1200.0)
    top = mixed.get("pass1_wall_ms")
    assert not (isinstance(top, Mapping) and top.get("n") == 10), (
        "the top-level pass-1 wall is pooled"
    )


def test_stripped_close_per_format(mixed: dict[str, Any]) -> None:
    """cc: 1 of 2 routed drafts had a ``</cc>`` stripped (B). native: 0 (native-tag routes record ""). Pooled: 1/6."""
    assert _kn(sec(mixed, "cc", "stripped_close")) == (1, 2)
    assert _k(sec(mixed, "native", "stripped_close")) == 0


def test_draft_agreement_pairs_each_cell_with_its_own_formats_draft(mixed: dict[str, Any]) -> None:
    """``p1_by_case`` per format. cc ``T1/named/perstep_mar``: valid cc draft A = G_A, pass 2 G_A -> equals 1/1; invalid
    draft B repaired 1/1. (Last-write-wins keeps the native drafts: A invalid -> repaired 1/1 and B's valid G_B != Z ->
    equals 0/1.) native ``native/T4/named/perstep_mar``: valid drafts B (G_B -> G_B) and C (W -> W): equals 2/2; invalid
    A and N1, both repaired: 2/2. (First-write-wins keeps the cc drafts: equals 1/1, repaired 3/3.)"""
    ag = sec(mixed, "cc", "draft_agreement")["T1/named/perstep_mar"]
    assert _kn(ag["pass2_equals_valid_draft"]) == (1, 1)
    assert _kn(ag["repair_of_invalid_draft"]) == (1, 1)
    ag = sec(mixed, "native", "draft_agreement")["native/T4/named/perstep_mar"]
    assert _kn(ag["pass2_equals_valid_draft"]) == (2, 2)
    assert _kn(ag["repair_of_invalid_draft"]) == (2, 2)


def test_seed_report_per_format(mixed: dict[str, Any]) -> None:
    """cc seed cell: routed 2 (A, B), seeded 2, no ``missing``; copy on the valid draft (A) 1/1, invalid (B) 0/1; trimmed
    1/2. native seed cell: routed 4 (A, B, C, N1), seeded 4, no ``missing``; copy valid (B yes, C no) 1/2, invalid (A
    yes, N1 no) 1/2. Pooled: the cc cell counts 6 routed with C and N1 ``missing``; the native cell 6 routed."""
    cc = sec(mixed, "cc", "seed")["T1/named/perstep_mar+seed"]
    assert (cc["routed"], cc["seeded"]) == (2, 2)
    assert "missing" not in (cc["no_seed"] or {})
    assert _kn(cc["copy"]["valid_draft"]) == (1, 1)
    assert _kn(cc["copy"]["invalid_draft"]) == (0, 1)
    assert _kn(cc["trimmed"]) == (1, 2)
    nat = sec(mixed, "native", "seed")["native/T4/named/perstep_mar+seed"]
    assert (nat["routed"], nat["seeded"]) == (4, 4)
    assert "missing" not in (nat["no_seed"] or {})
    assert _kn(nat["copy"]["valid_draft"]) == (1, 2)
    assert _kn(nat["copy"]["invalid_draft"]) == (1, 2)


def test_mcnemar_against_the_formats_own_draft_only(mixed: dict[str, Any]) -> None:
    """Native cells are paired with ``draft_only:native``. Tool cases A, B, C: draft_only:native exact F/T/F;
    ``cd_always:native/T4/named/perstep_mar`` exact F (W_A)/T/F -> n 3, b 0, c 0. Against cc ``draft_only`` (T/F/F) it
    would be b 1, c 1. No McNemar key anywhere pairs a native cell with the cc ``draft_only``."""
    m = sec(mixed, "native", "mcnemar_exact")
    key = "cd_always:native/T4/named/perstep_mar vs draft_only:native"
    assert key in m, sorted(m)
    assert (m[key]["all"]["n"], m[key]["all"]["b"], m[key]["all"]["c"]) == (3, 0, 0)
    cc = sec(mixed, "cc", "mcnemar_exact")
    a = cc["cd_always:T1/named/perstep_mar vs draft_only"]["all"]
    assert (a["n"], a["b"], a["c"]) == (3, 0, 0)  # cc: A T/T, B F(Z)/F, C F/F
    bad = re.compile(r"native/.* vs draft_only\Z")
    keys = [*m, *cc, *(mixed.get("mcnemar_exact") or {})]
    assert [k for k in keys if bad.search(k)] == []


def test_dev_stage1_line_per_format(mixed: dict[str, Any]) -> None:
    """cc: the one cc perstep_mar cell -> ``T1/named/perstep_mar``. native: step 1 over the native perstep_mar cells ->
    native T4 (every called row of ``native/T4/named/perstep_mar`` is valid). A pooled line names the cc cell for
    both."""
    assert sec(mixed, "cc", "selection")["choice"] == "T1/named/perstep_mar"
    choice = sec(mixed, "native", "selection")["choice"]
    assert (
        isinstance(choice, str) and "native" in choice and choice.endswith("T4/named/perstep_mar")
    ), choice


# =========================================================================== criterion 3: additions per format


def test_route_recall_is_by_the_formats_own_tag(mixed_d: dict[str, Any]) -> None:
    """With case D (native pass 1 routed with ``<cc>``: ``wrong_tag``): native recall 3/4 -- D is routed but not by the
    native tag. cc recall 2/4. Counted by any tag, native would be 4/4."""
    assert _kn(sec(mixed_d, "native", "route_recall")) == (3, 4)
    assert _kn(sec(mixed_d, "cc", "route_recall")) == (2, 4)


def test_wrong_tag_and_open_thought_call_counts(mixed_d: dict[str, Any]) -> None:
    """native: one ``wrong_tag`` (D), one ``open_thought_call`` (N2)."""
    assert _k(sec(mixed_d, "native", "wrong_tag")) == 1
    assert _k(sec(mixed_d, "native", "open_thought_call")) == 1


def test_draft_parse_failure_histogram_per_format(mixed: dict[str, Any]) -> None:
    """cc: exactly ``{"trailing text": 1}`` (B). native: ``missing close`` 1 (A) and ``bad value`` 1 (N1); no cc reason,
    no ``ok``."""
    assert dict(sec(mixed, "cc", "draft_parse_failures")) == {"trailing text": 1}
    nat = dict(sec(mixed, "native", "draft_parse_failures"))
    assert nat.get("missing close") == 1 and nat.get("bad value") == 1, nat
    assert "trailing text" not in nat and "ok" not in nat, nat


def test_spoken_prefix_share_over_routed_tool_cases_after_strip(mixed: dict[str, Any]) -> None:
    """Ruling D3: share = tool cases routed in that format whose stripped spoken prefix is non-empty / tool cases routed
    in that format. native: A "I'll click.", B "   ", C "Typing." -> 2/3 (B is whitespace; the no-tool N1 "Sure." is
    not a tool case: counting it gives 3/4, unstripped 3/3). cc: A, B with "I will do it. " -> 2/2."""
    assert _kn(sec(mixed, "native", "spoken_prefix_share")) == (2, 3)
    assert _kn(sec(mixed, "cc", "spoken_prefix_share")) == (2, 2)


def test_spoken_prefix_median_chars_over_the_non_empty_prefixes(mixed: dict[str, Any]) -> None:
    """Ruling D3: median chars over the non-empty prefixes of the routed tool cases. native: "I'll click." (11) and
    "Typing." (7) -> 9.0 (B's 3 blanks excluded: with them, 7). Key pinned: ``spoken_prefix_median_chars``."""
    assert sec(mixed, "native", "spoken_prefix_median_chars") == 9.0


def test_wrong_tag_is_its_own_reason_in_the_native_histogram(mixed_d: dict[str, Any]) -> None:
    """Ruling D3: ``wrong_tag`` appears as its own reason in the native parse-failure histogram (case D)."""
    hist = dict(sec(mixed_d, "native", "draft_parse_failures"))
    assert hist.get("wrong_tag") == 1, hist
    assert hist.get("missing close") == 1 and hist.get("bad value") == 1, hist


def test_native_seed_report_counts_the_wrong_tag_case_as_picked(mixed_d: dict[str, Any]) -> None:
    """T-0084d criterion 7 (with D3): the wrong_tag case D now sends pass 2 through the picker, so the native seed cell
    counts it routed and not seeded for reason ``picker`` -- never ``missing``: routed 5, seeded 4."""
    nat = sec(mixed_d, "native", "seed")["native/T4/named/perstep_mar+seed"]
    assert (nat["routed"], nat["seeded"]) == (5, 4)
    assert dict(nat["no_seed"]) == {"picker": 1}


def test_text_after_close_count(mixed: dict[str, Any]) -> None:
    """native: one row (B, " ok") has text after ``<tool_call|>``."""
    assert _k(sec(mixed, "native", "text_after_close")) == 1


def test_pass1_prompt_tokens(mixed: dict[str, Any]) -> None:
    """native ``prompt_len`` 1000, 1010, 1020, 1030, 1040 -> p50 1020, n 5."""
    d = sec(mixed, "native", "pass1_prompt_tokens")
    assert (d["n"], d["p50"]) == (5, 1020.0)


# =========================================================================== criterion 3: the format comparison


def _full_dev(tmp_path: Path, walls: Mapping[str, float]) -> tuple[dict[str, Any], str]:
    rep = _report(tmp_path, "dev", na.DEV_CELLS, na.full_rows(na.DEV_CELLS, walls=walls))
    return rep, twopass().render_markdown(rep)


def test_dev_report_names_the_native_winner_and_is_marked_a_pilot(tmp_path: Path) -> None:
    """``full_rows`` over the 16 dev cells, ``native/T4/named/end_projection`` at pass-2 wall 10 (all else 500): step 1
    ties -> θ_n T4; step 2: e2e 1010 for both kinds of that cell (every draft invalid), kind order ->
    ``cd_on_invalid:native/T4/named/end_projection``, winner native. The markdown carries the winner line and the dev
    marker."""
    rep, md = _full_dev(tmp_path, {"native/T4/named/end_projection": 10.0})
    fs = _fs(rep)
    assert (fs["native_thought"], fs["choice"], fs["winner"]) == (
        "T4",
        "cd_on_invalid:native/T4/named/end_projection",
        "native",
    )
    assert re.search(
        r"format winner: native \(policy cd_on_invalid:native/T4/named/end_projection\b", md
    ), md
    assert "dev pilot; not a held-out result" in md


def test_dev_report_a_full_tie_names_cc(tmp_path: Path) -> None:
    """No override: every policy ties on every key down to the format -> ``cd_on_invalid:T1/named/end_projection``,
    winner cc."""
    rep, md = _full_dev(tmp_path, {})
    assert _fs(rep)["winner"] == "cc"
    assert re.search(r"format winner: cc \(policy cd_on_invalid:T1/named/end_projection\b", md), md


def test_report_tables_and_cross_format_mcnemar(tmp_path: Path) -> None:
    """The step-1 and step-2 tables name every step-1 cell and every family policy; the cross-format McNemar is paired
    by case over the 4 tool cases, with a Bonferroni-adjusted p not below the raw p."""
    rep, _md = _full_dev(tmp_path, {"native/T4/named/end_projection": 10.0})
    tables = repr(_fs(rep)["tables"])
    for t in ("T1", "T2", "T4"):
        assert f"native/{t}/named/perstep_mar" in tables
    for a in na.ARMS:
        assert f"native/T4/named/{a}" in tables and f"T1/named/{a}" in tables
    assert "draft_only:native" in tables
    assert "format_mcnemar" in rep, f"no cross-format McNemar; keys {sorted(rep)}"
    mc = rep["format_mcnemar"]["all"]
    assert mc["n"] == 4
    assert mc["p_bonferroni"] >= mc["p"]


def test_cross_format_mcnemar_pairs_each_formats_best_eligible_policy(tmp_path: Path) -> None:
    """Ruling D3: the cross-format McNemar pairs each format's best eligible policy under the step-2 rule restricted to
    that format. Dev, 16 cells, every draft invalid (``cd_on_invalid`` = ``cd_always``), pass 2 exact only where listed:
    cc ``T1/named/end_projection`` exact on t0, t1, t2 (3/4); native ``T4/named/perstep_mar`` exact on t3 (1/4); every
    other cell 0/4. Restricted to cc the bar is 3/4 - 1/10 -> only end_projection; restricted to native, 1/4 - 1/10 ->
    only T4 perstep_mar (θ_n T4 by the step-1 tie). Paired by case over t0..t3: one side exact on 3 cases the other is
    not, the other on 1 -> {b, c} = {3, 1}, n 4. Pairing the overall winner (cc) with itself gives 0/0; with any other
    native policy 3/0."""
    rows = na.full_rows(
        na.DEV_CELLS,
        exact_default=False,
        exact_cases={
            "T1/named/end_projection": {"t0", "t1", "t2"},
            "native/T4/named/perstep_mar": {"t3"},
        },
    )
    rep = _report(tmp_path, "dev", na.DEV_CELLS, rows)
    assert "format_mcnemar" in rep, f"no cross-format McNemar; keys {sorted(rep)}"
    mc = rep["format_mcnemar"]["all"]
    assert mc["n"] == 4
    assert sorted((mc["b"], mc["c"])) == [1, 3], mc


def test_one_wrong_tag_case_no_longer_makes_native_ineligible(tmp_path: Path) -> None:
    """T-0084d criterion 7 (H-T84-0 resolved D5): dev, 16 cells, case t0's native pass 1 is a ``wrong_tag``; its native
    pass 2 (through the picker) is valid, exact and CS-true like every other. Native T4 end_projection at pass-2 wall 10
    -> native policies stay eligible (t0 scores from pass 2) and ``cd_on_invalid:native/T4/named/end_projection`` wins.
    Under T-0084c's C8 (the draft's outcome kept) every native policy had schema 3/4 and cc would win."""
    rows = na.full_rows(
        na.DEV_CELLS, walls={"native/T4/named/end_projection": 10.0}, wrong_tag=["t0"]
    )
    tp = twopass()
    raw = [r for r in rows if r.get("pass") in (1, 2)]
    (d,) = [
        r
        for r in tp.derive(raw, "cd_on_invalid:native/T4/named/end_projection")
        if r["case"] == "t0"
    ]
    assert (d["schema_valid"], d["exact"], d["pass2_sent"]) == (True, True, True)
    fs = _fs(_report(tmp_path, "dev", na.DEV_CELLS, rows))
    assert (fs["choice"], fs["winner"]) == (
        "cd_on_invalid:native/T4/named/end_projection",
        "native",
    )


def test_test_report_takes_theta_n_from_its_native_cells_and_is_not_a_pilot(tmp_path: Path) -> None:
    """A test-stage artifact: cc ``T1/named/{4}`` + native ``T2/named/{4}``; native T2 end_projection at wall 10 ->
    ``cd_on_invalid:native/T2/named/end_projection``, θ_n T2, winner native; no dev marker."""
    cells = [*na.cc_cells("T1"), *na.native_cells("T2")]
    rep = _report(
        tmp_path, "test", cells, na.full_rows(cells, walls={"native/T2/named/end_projection": 10.0})
    )
    md = twopass().render_markdown(rep)
    fs = _fs(rep)
    assert (fs["native_thought"], fs["choice"], fs["winner"]) == (
        "T2",
        "cd_on_invalid:native/T2/named/end_projection",
        "native",
    )
    assert "format winner: native" in md
    assert "dev pilot" not in md


# =========================================================================== T-0085 criteria 7 and 8: a stopped report


def _stopped_dev(tmp_path: Path, stopped: str | None) -> tuple[dict[str, Any], str]:
    """The ``_full_dev`` artifact that names a native winner when complete, with its trailer's ``stopped`` set."""
    art = na.write_artifact(
        tmp_path / "20261011T000000Z-toolcall-dev.jsonl",
        "dev",
        na.DEV_CELLS,
        na.full_rows(na.DEV_CELLS, walls={"native/T4/named/end_projection": 10.0}),
    )
    lines = [json.loads(x) for x in art.read_text().splitlines() if x.strip()]
    assert lines[-1].get("kind") == "trailer"
    lines[-1]["stopped"] = stopped
    art.write_text("".join(json.dumps(x) + "\n" for x in lines))
    rep = dict(twopass().report([art]))
    return rep, twopass().render_markdown(rep)


def test_t0085_a_stopped_mixed_report_has_no_winner(tmp_path: Path) -> None:
    """Criterion 7: with ``stopped`` non-empty, ``format_selection`` has choice and winner None and a reason starting
    with ``stopped``; the markdown prints ``format winner: none (run stopped; no decision)`` and omits the step
    tables."""
    stop = "engine: http_error (case click-01, cell native/T4/named/end_projection, scope click_at)"
    rep, md = _stopped_dev(tmp_path, stop)
    assert rep["stopped"], "premise: the report sees the stop"
    fs = _fs(rep)
    assert (fs.get("choice"), fs.get("winner")) == (None, None), fs
    assert str(fs.get("reason", "")).startswith("stopped"), fs.get("reason")
    assert "- format winner: none (run stopped; no decision)" in md.splitlines(), md
    assert "Step 1 (" not in md and "| step-1 cell |" not in md
    assert "Step 2 (" not in md and "| policy | format | eligible |" not in md


def test_t0085_the_same_report_unstopped_still_names_the_winner(tmp_path: Path) -> None:
    """Criterion 7, the other direction: the identical artifact without a stop names ``native`` and prints the step
    tables (the stop alone removes them)."""
    rep, md = _stopped_dev(tmp_path, None)
    assert _fs(rep)["winner"] == "native"
    assert "Step 1 (" in md and "Step 2 (" in md


#: T-0085's merge commit 794161d; BASE = ``git merge-base 794161d^1 794161d^2``, TIP = ``794161d^2`` (T-0094b criterion
#: 3). The declared fork point 3133fac carries the same ``twopass.py`` blob as b7f38ef.
T0085_MERGE = "794161d"
T0085_BASE = "b7f38ef"
T0085_TIP = "9063661"
#: The power pair of the report byte-identity predicate (T-0094b ruling P4), shared by the T-0084d and T-0085 rows:
#: T-0082b's 430a6ed (parent b9c5a8b) added ``pass2_wall_ms_by_position`` to every report, stored artifacts included.
REPORT_PRIOR_PARENT = "b9c5a8b"
REPORT_PRIOR = "430a6ed"


@pytest.mark.parametrize("key", list(na.STORED))
def test_t0085_cc_only_stored_reports_are_byte_identical_to_the_fork(key: str) -> None:
    """Criterion 8: on the four stored cc artifacts the report (JSON) and markdown of T-0085's TIP blob equal its BASE
    blob's (T-0094b: T-0085's own two blobs, never BASE against the live module)."""
    gr.prove_range(T0085_BASE, T0085_TIP, T0085_MERGE)
    old_rev, new_rev = T0085_BASE, T0085_TIP
    assert _twopass_blob(old_rev) != _twopass_blob(new_rev), "premise: two blobs"
    (js, md), (js0, md0) = _reports(old_rev, new_rev, [na.stored(key)])
    assert js == js0
    assert md == md0


@pytest.mark.parametrize("key", list(na.STORED))
def test_the_report_byte_identity_predicate_has_power(key: str) -> None:
    """T-0094b criterion 5 (adversarial), for both differential report rows (T-0084d 3a and T-0085 criterion 8): over
    430a6ed's parent and 430a6ed the same predicate reports the named difference on this stored artifact, the new key
    ``pass2_wall_ms_by_position`` and its markdown line; a predicate that is always equal fails."""
    gr.require_commits(REPORT_PRIOR_PARENT, REPORT_PRIOR)
    assert gr.is_ancestor(REPORT_PRIOR, na.T0084D_BASE) and gr.is_ancestor(REPORT_PRIOR, T0085_BASE)
    (js, md), (js0, md0) = _reports(REPORT_PRIOR_PARENT, REPORT_PRIOR, [na.stored(key)])
    assert js != js0 and md != md0
    assert "pass2_wall_ms_by_position" in json.loads(js)
    assert "pass2_wall_ms_by_position" not in json.loads(js0)
    assert "pass-2 wall ms by position" in md and "pass-2 wall ms by position" not in md0


def test_the_t0084d_report_rows_compare_two_different_blobs(tmp_path: Path) -> None:
    """T-0094b too-strict twin of 3a: on the mixed artifact (outside 3a's cc-only allowance) T-0084d's TIP report has
    ``by_format`` and its BASE report has not (T-0084d criterion 3a); a pair reduced to TIP vs TIP fails here."""
    gr.prove_range(na.T0084D_BASE, na.T0084D_TIP, na.T0084D_MERGE)
    art = na.write_artifact(
        tmp_path / "20261011T000000Z-toolcall-dev.jsonl", "dev", na.MIXED_CELLS, na.mixed_rows()
    )
    (js, _md), (js0, _md0) = _reports(na.T0084D_BASE, na.T0084D_TIP, [art])
    assert "by_format" in json.loads(js) and "by_format" not in json.loads(js0)


def test_the_t0085_report_row_compares_two_different_blobs(tmp_path: Path) -> None:
    """T-0094b too-strict twin of criterion 8: on a stopped mixed dev artifact (outside the cc-only allowance) T-0085's
    TIP markdown says ``format winner: none (run stopped; no decision)`` (criterion 7) and its BASE markdown does not;
    a pair reduced to TIP vs TIP fails here."""
    gr.prove_range(T0085_BASE, T0085_TIP, T0085_MERGE)
    art = na.write_artifact(
        tmp_path / "20261011T000000Z-toolcall-dev.jsonl",
        "dev",
        na.DEV_CELLS,
        na.full_rows(na.DEV_CELLS, walls={"native/T4/named/end_projection": 10.0}),
    )
    lines = [json.loads(x) for x in art.read_text().splitlines() if x.strip()]
    assert lines[-1].get("kind") == "trailer"
    lines[-1]["stopped"] = _K_STOP  # defined with the K6/K7 rows below
    art.write_text("".join(json.dumps(x) + "\n" for x in lines))
    (_js, md), (_js0, md0) = _reports(T0085_BASE, T0085_TIP, [art])
    none_line = "format winner: none (run stopped; no decision)"
    assert none_line in md and none_line not in md0


# =========================================================================== T-0085 rulings K6 and K7: nothing "best" from a partial run

_K_STOP = "engine: http_error (case click-01, cell native/T4/named/end_projection, scope click_at)"


def test_t0085_k6_a_stopped_mixed_report_has_no_cross_format_mcnemar(tmp_path: Path) -> None:
    """K6: on a stopped mixed report ``format_mcnemar`` is exactly ``{"reason": "stopped..."}`` and the markdown prints
    no cross-format McNemar line naming policies."""
    rep, md = _stopped_dev(tmp_path, _K_STOP)
    fmc = rep.get("format_mcnemar")
    assert isinstance(fmc, dict) and set(fmc) == {"reason"}, fmc
    assert str(fmc["reason"]).startswith("stopped"), fmc
    assert not [
        ln for ln in md.splitlines() if ln.startswith("- cross-format McNemar (reported only)")
    ]
    assert "_policy" not in json.dumps(fmc)


def test_t0085_k6_the_same_report_unstopped_names_the_mcnemar_policies(tmp_path: Path) -> None:
    """K6, the other direction: unstopped, the cross-format McNemar names both policies and prints its line."""
    rep, md = _stopped_dev(tmp_path, None)
    assert {"cc_policy", "native_policy"} <= set(rep["format_mcnemar"]), rep["format_mcnemar"]
    assert any(ln.startswith("- cross-format McNemar (reported only)") for ln in md.splitlines())


def _stage_lines(md: str) -> list[str]:
    return [ln for ln in md.splitlines() if " stage line: " in ln]


def test_t0085_k7_stopped_stage_lines_name_no_choice(tmp_path: Path) -> None:
    """K7: on the stopped report each format's stage-1 line has ``choice`` None and a reason starting ``stopped``;
    the markdown's stage lines name no policy."""
    rep, md = _stopped_dev(tmp_path, _K_STOP)
    for fmt in ("cc", "native"):
        line = sec(rep, fmt, "selection")
        assert isinstance(line, dict), line
        assert line.get("choice") is None, (fmt, line)
        assert str(line.get("reason", "")).startswith("stopped"), (fmt, line)
    lines = _stage_lines(md)
    assert len(lines) == 2, lines
    for ln in lines:
        assert "named/" not in ln and "draft_only" not in ln, ln


def test_t0085_k7_unstopped_stage_lines_still_name_a_choice(tmp_path: Path) -> None:
    """K7, the other direction: unstopped, each stage-1 line names a choice."""
    rep, md = _stopped_dev(tmp_path, None)
    for fmt in ("cc", "native"):
        assert sec(rep, fmt, "selection").get("choice"), fmt
    assert all("named/" in ln for ln in _stage_lines(md)), _stage_lines(md)
