"""T-0070 criterion 3: the vendored scorers applied to the preserved JAX emissions reproduce the
published figures exactly.

Oracle: diffgemma-fa docs/RESULTS.md at 27c7d0d (rescored 2026-08-08): unconstrained BFCL arg acc 0.639,
schema valid 0.631 (82/130), exact call 0.500; ``exp_e5_grammar130_map`` 0.663 (193/291), 1.000, 0.531;
Sudoku unconstrained 216/250, ``j0map`` 0/250. The counts are the published numerators. Regime: CPU,
preserved texts (tests/fixtures/cd/dgfa_preserved), gorilla 6ea57973c7a6.

Schema validity is scored against the schema the arm's grammar was compiled from (diffgemma-fa
``eval/run.py``): the unconstrained sample predates ``--ci-enums``/``--expand-wildcard`` (both off); the
e5 map arm ran with both on. With both on, the unconstrained texts score 89/130, not the published 82.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.cd import _compile_support as cs
from tests.cd import _eval_support as es

REPO = Path(__file__).resolve().parents[3]
#: sha256 of diffgemma-fa ``eval/metrics.py`` and ``eval/stats.py`` at 27c7d0d (``git show | sha256sum``).
UPSTREAM_SHA = {
    "metrics.py": "7e786563a5d7b3c1a2b5680b817ff4e49ef2b5db7882be43376cccf6b8d59843",
    "stats.py": "f8396bf7c3ec333bbbb2a323523352fab79975b4b522ac3b8c39551520efbaf1",
}


def _bfcl(artifact: str, **opts: bool) -> dict[str, int]:
    recs = {r.id: r for r in es.cuts().load("bfcl", "test")}
    tot = {
        "n": 0,
        "schema_valid": 0,
        "parsed": 0,
        "arg_correct": 0,
        "arg_total": 0,
        "exact_call": 0,
    }
    for row in cs.preserved(artifact)["rows"]:
        s = es.cuts().score("bfcl", recs[row["id"]], row["text"], **opts)
        tot["n"] += 1
        for k in ("schema_valid", "parsed", "exact_call"):
            tot[k] += int(bool(s[k]))
        tot["arg_correct"] += int(s["arg_correct"])
        tot["arg_total"] += int(s["arg_total"])
    return tot


def test_bfcl_unconstrained_reproduces_published_figures() -> None:
    t = _bfcl("eval_bfcl_live_simple_unconstrained_sample", ci_enums=False, expand_wildcard=False)
    assert (
        t["n"],
        t["schema_valid"],
        t["parsed"],
        t["arg_correct"],
        t["arg_total"],
        t["exact_call"],
    ) == (
        130,
        82,
        126,
        186,
        291,
        65,
    )
    assert round(t["arg_correct"] / t["arg_total"], 3) == 0.639
    assert round(t["schema_valid"] / t["n"], 3) == 0.631
    assert round(t["exact_call"] / t["n"], 3) == 0.500


def test_bfcl_map_arm_reproduces_published_figures() -> None:
    t = _bfcl("exp_e5_grammar130_map", ci_enums=True, expand_wildcard=True)
    assert (t["n"], t["schema_valid"], t["arg_correct"], t["arg_total"], t["exact_call"]) == (
        130,
        130,
        193,
        291,
        69,
    )
    assert round(t["arg_correct"] / t["arg_total"], 3) == 0.663
    assert round(t["exact_call"] / t["n"], 3) == 0.531


def test_bfcl_arg_total_is_a_property_of_the_benchmark() -> None:
    """[AUDIT-B]: an empty emission scores every expected key as a miss; arg_total stays 291."""
    recs = es.cuts().load("bfcl", "test")
    s = [es.cuts().score("bfcl", r, "") for r in recs]
    assert sum(int(x["arg_total"]) for x in s) == 291
    assert sum(int(x["arg_correct"]) for x in s) == 0
    assert not any(x["parsed"] or x["exact_call"] or x["schema_valid"] for x in s)


@pytest.mark.parametrize(
    ("artifact", "solved"), [("task_sudoku_unconstrained", 216), ("task_sudoku_j0map", 0)]
)
def test_sudoku_reproduces_published_solved(artifact: str, solved: int) -> None:
    recs = {r.id: r for r in es.cuts().load("sudoku", "test")}
    rows = cs.preserved(artifact)["rows"]
    assert len(rows) == 250
    got = sum(bool(es.cuts().score("sudoku", recs[r["id"]], r["text"])["solved"]) for r in rows)
    assert got == solved


@pytest.mark.parametrize("name", sorted(UPSTREAM_SHA))
def test_vendored_eval_file_is_upstream_byte_for_byte(name: str) -> None:
    """metrics.py and stats.py import only the stdlib (and jsonschema lazily): vendored unmodified."""
    path = REPO / "local_astra" / "eval" / "cd" / "_vendor" / name
    if not path.is_file():
        pytest.fail(f"T-0070 not implemented: {path.relative_to(REPO)} does not exist")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == UPSTREAM_SHA[name]
    prov = REPO / "local_astra" / "eval" / "cd" / "_vendor" / "PROVENANCE.md"
    assert prov.is_file() and UPSTREAM_SHA[name] in prov.read_text()


# ------------------------------------------------------------------ E2: default normalisation is the grammar's


def test_e2_default_scoring_accepts_an_enum_in_another_case() -> None:
    """live_simple_2-2-0 declares ``enum: [plus, comfort, black]``; the preserved unconstrained text
    writes ``"Comfort"``. Default scoring (E2: ci_enums and expand_wildcard on) calls it schema-valid;
    the published normalisation (both off) does not."""
    rec = next(r for r in es.cuts().load("bfcl", "test") if r.id == "live_simple_2-2-0")
    text = next(
        r["text"]
        for r in cs.preserved("eval_bfcl_live_simple_unconstrained_sample")["rows"]
        if r["id"] == rec.id
    )
    assert '"Comfort"' in text
    assert es.cuts().score("bfcl", rec, text)["schema_valid"] is True
    assert (
        es.cuts().score("bfcl", rec, text, ci_enums=False, expand_wildcard=False)["schema_valid"]
        is False
    )
