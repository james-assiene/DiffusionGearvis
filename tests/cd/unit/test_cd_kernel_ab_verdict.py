"""T-0077c criterion 9 (retargeted from T-0077b's criterion 8 under ruling O3): the verdict of
``scripts/gpu/cd_kernel_ab.py`` (pure functions; CPU, no container).

Path A is the frozen branch base (``tests.cd.oracle.core_frozen_t0077c``: T-0077b's code, torch ``log_matmul`` and the
Triton H tail), path B the live modules with the ``log_matmul`` kernel dispatched. Synthetic per-bucket facts in the
shape ``measure_bucket`` emits; every expected count below is written out by hand from the task, not read from the
script:

- timed rows: T-0077b's 9 components + ``lm_plain`` + ``lm_comp`` (11 ``("components", c)`` rows) + the 5 arms'
  ``whole_call`` = 16;
- the 9 N3 names of T-0077a (``torch.equal``);
- T-0077b's criterion-2 equality set is **retired** (task criterion 9) in favour of the N4 gate (L6): the facts key
  ``"n4"`` carries ``dq``, ``dh`` (gated: ≤ 1e-6, ≤ 1e-5) and ``dlogz`` (reported);
- **new (API ruling, test-author 2026-10-05):** ``"dead_set"`` maps exactly ``log_a``, ``log_b`` to ``True``
  (``torch.equal`` of the dead sets, live vs frozen); ``"dispatch"`` carries ``{"log_matmul": <int>, "log_matmul_plain":
  <int>, "log_matmul_comp": <int>}`` (per-variant keys required: ``measure_bucket`` always emits them; ruling O9:
  plain exactly 0, compensated ≥ 1), the number of
  entries into ``cd/kernels/logmatmul.py`` the live path made on the bucket (V2: ≥ 1 required; missing = a problem);
- the live repeat row's 8 fields.

Planned gated count per launch (task): 36 N3 + 8 N4 + 8 dead-set + 32 repeat + 64 timed + 20 peaks + 4 container lines
= 172.

A mutation of the facts must add a problem that complete facts do not have, naming what was broken. Boundaries are
tested from both sides: a ratio of exactly 1.10 and an equal peak pass; 1.11 and +0.1 MiB fail; dq at 1e-6 passes,
1.1e-6 fails.
"""

from __future__ import annotations

import copy
import importlib
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
BUCKETS = (64, 128, 256, 512)
COMPONENTS = (
    "emission",
    "pass1_tree",
    "marginals_pass1",
    "pass2",
    "marginals_full",
    "map",
    "map_sweep",
    "draw",
    "hq_tail",
    "lm_plain",
    "lm_comp",
)
ARMS = ("end_projection", "perstep_mf", "perstep_mar", "perstep_mar_sc", "perstep_mar_qargmax")
N3 = (
    "log_transfer",
    "maxplus_tree_live_block",
    "map_tokens",
    "map_score",
    "map_empty",
    "draw_tokens",
    "draw_states",
    "draw_empty",
    "generator_state",
)
DEAD_SET = ("log_a", "log_b")
REPEAT = (
    "q",
    "entropy",
    "log_a",
    "log_b",
    "draw_tokens",
    "draw_states",
    "draw_empty",
    "generator_state",
)
MIB = 2**20


def ab() -> ModuleType:
    gpu = str(REPO / "scripts" / "gpu")
    if gpu not in sys.path:
        sys.path.insert(0, gpu)
    return importlib.import_module("cd_kernel_ab")


def _rows() -> list[dict[str, Any]]:
    rows = [
        {
            "arm": "components",
            "component": c,
            "ratio": 0.9,
            "median_frozen_ms": 1.0,
            "median_live_ms": 0.9,
            "peak_frozen": 100 * MIB,
            "peak_live": 90 * MIB,
        }
        for c in COMPONENTS
    ]
    rows += [
        {
            "arm": a,
            "component": "whole_call",
            "ratio": 0.95,
            "median_frozen_ms": 20.0,
            "median_live_ms": 19.0,
            "peak_frozen": 1500 * MIB,
            "peak_live": 1400 * MIB,
        }
        for a in ARMS
    ]
    return rows


def complete() -> list[dict[str, Any]]:
    return [
        {
            "bucket": b,
            "automaton": {"n_states": b // 2, "bucket": b},
            "n3": dict.fromkeys(N3, True),
            "dead_set": dict.fromkeys(DEAD_SET, True),
            "n4": {"dq": 3e-8, "dh": 2e-7, "dlogz": 1e-6},
            "dispatch": {"log_matmul": 12, "log_matmul_plain": 0, "log_matmul_comp": 12},
            "repeat": dict.fromkeys(REPEAT, True),
            "rows": _rows(),
            "regime": {},
        }
        for b in BUCKETS
    ]


def new_problems(facts: list[dict[str, Any]]) -> list[str]:
    base = set(ab().verdict_problems(complete()))
    return [p for p in ab().verdict_problems(facts) if p not in base]


def _row(f: dict[str, Any], arm: str, comp: str) -> dict[str, Any]:
    return next(r for r in f["rows"] if r["arm"] == arm and r["component"] == comp)


# --------------------------------------------------------------------------- complete facts


def test_complete_passing_facts_give_no_problem() -> None:
    assert ab().verdict_problems(complete()) == []


def test_path_a_is_the_t0077c_frozen_copy_and_the_report_is_t0077c() -> None:
    mod = ab()
    assert mod.FROZEN_PKG == "tests.cd.oracle.core_frozen_t0077c"
    assert "T0077C" in mod.MARKER and "T0077B" not in mod.MARKER, mod.MARKER
    head = mod.format_report(complete(), "regime: test").splitlines()[0]
    assert "T-0077c" in head, head


def test_the_retired_equality_set_is_not_required() -> None:
    """Criterion 9: T-0077b's criterion-2 equality set is retired; facts without ``"equal"`` are complete."""
    assert all("equal" not in f for f in complete())
    assert ab().verdict_problems(complete()) == []


# --------------------------------------------------------------------------- V1: completeness of the gate sets


@pytest.mark.parametrize(
    "mutation", ["dead_set_missing", "dead_set_short", "dead_set_extra", "dead_set_not_bool"]
)
def test_dead_set_gate_must_be_complete(mutation: str) -> None:
    facts = complete()
    f = facts[2]  # bucket 256
    if mutation == "dead_set_missing":
        del f["dead_set"]
    elif mutation == "dead_set_short":
        del f["dead_set"]["log_b"]
    elif mutation == "dead_set_extra":
        f["dead_set"]["q"] = True
    else:
        f["dead_set"]["log_a"] = "error: CUDA OOM"
    got = new_problems(facts)
    assert got and all("256" in p for p in got), got


@pytest.mark.parametrize("name", DEAD_SET)
def test_dead_set_mismatch_fails(name: str) -> None:
    """Criterion 9: the dead sets of log_a and log_b ``torch.equal`` live vs frozen."""
    facts = complete()
    facts[1]["dead_set"][name] = False
    got = new_problems(facts)
    assert got and all("128" in p for p in got), got
    assert any(re.search(rf"(?<![a-z_]){name}(?![a-z_])", p) for p in got), got


@pytest.mark.parametrize(
    ("key", "value", "fails"),
    [
        ("dq", 1e-6, False),
        ("dq", 1.1e-6, True),
        ("dh", 1e-5, False),
        ("dh", 1.1e-5, True),
        ("dq", float("nan"), True),
        ("dh", None, True),
        ("dlogz", 5.0, False),
    ],
)
def test_n4_gate(key: str, value: object, fails: bool) -> None:
    """N4 as L6: max|dq| ≤ 1e-6 and max|dH| ≤ 1e-5 gated (both sides of the bar; missing/NaN fail); |d log Z| is
    reported, not gated."""
    facts = complete()
    facts[3]["n4"][key] = value
    got = new_problems(facts)
    if fails:
        assert got and all("512" in p for p in got) and any(key in p for p in got), got
    else:
        assert got == [], got


def test_n4_missing_fails() -> None:
    facts = complete()
    del facts[0]["n4"]
    got = new_problems(facts)
    assert got and all("64" in p for p in got), got


@pytest.mark.parametrize("mutation", ["missing", "zero", "key_missing"])
def test_missing_dispatch_observation_fails(mutation: str) -> None:
    """V2: the ``log_matmul`` kernel observed dispatched on every bucket."""
    facts = complete()
    if mutation == "missing":
        del facts[2]["dispatch"]
    elif mutation == "zero":
        facts[2]["dispatch"]["log_matmul"] = 0
    else:
        facts[2]["dispatch"] = {}
    got = new_problems(facts)
    assert got and all("256" in p for p in got), got
    assert any("log_matmul" in p or "dispatch" in p for p in got), got


@pytest.mark.parametrize("component", ["hq_tail", "lm_plain", "lm_comp"])
def test_missing_component_row_fails(component: str) -> None:
    facts = complete()
    facts[0]["rows"] = [r for r in facts[0]["rows"] if r["component"] != component]
    got = new_problems(facts)
    assert got and all("64" in p for p in got), got


def test_extra_and_short_row_sets_fail() -> None:
    extra = complete()
    extra[3]["rows"].append({**_row(extra[3], "components", "draw"), "component": "class_max"})
    assert new_problems(extra)
    short = complete()
    short[3]["rows"] = [r for r in short[3]["rows"] if r["arm"] != "perstep_mar_qargmax"]
    assert new_problems(short)


def test_short_n3_and_repeat_sets_fail() -> None:
    a = complete()
    del a[0]["n3"]["generator_state"]
    assert new_problems(a)
    b = complete()
    del b[0]["repeat"]["log_b"]
    assert new_problems(b)


def test_l8_first_run_must_also_be_complete() -> None:
    """Ruling M4 (= L8): the superseded first run is checked for completeness too, lm_comp included."""
    ok = complete()
    first = _rows()
    _row({"rows": first}, "components", "lm_comp")["ratio"] = 1.3
    ok[2]["rows_first"] = first
    assert ab().verdict_problems(ok) == [], (
        "a complete over-1.10 first run superseded by a re-run passes"
    )
    bad = copy.deepcopy(ok)
    bad[2]["rows_first"] = [r for r in first if r["component"] != "lm_comp"]
    assert new_problems(bad)


def test_missing_bucket_and_error_bucket_fail() -> None:
    gone = complete()[:3]
    assert any("512" in p for p in new_problems(gone))
    err = complete()
    err[1] = {"bucket": 128, "error": "Traceback ...\nAssertionError: V2", "rows": []}
    assert any("128" in p for p in new_problems(err))


# --------------------------------------------------------------------------- timing and memory gates


@pytest.mark.parametrize("component", ["hq_tail", "lm_plain", "lm_comp"])
@pytest.mark.parametrize(("ratio", "fails"), [(1.10, False), (1.0999, False), (1.11, True)])
def test_component_ratio_gate(component: str, ratio: float, fails: bool) -> None:
    """Every timed row ≤ 1.10× frozen; ``lm_plain``/``lm_comp`` (and ``hq_tail``) gated like the others."""
    facts = complete()
    _row(facts[1], "components", component)["ratio"] = ratio
    got = new_problems(facts)
    if fails:
        assert got and any(component in p and "128" in p for p in got), got
    else:
        assert got == [], got


@pytest.mark.parametrize("component", ["hq_tail", "lm_plain", "lm_comp"])
def test_untimed_component_row_fails(component: str) -> None:
    facts = complete()
    _row(facts[1], "components", component)["ratio"] = None
    assert any(component in p for p in new_problems(facts))


@pytest.mark.parametrize(("delta", "fails"), [(0, False), (MIB // 10, True)])
def test_whole_call_peak_gate(delta: int, fails: bool) -> None:
    """Whole-call peak added ≤ frozen per arm and bucket, no tolerance (ruling L9): equal passes, +0.1 MiB fails."""
    facts = complete()
    r = _row(facts[3], "perstep_mar", "whole_call")
    r["peak_live"] = r["peak_frozen"] + delta
    got = new_problems(facts)
    if fails:
        assert got and any("perstep_mar" in p and "512" in p for p in got), got
    else:
        assert got == [], got


@pytest.mark.parametrize(
    ("plain", "comp", "fails"),
    [(0, 1, False), (0, 40, False), (1, 39, True), (3, 0, True), (0, 0, True)],
)
def test_dispatch_variants_under_o9(plain: int, comp: int, fails: bool) -> None:
    """Ruling O9: every live kernel entry from ``semiring`` is compensated, so the bucket passes only with
    ``log_matmul_plain == 0`` and ``log_matmul_comp >= 1`` (both directions)."""
    facts = complete()
    facts[1]["dispatch"].update(
        log_matmul=plain + comp, log_matmul_plain=plain, log_matmul_comp=comp
    )
    got = new_problems(facts)
    if fails:
        assert got and all("128" in p for p in got), got
        assert any(
            "log_matmul_plain" in p or "log_matmul_comp" in p or "log_matmul" in p for p in got
        ), got
    else:
        assert got == [], got


@pytest.mark.parametrize("key", ["log_matmul_plain", "log_matmul_comp"])
def test_dispatch_variant_key_is_required(key: str) -> None:
    """``measure_bucket`` always emits both per-variant counts; facts without one are incomplete (V1), not a pass."""
    facts = complete()
    del facts[3]["dispatch"][key]
    got = new_problems(facts)
    assert got and all("512" in p for p in got), got
    assert any(key in p for p in got), got


@pytest.mark.parametrize("value", [True, "3", None, -1])
def test_dispatch_variant_counts_must_be_ints(value: object) -> None:
    facts = complete()
    facts[0]["dispatch"]["log_matmul_plain"] = value
    got = new_problems(facts)
    assert got and all("64" in p for p in got), got
