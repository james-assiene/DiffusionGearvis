"""T-0063 criteria 7 (size report and gate) and 9 (determinism, cache key).

API these rows assume (named in the T-0063 test-author report):

- ``report.gate(automaton, max_states=128)`` raises ``report.TooLarge`` when ``automaton.n_states`` exceeds
  ``max_states`` and returns normally otherwise;
- ``report.compile_report(records, *, max_states)`` takes the vendored record types
  (``bfcl_data.BfclRecord`` / ``tasks.sudoku.SudokuRecord``) and returns one row per record, in order.
  A row exposes ``id``, ``n_states``, ``bucket``, ``is_dfa``, ``n_classes``, ``k_max``, ``max_neg_size``
  and ``build_seconds`` (as keys or attributes); a record above ``max_states`` yields a
  ``report.TooLargeRecord`` row carrying ``id`` and ``n_states`` instead of disappearing;
- ``grammars.compile_bfcl(function, **options)`` accepts each key of ``BFCL_BASELINE_OPTIONS`` as an
  override; the cache key is the vendored triple ``(schema_hash, tokenizer_hash, compiler_version)``.

Expected values are hand-derived: a Sudoku 4x4 automaton has 11 header states (open, nine name-count
states, close) + 20 grammar states (one per character position of 16 digits and 3 newlines, plus the
start) + ACC = **32** states, bucket 64 (the bucket holds the real states plus the dead state, SPEC §5.5).
Payload bytes are T-0064's and are not asserted here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from tests.cd._compile_support import (
    DGFA_COMMIT,
    REPO,
    TOKENIZER_SHA256,
    VENDOR_PKG,
    VOCAB_SIZE,
    bfcl_by_id,
    bfcl_dev_cut_ids,
    bfcl_test_cut_ids,
    compiled_bfcl,
    compiled_function,
    expected_bucket,
    field,
    fresh_compile,
    grammars,
    report,
    sudoku_cut,
    vendored,
    warm_bfcl,
    warm_sudoku,
)

SUDOKU_N_STATES = 32
SUDOKU_BUCKET = 64
LIVE_PATH_MAX_STATES = 128  # docs/07 §0
EVAL_MAX_STATES = 512  # T-0062 decision 5

#: A single enum property with 40 distinct 20-odd-character literals. Expected to compile far above 512
#: states (a scratch build of this schema under the baseline options measured 704); the rows below only
#: rely on "> 512", which they assert as a precondition.
_WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
    "zulu",
]
OVERSIZE_FN: dict[str, Any] = {
    "name": "pick_code",
    "description": "Pick a code.",
    "parameters": {
        "type": "dict",
        "required": ["code"],
        "properties": {
            "code": {
                "type": "string",
                "description": "The code.",
                "enum": [
                    f"code_{i:03d}_{_WORDS[i % 26]}_{_WORDS[(7 * i + 3) % 26]}" for i in range(40)
                ],
            }
        },
    },
}


def _chain(n_states: int) -> Any:
    """An automaton with exactly ``n_states`` states, built with the vendored construction: a chain of
    ``n_states - 1`` grammar states over the token for ``1`` (id 236770), then stop augmentation adds ACC."""
    Dfa = vendored("minimize").Dfa
    k = n_states - 1
    dfa = Dfa(
        n_states=k,
        transitions=tuple((i, 236770, i + 1) for i in range(k - 1)),
        start=0,
        finals=frozenset({k - 1}),
    )
    a = vendored("automaton").compile_automaton(
        dfa, name=f"chain{n_states}", end_tokens=(1, 106, 50), vocab_size=VOCAB_SIZE
    )
    assert a.n_states == n_states, f"precondition: built {a.n_states} states, wanted {n_states}"
    return a


# --------------------------------------------------------------------------- the gate, both directions


@pytest.mark.parametrize(
    ("n_states", "max_states"), [(128, 128), (129, 512), (512, 512), (16, 128)]
)
def test_gate_passes_at_or_below_the_limit(n_states: int, max_states: int) -> None:
    report().gate(_chain(n_states), max_states)


@pytest.mark.parametrize(("n_states", "max_states"), [(129, 128), (513, 512)])
def test_gate_raises_too_large_above_the_limit(n_states: int, max_states: int) -> None:
    rep = report()
    with pytest.raises(rep.TooLarge) as info:
        rep.gate(_chain(n_states), max_states)
    msg = str(info.value)
    assert str(n_states) in msg and str(max_states) in msg, (
        f"TooLarge should name |S| and the limit: {msg!r}"
    )


def test_gate_default_is_the_live_path_limit_128() -> None:
    rep = report()
    rep.gate(_chain(LIVE_PATH_MAX_STATES))
    with pytest.raises(rep.TooLarge):
        rep.gate(_chain(LIVE_PATH_MAX_STATES + 1))


def test_too_large_is_a_named_exception_type() -> None:
    rep = report()
    assert isinstance(rep.TooLarge, type) and issubclass(rep.TooLarge, Exception)
    assert rep.TooLarge.__name__ == "TooLarge"


# --------------------------------------------------------------------------- compile_report


def _sudoku_records(cut: str) -> list[Any]:
    SudokuRecord = vendored("tasks.sudoku").SudokuRecord
    return [SudokuRecord(id=rid, puzzle=p, solution=s) for rid, p, s in sudoku_cut(cut)]


def _bfcl_records(ids: list[str]) -> list[Any]:
    BfclRecord = vendored("bfcl_data").BfclRecord
    by_id = bfcl_by_id()
    return [
        BfclRecord(
            id=rid,
            split="BFCL_v4_live_simple.json",
            question=by_id[rid].get("question", []),
            functions=tuple(by_id[rid]["function"]),
        )
        for rid in ids
    ]


def _check_row_shape(row: Any) -> None:
    for name, typ in (("n_classes", int), ("k_max", int), ("max_neg_size", int)):
        v = field(row, name)
        assert isinstance(v, typ) and not isinstance(v, bool) and v >= 0, (name, v)
    assert isinstance(field(row, "is_dfa"), bool)
    secs = field(row, "build_seconds")
    # >= 0: under the test-side compile cache a report row may be served from the memo. The timing
    # itself is pinned uncached by test_build_seconds_measures_a_real_compile.
    assert isinstance(secs, float) and secs >= 0.0, secs
    n = field(row, "n_states")
    assert isinstance(n, int) and field(row, "bucket") == expected_bucket(n)


@pytest.mark.parametrize(("cut", "n"), [("test", 250), ("dev", 100)])
def test_compile_report_sudoku_cut(cut: str, n: int) -> None:
    warm_sudoku(cut)
    records = _sudoku_records(cut)
    assert len(records) == n
    rows = list(report().compile_report(records, max_states=EVAL_MAX_STATES))
    assert [field(r, "id") for r in rows] == [r.id for r in records], "one row per record, in order"
    for row in rows:
        _check_row_shape(row)
        assert field(row, "n_states") == SUDOKU_N_STATES, (field(row, "id"), field(row, "n_states"))
        assert field(row, "bucket") == SUDOKU_BUCKET
        assert field(row, "is_dfa") is True


def _check_bfcl_rows(rows: list[Any], ids: list[str]) -> None:
    rep = report()
    assert [field(r, "id") for r in rows] == ids, "one row per record, in order"
    for row in rows:
        rid = field(row, "id")
        n_expected = compiled_bfcl(rid).n_states
        assert field(row, "n_states") == n_expected, rid
        if n_expected > EVAL_MAX_STATES:
            assert isinstance(row, rep.TooLargeRecord), (
                f"{rid}: |S| {n_expected} > 512 must be typed"
            )
        else:
            assert not isinstance(row, rep.TooLargeRecord), rid
            _check_row_shape(row)


def test_compile_report_bfcl_test_cut() -> None:
    ids = bfcl_test_cut_ids()
    assert len(ids) == 130
    warm_bfcl(ids)
    rows = list(report().compile_report(_bfcl_records(ids), max_states=EVAL_MAX_STATES))
    _check_bfcl_rows(rows, ids)


def test_compile_report_bfcl_dev_cut() -> None:
    ids = bfcl_dev_cut_ids()
    assert len(ids) == 100 and not set(ids) & set(bfcl_test_cut_ids()), (
        "dev cut is 100 ids disjoint from test"
    )
    warm_bfcl(ids)
    rows = list(report().compile_report(_bfcl_records(ids), max_states=EVAL_MAX_STATES))
    _check_bfcl_rows(rows, ids)


def test_a_record_above_512_is_counted_not_dropped() -> None:
    """Criterion 7 negative: an oversize record yields a typed row the harness must count; a record
    within the limit in the same call is not refused (too-wide guard)."""
    assert compiled_function(OVERSIZE_FN).n_states > EVAL_MAX_STATES, (
        "precondition: the schema is oversize"
    )
    BfclRecord = vendored("bfcl_data").BfclRecord
    big = BfclRecord(
        id="synthetic_oversize", split="synthetic", question=[], functions=(OVERSIZE_FN,)
    )
    small = _sudoku_records("test")[:1]
    rep = report()
    rows = list(rep.compile_report([*small, big], max_states=EVAL_MAX_STATES))
    assert [field(r, "id") for r in rows] == [small[0].id, "synthetic_oversize"]
    assert not isinstance(rows[0], rep.TooLargeRecord)
    assert isinstance(rows[1], rep.TooLargeRecord)
    assert field(rows[1], "n_states") == compiled_function(OVERSIZE_FN).n_states
    assert sum(isinstance(r, rep.TooLargeRecord) for r in rows) == 1


# --------------------------------------------------------------------------- criterion 9: determinism


_LOOKUP: dict[str, Any] = {
    "name": "lookup_user",
    "description": "Look up a user record by id.",
    "parameters": {
        "type": "dict",
        "required": ["user_id", "mode"],
        "properties": {
            "user_id": {"type": "integer", "description": "The user's id."},
            "mode": {"type": "string", "enum": ["FAST", "SLOW"], "description": "Mode."},
            "special": {"type": "string", "description": "A note.", "default": "none"},
        },
    },
}


_DUMP = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    from local_astra.decoding import grammars
    fn, puzzle, out = json.loads(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
    arrays = {}
    for tag, a in (("bfcl", grammars.compile_bfcl(fn)), ("sudoku", grammars.compile_sudoku(tuple(map(tuple, puzzle))))):
        t = a.tables
        for name in ("edge_src", "edge_dst", "edge_class", "d", "is_final", "start_vector"):
            arrays[f"{tag}.{name}"] = np.asarray(getattr(a, name))
        for name in ("class_id", "class_size", "sum_is_neg", "sum_indptr", "sum_indices",
                     "max_is_neg", "max_indptr", "max_indices"):
            arrays[f"{tag}.tables.{name}"] = np.asarray(getattr(t, name))
        arrays[f"{tag}.key"] = np.frombuffer(json.dumps(
            [a.schema_hash, a.tokenizer_hash, a.compiler_version, a.n_states, a.n_states_bucket,
             bool(a.is_dfa)]).encode(), dtype=np.uint8)
    np.savez(out, **arrays)
    """
)


def _compile_in_fresh_process(tmp: Path, hashseed: str) -> dict[str, Any]:
    import numpy as np

    out = tmp / f"dump_{hashseed}.npz"
    puzzle = [list(r) for r in sudoku_cut("test")[0][1]]
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _DUMP,
            json.dumps(_LOOKUP),
            json.dumps(puzzle),
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=REPO,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    with np.load(out, allow_pickle=False) as z:
        return {k: (z[k].dtype.str, z[k].shape, z[k].tobytes()) for k in z.files}


def test_compiling_twice_in_fresh_processes_is_byte_identical(tmp_path: Path) -> None:
    """Two interpreters, two different ``PYTHONHASHSEED`` values: set/dict iteration order cannot leak
    into the arrays. (An in-process repeat could be served from a cache and prove nothing.)"""
    first = _compile_in_fresh_process(tmp_path, "0")
    second = _compile_in_fresh_process(tmp_path, "4242")
    assert first.keys() == second.keys()
    differ = sorted(k for k in first if first[k] != second[k])
    assert not differ, f"not byte-identical across processes: {differ}"


def test_cache_key_names_the_tokenizer_and_the_vendored_compiler() -> None:
    a = compiled_function(_LOOKUP)
    assert a.tokenizer_hash == TOKENIZER_SHA256
    assert DGFA_COMMIT in a.compiler_version, (
        f"compiler_version={a.compiler_version!r} does not name {DGFA_COMMIT}"
    )
    assert a.schema_hash, "empty schema_hash"


_FLIPS = {
    "whitespace": "json",
    "fence": False,
    "ci_enums": False,
    "expand_wildcard": False,
    "nonempty_strings": False,
    "channel_header": False,
}


def test_cache_key_changes_with_every_grammar_option() -> None:
    g = grammars()
    assert set(_FLIPS) <= set(g.BFCL_BASELINE_OPTIONS), sorted(g.BFCL_BASELINE_OPTIONS)
    keys = {"baseline": _key(g.compile_bfcl(_LOOKUP))}
    for option, value in _FLIPS.items():
        assert g.BFCL_BASELINE_OPTIONS[option] != value, option
        keys[option] = _key(g.compile_bfcl(_LOOKUP, **{option: value}))
    collisions = [(a, b) for a in keys for b in keys if a < b and keys[a] == keys[b]]
    assert not collisions, f"cache keys collide across grammar options: {collisions}"


def _key(a: Any) -> tuple[str, str, str]:
    return (a.schema_hash, a.tokenizer_hash, a.compiler_version)


def test_vendored_package_path_is_used() -> None:
    """The automata these rows compare come from the vendored compiler, not a second copy."""
    a = compiled_function(_LOOKUP)
    assert type(a).__module__ == f"{VENDOR_PKG}.automaton", type(a).__module__


# --------------------------------------------------------------------------- the test-side cache itself


def test_build_seconds_measures_a_real_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uncached: with the implementation's own compile function in place, ``build_seconds`` is a
    positive wall time (the other report rows may be served from the test-side memo)."""
    from tests.cd import _compile_cache

    g = grammars()
    monkeypatch.setattr(g, "compile_sudoku", _compile_cache.ORIGINALS["compile_sudoku"])
    rows = list(report().compile_report(_sudoku_records("dev")[:1], max_states=EVAL_MAX_STATES))
    secs = field(rows[0], "build_seconds")
    assert isinstance(secs, float) and secs > 0.0, secs


def _arrays(a: Any) -> dict[str, Any]:
    import numpy as np

    out: dict[str, Any] = {}
    for name in ("edge_src", "edge_dst", "edge_class", "d", "is_final", "start_vector"):
        v = np.asarray(getattr(a, name))
        out[name] = (v.dtype.str, v.shape, v.tobytes())
    for name in (
        "class_id",
        "class_size",
        "sum_is_neg",
        "sum_indptr",
        "sum_indices",
        "max_is_neg",
        "max_indptr",
        "max_indices",
    ):
        v = np.asarray(getattr(a.tables, name))
        out[f"tables.{name}"] = (v.dtype.str, v.shape, v.tobytes())
    for name in (
        "schema_hash",
        "tokenizer_hash",
        "compiler_version",
        "n_states",
        "n_states_bucket",
        "is_dfa",
    ):
        out[name] = getattr(a, name)
    return out


def _sample_for_cache() -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = [("sudoku", sudoku_cut("test")[0][1]), ("bfcl", _LOOKUP)]
    out.append(("bfcl", bfcl_by_id()["live_simple_0-0-0"]["function"][0]))
    return out


def test_a_disk_cache_load_equals_a_fresh_compile() -> None:
    """The on-disk cache (vendored save/load) returns exactly what an uncached compile returns, on a
    sample of both grammar kinds. With ``ASTRA_CD_COMPILE_CACHE=0`` the row still compares the memo."""
    from tests.cd import _compile_cache

    g = grammars()
    automaton_mod = vendored("automaton")
    for kind, arg in _sample_for_cache():
        fresh = fresh_compile(kind, arg)
        cached = g.compile_sudoku(arg) if kind == "sudoku" else g.compile_bfcl(arg)
        assert cached is not fresh
        assert _arrays(cached) == _arrays(fresh), kind
        if _compile_cache.disk_enabled():
            key = _compile_cache.payload(kind, arg, {})
            path = _compile_cache.cache_dir() / f"{_compile_cache.disk_key(g, key)}.npz"
            assert path.is_file(), f"{kind}: nothing written to the disk cache at {path}"
            assert _arrays(automaton_mod.load(str(path))) == _arrays(fresh), kind
