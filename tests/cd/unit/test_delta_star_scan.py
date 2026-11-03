"""T-0072 criterion 1: the δ* scan is bit-identical to the frozen sequential δ* and agrees with δ.

``budget.delta_star`` keeps T-0065 R1's signature; ``states`` and ``empty`` must be ``torch.equal`` (same dtype,
shape, device) to ``tests/cd/oracle/delta_star_seq.py`` (the pre-change code at f9304b1, frozen) on every
T-0063 cut automaton (BFCL test and dev cuts under JSON and pretty whitespace; Sudoku test and dev), at
buckets 64, 128 and 512 (those that hold the automaton), for L ∈ {0, 1, 2, 3, 63, 64, 65, 255, 256}; on
hand-built NFAs; with a stop id at 0, at L−1, absent and repeated; with a set that dies mid-canvas and with
ids after the first stop that would kill the set. Independently, it agrees with ``enumerate.advance`` (δ) on
the hand-built automata (T-0065 criterion 7).

These rows compare against the frozen copy, so they pass on the pre-change code: they pin the behaviour the
rewrite must keep. The adversarial rows (non-commuting transitions, the odd carry at L = 65/255, ids after
the first stop) are built so a scan that composes in the wrong order, drops a carried node, or consumes past
the stop gives a different set; each row proves its own case discriminates.

Regime: CPU, torch 2.13.0+cpu, seeds fixed per row (named in the ids).
"""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch

from tests.cd.oracle.enumerate import Toy, advance
from tests.cd.unit._core_support import mask, random_toy, tables_for
from tests.cd.unit._perf_support import (
    STOPS,
    identical,
    scan,
    seq,
    start_of,
    tables_from_compiled,
    walk_ids,
)

LS = (0, 1, 2, 3, 63, 64, 65, 255, 256)
BUCKETS = (64, 128, 512)


def _ids(xs: list[int]) -> torch.Tensor:
    return torch.tensor(xs, dtype=torch.int64)


def _expected_delta(toy: Toy, start: set[int], ids: list[int], stops: tuple[int, ...]) -> set[int]:
    """δ over the ids through the first stop (inclusive), from the enumeration oracle."""
    cut = len(ids)
    for i, v in enumerate(ids):
        if v in stops:
            cut = i + 1
            break
    return advance(toy, start, ids[:cut])


def _as_set(t: torch.Tensor) -> set[int]:
    return {i for i, on in enumerate(t.tolist()) if on}


def _check(tables: Any, a: torch.Tensor, ids: list[int], stops: tuple[int, ...], what: str) -> Any:
    want = seq(tables, a, _ids(ids), stops)
    got = scan(tables, a.clone(), _ids(ids), stops)
    bad = identical(got, want)
    assert not bad, f"{what}: δ* scan differs from the frozen sequential δ*: {bad}"
    return got


# --------------------------------------------------------------------------- hand-built NFAs

#: V = 12; listed ids 0..7 form U, so 8..11 are outside U (admitted only by negated classes).
#: Parallel overlapping labels: 0 -{2,3,4}-> 1 and 0 -{3,4,5}-> 2, and 0 -{3,6}-> 1 in parallel with the first.
NFA_OVERLAP = Toy(
    4,
    12,
    (
        (frozenset({2, 3, 4}), False),  # c0
        (frozenset({3, 4, 5}), False),  # c1
        (frozenset({0, 1, 2, 3, 4, 5, 6, 7}), True),  # c2: {8..11} -- only ids outside U
        (frozenset({1}), False),  # c3: the stop id 1
        (frozenset({2}), True),  # c4: everything but 2 (incl. outside U)
        (frozenset({3, 6}), False),  # c5
        (frozenset({5, 7}), False),  # c6
    ),
    (
        (0, 1, 0),
        (0, 2, 1),
        (0, 1, 5),
        (1, 1, 2),
        (1, 3, 3),
        (2, 3, 2),
        (2, 2, 4),
        (3, 0, 0),
        (3, 3, 6),
        (3, 3, 3),
    ),
    frozenset({3}),
)
TOY_STOPS = (1, 7)


def _toys() -> list[tuple[str, Toy]]:
    out = [("nfa_overlap", NFA_OVERLAP)]
    for seed in range(4):
        rng = random.Random(100 + seed)
        out.append((f"random_nfa_{seed}", random_toy(rng, V=12, S=5, nfa=True, neg=True)))
    return out


def _starts(toy: Toy) -> list[set[int]]:
    n = toy.n_states
    return [{0}, {0, 2 % n}, set(range(1, n))]


def _canvases(toy: Toy, L: int, seed: int) -> dict[str, list[int]]:
    """Random ids plus the stop placements of criterion 1 (stop ids from ``TOY_STOPS``)."""
    rng = random.Random(seed)
    base = [rng.choice([v for v in range(toy.vocab_size) if v not in TOY_STOPS]) for _ in range(L)]
    out = {"no_stop": base}
    if L >= 1:
        out["stop_at_0"] = [1, *base[1:]]
        out["stop_at_last"] = [*base[:-1], 7]
    if L >= 3:
        rep = list(base)
        rep[L // 3] = 1
        rep[L // 3 + 1] = 7
        rep[-1] = 1
        out["stop_repeated"] = rep
    return out


@pytest.mark.parametrize("bucket", [None, 64, 128, 512])
@pytest.mark.parametrize("L", LS)
def test_hand_built_nfas_bit_identical_and_agree_with_delta(L: int, bucket: int | None) -> None:
    n_checked = 0
    for name, toy in _toys():
        tables = tables_for(toy, bucket=bucket)
        B = tables.bucket
        for start in _starts(toy):
            for kind, ids in _canvases(toy, L, seed=L * 7 + len(name)).items():
                got = _check(tables, mask(start, B), ids, TOY_STOPS, f"{name} {kind} L={L} B={B}")
                want = _expected_delta(toy, start, ids, TOY_STOPS)
                assert _as_set(got.states) == want, (name, kind, sorted(start), L)
                assert bool(got.empty) == (not want)
                n_checked += 1
    assert n_checked >= 5 * 3


def test_ids_outside_u_are_admitted_by_negated_classes_only() -> None:
    """NFA_OVERLAP: from {1}, id 9 (outside U) follows c2 (negated) to 1; from {2}, via c2 to 3 and c4 to 2."""
    tables = tables_for(NFA_OVERLAP, bucket=8)
    for start, ids, want in (
        ({1}, [9], {1}),
        ({2}, [9], {2, 3}),
        ({0}, [9], {0} & set()),
        ({1}, [2], set()),
    ):
        got = _check(tables, mask(start, 8), ids, TOY_STOPS, f"outside-U {start} {ids}")
        assert _as_set(got.states) == want == _expected_delta(NFA_OVERLAP, start, ids, TOY_STOPS)


def test_a_set_that_dies_mid_canvas_stays_dead() -> None:
    """Alive for 40 ids, killed by id 2 from {1} (no edge of state 1 admits 2), then 60 more ids."""
    tables = tables_for(NFA_OVERLAP, bucket=64)
    ids = [9] * 40 + [2] + [9, 3, 4] * 20
    assert _expected_delta(NFA_OVERLAP, {1}, ids[:40], TOY_STOPS) == {1}
    assert _expected_delta(NFA_OVERLAP, {1}, ids, TOY_STOPS) == set()
    got = _check(tables, mask({1}, 64), ids, TOY_STOPS, "dies mid-canvas")
    assert _as_set(got.states) == set() and bool(got.empty)


# --------------------------------------------------------------------------- adversarial

#: 0 -a-> 1 -b-> 2 (a = 3, b = 4) with every state looping on 5: a before b reaches {2}; b before a is ∅.
CHAIN = Toy(
    3,
    8,
    ((frozenset({3}), False), (frozenset({4}), False), (frozenset({5}), False)),
    ((0, 1, 0), (1, 2, 1), (0, 0, 2), (1, 1, 2), (2, 2, 2)),
    frozenset({2}),
)


#: 0 loops on a (= 2), 0 -b-> 1 (b = 3), 1 loops on c (= 4): the set is {1} iff b sits at the one place
#: every position before it is a and every position after it is c.
MARKER = Toy(
    2,
    8,
    ((frozenset({2}), False), (frozenset({3}), False), (frozenset({4}), False)),
    ((0, 0, 0), (0, 1, 1), (1, 1, 2)),
    frozenset({1}),
)


@pytest.mark.parametrize("bucket", [8, 128, 512])
@pytest.mark.parametrize("L", [2, 3, 64, 65, 255, 256])
def test_non_commuting_transitions_compose_left_to_right(L: int, bucket: int) -> None:
    """Adversarial: the product must be T_{L-1}···T_1·T_0 applied to the start vector, not reversed at any
    level: a at p, b at q > p (5 elsewhere) reaches {2}; swapping any two subtrees that hold them gives ∅."""
    tables = tables_for(CHAIN, bucket=bucket)
    pairs = {
        (0, 1),
        (0, L - 1),
        (L // 2 - 1, L // 2),
        (L - 2, L - 1),
        (1, L // 2 + 1),
        (L // 3, 2 * L // 3),
    }
    n = 0
    for p, q in sorted(pairs):
        if not 0 <= p < q < L:
            continue
        ids = [5] * L
        ids[p], ids[q] = 3, 4
        swapped = [5] * L
        swapped[p], swapped[q] = 4, 3
        assert _expected_delta(CHAIN, {0}, ids, STOPS) == {2}
        assert _expected_delta(CHAIN, {0}, swapped, STOPS) == set(), (
            "fixture must discriminate order"
        )
        got = _check(tables, mask({0}, bucket), ids, STOPS, f"chain L={L} a@{p} b@{q}")
        assert _as_set(got.states) == {2}, (L, p, q)
        n += 1
    assert n >= 1


@pytest.mark.parametrize("bucket", [8, 512])
@pytest.mark.parametrize("L", [63, 64, 65, 255, 256])
def test_every_position_counts_including_the_odd_carry(L: int, bucket: int) -> None:
    """Adversarial: for every position p, ids = a^p · b · c^(L-p-1) reach {1}; a scan that drops or
    duplicates any leaf or carried node (L = 65, 255 have odd carries) gets {0} or ∅ at some p."""
    tables = tables_for(MARKER, bucket=bucket)
    a0 = mask({0}, bucket)
    for p in range(L):
        ids = [2] * p + [3] + [4] * (L - p - 1)
        assert _expected_delta(MARKER, {0}, ids, STOPS) == {1}
        # the case discriminates: without position p the set is {0} (p = L-1) or ∅ (p < L-1)
        assert _expected_delta(MARKER, {0}, ids[:p] + ids[p + 1 :], STOPS) != {1}
        got = _check(tables, a0, ids, STOPS, f"marker L={L} p={p}")
        assert _as_set(got.states) == {1}, (L, p)


@pytest.mark.parametrize("stop_pos", [0, 1, 32, 64, 127, 191])
def test_ids_after_the_first_stop_are_not_consumed(stop_pos: int) -> None:
    """Adversarial: after the first stop come ids that would kill the set (and a second stop); a scan that
    consumes them returns ∅ where δ* must return the non-empty set reached at the stop."""
    L = 255
    # MARKER + stop: 0 loops on 2 and on the stop 1 via an extra class; 5 kills everything
    toy = Toy(
        2, 8, (*MARKER.classes, (frozenset({1}), False)), (*MARKER.edges, (0, 0, 3)), frozenset({1})
    )
    ids = [2] * stop_pos + [1] + [5, 1, 5] * ((L - stop_pos - 1) // 3 + 1)
    ids = ids[:L]
    want = _expected_delta(toy, {0}, ids, (1, 50))
    assert want == {0}, "fixture: the set survives through the first stop"
    assert advance(toy, {0}, ids) == set(), "fixture: consuming past the stop kills the set"
    tables = tables_for(toy, bucket=128)
    got = _check(tables, mask({0}, 128), ids, (1, 50), f"after-stop at {stop_pos}")
    assert _as_set(got.states) == {0} and not bool(got.empty)


# --------------------------------------------------------------------------- T-0063 cut automata


def _cs() -> Any:
    from tests.cd.unit._draw_support import need_t0063

    return need_t0063()


#: 17 cut automata have bucket 1024 > 512: refused on the wire (T-0062 decision 5, T-0064 W3), so they never reach
#: ``delta_star`` and are outside criterion 1. 14 BFCL pretty test-cut, 1 BFCL pretty dev-cut, 2 BFCL JSON test-cut
#: (measured from T-0063's compile cache, 2026-09-30). Excluded from the sweep; ``test_over_ladder_automata`` pins them.
_PRETTY_TEST_OVER = (
    "44-18-0", "45-18-1", "48-21-0", "49-21-1", "51-23-0", "52-23-1", "70-34-0",
    "71-35-0", "106-63-0", "110-67-0", "111-67-1", "112-68-0", "117-73-0", "122-78-0",
)  # fmt: skip
OVER_LADDER = frozenset(
    [("bfcl_pretty", "test", f"live_simple_{x}") for x in _PRETTY_TEST_OVER]
    + [("bfcl_pretty", "dev", "live_simple_165-98-0")]
    + [("bfcl_json", "test", f"live_simple_{x}") for x in ("117-73-0", "122-78-0")]
)
assert len(OVER_LADDER) == 17


def _cut_ids() -> list[tuple[str, str, str]]:
    """(kind, cut, record) for every T-0063 cut automaton, resolved at collection from the fixtures."""
    from tests.cd import _compile_support as cs

    out = [("sudoku", cut, rid) for cut in ("test", "dev") for rid, _, _ in cs.sudoku_cut(cut)]
    for cut, ids in (("test", cs.bfcl_test_cut_ids()), ("dev", cs.bfcl_dev_cut_ids())):
        for ws in ("pretty", "json"):
            out += [(f"bfcl_{ws}", cut, rid) for rid in ids]
    return [x for x in out if x not in OVER_LADDER]


def _compiled(kind: str, cut: str, rid: str) -> Any:
    cs = _cs()
    if kind == "sudoku":
        puzzle = {r: p for r, p, _ in cs.sudoku_cut(cut)}[rid]
        return cs.compiled_sudoku(puzzle)
    fn = cs.bfcl_function(rid)
    if kind == "bfcl_pretty":
        return cs.compiled_bfcl(rid)  # T-0063's default (BFCL_BASELINE_OPTIONS: pretty whitespace)
    return cs.grammars().compile_bfcl(fn, whitespace="json")


@pytest.mark.parametrize(("kind", "cut", "rid"), _cut_ids())
def test_every_cut_automaton_bit_identical(kind: str, cut: str, rid: str) -> None:
    a = _compiled(kind, cut, rid)
    own = int(a.n_states_bucket)
    buckets = [b for b in BUCKETS if b >= own]
    assert buckets, f"{rid}: bucket {own} > 512 (outside criterion 1's ladder)"
    seed = sum(map(ord, rid))
    detailed = seed % 17 == 0  # stop placements on a deterministic ~6 % subset (every one is cheap)
    for b in buckets:
        tables = tables_from_compiled(a, b)
        a0 = start_of(a, b)
        for L in LS:
            ids = walk_ids(a, L, seed + L)
            _check(tables, a0, ids, STOPS, f"{kind} {rid} B={b} L={L}")
            if detailed and L >= 3:
                for stop_ids in (
                    [1, *ids[1:]],
                    [*ids[:-1], 106],
                    [*ids[: L // 2], 50, *ids[L // 2 + 1 : -1], 1],
                ):
                    _check(tables, a0, stop_ids, STOPS, f"{kind} {rid} B={b} L={L} stops")


@pytest.mark.parametrize(("kind", "cut", "rid"), sorted(OVER_LADDER))
def test_over_ladder_automata_are_outside_criterion_1(kind: str, cut: str, rid: str) -> None:
    """The excluded automata really have bucket > 512 (so the exclusion cannot hide an in-ladder one)."""
    assert int(_compiled(kind, cut, rid).n_states_bucket) > 512, (kind, cut, rid)


def test_cut_walks_are_not_all_dead() -> None:
    """Fixture liveness: on the first automaton of each kind the walked canvas keeps the set alive at L = 12 (Sudoku) / 16 (BFCL),
    so the bit-identity rows compare live sets, not ∅ against ∅."""
    for kind, cut, rid in (
        ("sudoku", "test", _cs().sudoku_cut("test")[0][0]),
        ("bfcl_pretty", "test", _cs().bfcl_test_cut_ids()[0]),
    ):
        L = (
            12 if kind == "sudoku" else 16
        )  # walks end in dead ends after ~25-55 ids (measured), by design
        a = _compiled(kind, cut, rid)
        b = int(a.n_states_bucket)
        res = seq(tables_from_compiled(a, b), start_of(a, b), _ids(walk_ids(a, L, 1)), STOPS)
        assert not bool(res.empty), (
            f"{kind} {rid}: the walk dies by L = {L}; the rows would be vacuous"
        )
