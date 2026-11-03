"""T-0067 criterion 5: registry and slot hygiene (adversarial).

Options are parsed with T-0064's parser (its reason codes surface), both ``add_request`` orderings are
tolerated, state is reset in the model state's ``remove_request``, and a slot freed and re-taken (LIFO) never
leaks the previous occupant's automaton, A_k or emitted count — including a slot freed between a commit and
its record flush.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.cd.fakes.diffusion_sampler import Cfg, World, rows_differ
from tests.cd.unit._sampler_support import G1, G2, Grammar, Records, brute_map, cd_xargs
from tests.cd.unit._scenario import canvas_logits

BIG = 10**6


@pytest.fixture(scope="module")
def g1() -> Grammar:
    return Grammar(G1, "t0067-reg-g1")


@pytest.fixture(scope="module")
def g2() -> Grammar:
    return Grammar(G2, "t0067-reg-g2")


def _first_commit(w: World, req: str, fn: Any, limit: int = 8) -> tuple[int, list[int]]:
    for _ in range(limit):
        s = w.decode(fn)
        ids = s.committed(req)
        if ids is not None:
            return s.call, ids
    raise AssertionError(f"{req} did not commit within {limit} calls")


@pytest.mark.parametrize("sampler_first", [False, True])
def test_both_add_request_orderings_constrain(sampler_first: bool, g1: Grammar) -> None:
    fn = canvas_logits({"c": 11, "u": 12}, period=3)
    w = World(Cfg(CL=4, stability_threshold=1), cd=True)
    w.add("c", xargs=cd_xargs(g1, f"order-{sampler_first}"), sampler_first=sampler_first)
    w.add("u")
    w.prefill()
    call, ids = _first_commit(w, "c", fn)
    want = brute_map(g1, fn(call - 1, "c", 4), {0}, 4, BIG)[0]
    assert ids == want, f"sampler_first={sampler_first}: committed {ids}, constrained MAP {want}"


def _stock_equivalent_after_reuse(g1: Grammar, *, cd: bool, live_cd: bool = False) -> list[Any]:
    """``live_cd``: another constrained request ``k`` stays live throughout, so the wrapper cannot take its
    all-unconstrained fast path and a stale per-slot flag or registry entry would show (review M6/M13/M14)."""
    fn = canvas_logits({"a": 21, "b": 22, "u": 23, "k": 24}, period=3)
    w = World(Cfg(CL=4, stability_threshold=1), cd=cd)
    if live_cd:
        w.add("k", xargs=cd_xargs(g1, "reuse-k") if cd else None)
    slot_a = w.add("a", xargs=cd_xargs(g1, "reuse-a") if cd else None)
    w.add("u")
    w.prefill()
    _first_commit(w, "a", fn)
    w.remove("a")
    assert w.add("b") == slot_a, "fixture: LIFO reuse"
    w.prefill()
    return [w.decode(fn) for _ in range(7)]


@pytest.mark.parametrize("live_cd", [False, True], ids=["alone", "with_live_cd"])
def test_slot_retaken_by_unconstrained_request_is_unconstrained(g1: Grammar, live_cd: bool) -> None:
    a = _stock_equivalent_after_reuse(g1, cd=True, live_cd=live_cd)
    b = _stock_equivalent_after_reuse(g1, cd=False, live_cd=live_cd)
    for i, (sa, sb) in enumerate(zip(a, b, strict=True)):
        for r in ("b", "u"):
            bad = rows_differ(sa, sb, r)
            assert bad == [], (
                f"call {i}: {r} (after reuse of a constrained slot) differs from stock in {bad}"
            )


@pytest.mark.parametrize("live_cd", [False, True], ids=["alone", "with_live_cd"])
def test_slot_retaken_by_constrained_request_starts_clean(
    g1: Grammar, g2: Grammar, live_cd: bool
) -> None:
    """A (G1, runs one canvas, emitted = CL) frees its slot; C (G2, max_tokens = CL + 1) takes it.
    C's first canvas must be G2's MAP from G2's start with the full budget — not G1, not A_1, not emitted = CL."""
    fn = canvas_logits({"a": 31, "c": 32, "u": 33, "k": 34}, period=3)
    w = World(Cfg(CL=4, stability_threshold=1), cd=True)
    if live_cd:
        w.add("k", xargs=cd_xargs(g1, "clean-k"))
    slot = w.add("a", xargs=cd_xargs(g1, "clean-a"))
    w.add("u")
    w.prefill()
    _first_commit(w, "a", fn)
    w.remove("a")
    assert w.add("c", xargs=cd_xargs(g2, "clean-c"), max_tokens=5) == slot
    w.prefill()
    call, ids = _first_commit(w, "c", fn)
    want = brute_map(g2, fn(call - 1, "c", 4), {0}, 4, 1)[0]
    assert ids == want, f"re-taken slot committed {ids}; G2's MAP from its start is {want}"


def test_slot_freed_between_commit_and_record_flush(
    g1: Grammar, g2: Grammar, caplog: Any, capsys: Any
) -> None:
    fn = canvas_logits({"a": 41, "d": 42, "u": 43}, period=3)
    records = Records(caplog, capsys)
    w = World(Cfg(CL=4, stability_threshold=1), cd=True)
    slot = w.add("a", xargs=cd_xargs(g1, "flush-a"))
    w.add("u")
    w.prefill()
    _call, a_ids = _first_commit(w, "a", fn)
    w.remove("a")  # before any later sampler call could flush A's record
    assert w.add("d", xargs=cd_xargs(g2, "flush-d")) == slot
    w.prefill()
    _dcall, d_ids = _first_commit(w, "d", fn)
    for _ in range(3):
        w.decode(fn)
    ra, rd = records.tagged("flush-a"), records.tagged("flush-d")
    assert [r["committed_ids"] for r in ra] == [a_ids], f"A's record lost or duplicated: {ra}"
    assert ra[0]["canvas_idx"] == 0 and ra[0]["arm"] == "end_projection"
    assert [r["committed_ids"] for r in rd] == [d_ids], f"D's records: {rd}"
    assert rd[0]["canvas_idx"] == 0, "the new occupant starts at canvas 0"
