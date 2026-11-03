"""T-0072 criterion 4: precomputed class quantities are passed through, not recomputed.

New keyword-only parameters, each defaulting to today's behaviour (API in ``_perf_support``):
``emission.log_transfer(..., *, class_mass)``, ``marginals.constrained_marginals(..., *, class_mass)``,
``draw.joint_draw(..., *, generator, class_log_weights)``, ``mapdecode.constrained_map(..., *, class_max)``.

Per function: (i) with the precomputed value the outputs are bit-identical to the call without it, and the
generator state afterwards is identical; (ii) **adversarial**: a perturbed value changes the output (the value
is used, not silently recomputed); (iii) the function issues no ``[L, V]`` op for that quantity: the ``[L, V]``
op count drops by at least the ``[L, V]`` ops of computing the quantity alone (``log_transfer``: to zero, since
it reads the logits only through the class weights). Plus: ``cd/sampler.py`` computes ``class_max`` once per
constrained slot per call and reuses it across its ``L_eff`` candidates; no positional parameter or return
type of T-0065 R1, T-0066 R1 or T-0067 S1 changes (the existing rows of those tasks cover the latter and are
not edited; ``test_positional_signatures_unchanged`` pins the positional names).

Regime: CPU, float32 and float64 logits, toys with V = 512 (so ``[L, V]`` is distinguishable from ``[L, C]``).
"""

from __future__ import annotations

import dataclasses
import inspect
import random
from typing import Any

import pytest
import torch

from tests.cd.oracle.enumerate import Toy
from tests.cd.unit._core_support import core, mask, random_toy, tables_for
from tests.cd.unit._perf_support import LVOps, new_kw

V = 512
L = 12


def _toy() -> Toy:
    return random_toy(random.Random(21), V=V, S=5, nfa=True, neg=True)


#: parallel edges 0 -> 1 on {2} and on {3}, then 1 -> 1 on anything (negated, stored ∅... use {4}).
PAR = Toy(
    2,
    V,
    ((frozenset({2}), False), (frozenset({3}), False), (frozenset({4}), True)),
    ((0, 1, 0), (0, 1, 1), (1, 1, 2)),
    frozenset({1}),
)


def _logits(dtype: torch.dtype, seed: int = 3, n: int = L) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, V, generator=g, dtype=torch.float64) * -20.0).to(dtype)


def _setup(toy: Toy, dtype: torch.dtype, *, bucket: int = 8, l_eff: int = L - 2) -> dict[str, Any]:
    tables = tables_for(toy, bucket=bucket)
    x = _logits(dtype)
    logm = core("emission").log_transfer(x, tables, l_eff)
    tree = core("tree").up_sweep(logm)
    return {
        "x": x,
        "tables": tables,
        "tree": tree,
        "a": mask({0}, bucket),
        "b": mask(sorted(toy.finals), bucket),
        "l_eff": l_eff,
    }


def _eq(a: Any, b: Any) -> list[str]:
    """Bitwise equality of every tensor field (dataclass or tensor)."""
    if isinstance(a, torch.Tensor):
        ok = (
            a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
            if not a.is_floating_point()
            else (
                a.dtype == b.dtype
                and a.shape == b.shape
                and torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))
            )
        )
        return [] if ok else ["tensor differs"]
    bad = []
    for f in dataclasses.fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, torch.Tensor) and _eq(va, vb):
            bad.append(f.name)
    return bad


def _lv_ops(fn: Any) -> tuple[int, Any]:
    with LVOps(L, V) as m:
        out = fn()
    return len(m.hits), out


DTYPES = [torch.float32, torch.float64]


# --------------------------------------------------------------------------- log_transfer(class_mass)


@pytest.mark.parametrize("dtype", DTYPES)
def test_log_transfer_class_mass_bit_identical(dtype: torch.dtype) -> None:
    s = _setup(_toy(), dtype)
    em = core("emission")
    want = em.log_transfer(s["x"], s["tables"], s["l_eff"])
    cm = em.class_mass(s["x"], s["tables"])
    got = new_kw(em.log_transfer, s["x"], s["tables"], s["l_eff"], class_mass=cm)
    assert not _eq(got, want)


def test_log_transfer_uses_the_supplied_class_mass() -> None:
    s = _setup(_toy(), torch.float64)
    em = core("emission")
    cm = em.class_mass(s["x"], s["tables"])
    shift = torch.arange(cm.class_anchor.shape[1], dtype=torch.float64) * 0.5
    bent = dataclasses.replace(cm, class_anchor=cm.class_anchor + shift)
    got = new_kw(em.log_transfer, s["x"], s["tables"], s["l_eff"], class_mass=bent)
    want = em.log_transfer(s["x"], s["tables"], s["l_eff"])
    assert _eq(got, want), (
        "a perturbed class_mass did not change log_transfer: it is recomputed, not used"
    )


def test_log_transfer_with_class_mass_issues_no_lv_op() -> None:
    s = _setup(_toy(), torch.float32)
    em = core("emission")
    cm = em.class_mass(s["x"], s["tables"])
    n_without, _ = _lv_ops(lambda: em.log_transfer(s["x"], s["tables"], s["l_eff"]))
    assert n_without > 0, "liveness: the [L, V] spy sees the default path"
    n_with, _ = _lv_ops(
        lambda: new_kw(em.log_transfer, s["x"], s["tables"], s["l_eff"], class_mass=cm)
    )
    assert n_with == 0, f"log_transfer issued {n_with} [L, V] ops with class_mass supplied"


# --------------------------------------------------------------------------- constrained_marginals(class_mass)


def _marg(s: dict[str, Any], **kw: Any) -> Any:
    mg = core("marginals")
    args = (s["x"], s["tables"], s["tree"], s["a"], s["b"], s["l_eff"])
    return new_kw(mg.constrained_marginals, *args, **kw) if kw else mg.constrained_marginals(*args)


@pytest.mark.parametrize("dtype", DTYPES)
def test_constrained_marginals_class_mass_bit_identical(dtype: torch.dtype) -> None:
    s = _setup(_toy(), dtype)
    cm = core("emission").class_mass(s["x"], s["tables"])
    assert not _eq(_marg(s, class_mass=cm), _marg(s))


def test_constrained_marginals_uses_the_supplied_class_mass() -> None:
    s = _setup(_toy(), torch.float64)
    cm = core("emission").class_mass(s["x"], s["tables"])
    shift = torch.arange(cm.class_anchor.shape[1], dtype=torch.float64) * 0.5
    bent = dataclasses.replace(cm, class_anchor=cm.class_anchor + shift)
    assert "q" in _eq(_marg(s, class_mass=bent), _marg(s)), "perturbed class_mass left q unchanged"


def test_constrained_marginals_with_class_mass_drops_its_lv_ops() -> None:
    s = _setup(_toy(), torch.float32)
    em = core("emission")
    n_cm, cm = _lv_ops(lambda: em.class_mass(s["x"], s["tables"]))
    n_without, _ = _lv_ops(lambda: _marg(s))
    n_with, _ = _lv_ops(lambda: _marg(s, class_mass=cm))
    assert n_cm > 0
    assert n_without - n_with >= n_cm, (
        f"[L, V] ops: {n_without} without, {n_with} with class_mass; computing it alone takes {n_cm}"
    )


# --------------------------------------------------------------------------- joint_draw(class_log_weights)


def _draw(s: dict[str, Any], seed: int, **kw: Any) -> tuple[Any, torch.Tensor]:
    dr = core("draw")
    g = torch.Generator().manual_seed(seed)
    args = (s["x"], s["tables"], s["tree"], s["a"], s["b"], s["l_eff"])
    out = (
        new_kw(dr.joint_draw, *args, generator=g, **kw) if kw else dr.joint_draw(*args, generator=g)
    )
    return out, g.get_state()


@pytest.mark.parametrize("dtype", DTYPES)
def test_joint_draw_class_log_weights_bit_identical_and_same_generator_state(
    dtype: torch.dtype,
) -> None:
    s = _setup(_toy(), dtype)
    cw = core("emission").class_log_weights(s["x"], s["tables"])
    for seed in range(5):
        want, g_want = _draw(s, seed)
        got, g_got = _draw(s, seed, class_log_weights=cw)
        assert not _eq(got, want), seed
        assert torch.equal(g_got, g_want), f"seed {seed}: generator state differs afterwards"


def test_joint_draw_uses_the_supplied_class_log_weights() -> None:
    """PAR, L = 1: logits put token 2 at 0 and token 3 at −20, so the {2} edge (and token 2) is drawn;
    weights that make class {2} impossible must yield token 3."""
    x = torch.full((1, V), -30.0, dtype=torch.float64)
    x[0, 2], x[0, 3] = 0.0, -20.0
    tables = tables_for(PAR, bucket=4)
    tree = core("tree").up_sweep(core("emission").log_transfer(x, tables, 1))
    s = {"x": x, "tables": tables, "tree": tree, "a": mask({0}, 4), "b": mask({1}, 4), "l_eff": 1}
    cw = core("emission").class_log_weights(x, tables)
    bent = cw.clone()
    bent[:, 0] = -1e30
    base, _ = _draw(s, 0)
    assert int(base.tokens[0]) == 2, "fixture: the unperturbed draw takes token 2"
    got, _ = _draw(s, 0, class_log_weights=bent)
    assert int(got.tokens[0]) == 3, "perturbed class_log_weights were not used for the edge draw"


def test_joint_draw_with_class_log_weights_drops_its_lv_ops() -> None:
    s = _setup(_toy(), torch.float32)
    n_cw, cw = _lv_ops(lambda: core("emission").class_log_weights(s["x"], s["tables"]))
    n_without, _ = _lv_ops(lambda: _draw(s, 1))
    n_with, _ = _lv_ops(lambda: _draw(s, 1, class_log_weights=cw))
    assert n_cw > 0
    assert n_without - n_with >= n_cw, (n_without, n_with, n_cw)


# --------------------------------------------------------------------------- constrained_map(class_max)


def _captured_class_max(s: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> tuple[Any, int]:
    """The ``(max, argmax)`` ``mapdecode.class_max`` returns inside a call without the keyword, and the
    ``[L, V]`` ops it issued: the object the function would have computed (note 3), not a re-derivation."""
    md = core("mapdecode")
    real = md.class_max
    seen: list[Any] = []

    def spy(*a: Any, **k: Any) -> Any:
        with LVOps(L, V) as m:
            out = real(*a, **k)
        seen.append((out, len(m.hits)))
        return out

    monkeypatch.setattr(md, "class_max", spy)
    md.constrained_map(s["x"], s["tables"], s["a"], s["b"], s["l_eff"])
    monkeypatch.setattr(md, "class_max", real)
    assert len(seen) == 1, f"mapdecode.class_max called {len(seen)} times by constrained_map"
    return seen[0]


def _map(s: dict[str, Any], **kw: Any) -> Any:
    md = core("mapdecode")
    args = (s["x"], s["tables"], s["a"], s["b"], s["l_eff"])
    return new_kw(md.constrained_map, *args, **kw) if kw else md.constrained_map(*args)


@pytest.mark.parametrize("dtype", DTYPES)
def test_constrained_map_class_max_bit_identical(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _setup(_toy(), dtype)
    cmx, _ = _captured_class_max(s, monkeypatch)
    assert not _eq(_map(s, class_max=cmx), _map(s))


def test_constrained_map_uses_the_supplied_class_max(monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.full((1, V), -30.0, dtype=torch.float32)
    x[0, 2], x[0, 3] = 0.0, -20.0
    tables = tables_for(PAR, bucket=4)
    s = {"x": x, "tables": tables, "a": mask({0}, 4), "b": mask({1}, 4), "l_eff": 1}
    (mx, arg), _ = _captured_class_max(s, monkeypatch)
    assert int(_map(s).tokens[0]) == 2, "fixture: the MAP takes token 2"
    bent = mx.clone()
    bent[:, 0] = -1e30
    assert int(_map(s, class_max=(bent, arg)).tokens[0]) == 3, "perturbed class_max was not used"


def test_constrained_map_with_class_max_drops_its_lv_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _setup(_toy(), torch.float32)
    cmx, n_cmx = _captured_class_max(s, monkeypatch)
    n_without, _ = _lv_ops(lambda: _map(s))
    n_with, _ = _lv_ops(lambda: _map(s, class_max=cmx))
    assert n_cmx > 0
    assert n_without - n_with >= n_cmx, (n_without, n_with, n_cmx)


# --------------------------------------------------------------------------- signatures


@pytest.mark.parametrize(
    ("mod", "fn", "positional", "keyword"),
    [
        ("emission", "log_transfer", ["logits", "tables", "l_eff"], "class_mass"),
        (
            "marginals",
            "constrained_marginals",
            ["logits", "tables", "tree", "a_start", "b_final", "l_eff"],
            "class_mass",
        ),
        (
            "draw",
            "joint_draw",
            ["logits", "tables", "tree", "a_start", "b_final", "l_eff"],
            "class_log_weights",
        ),
        (
            "mapdecode",
            "constrained_map",
            ["logits", "tables", "a_start", "b_final", "l_eff"],
            "class_max",
        ),
    ],
)
def test_positional_signatures_unchanged_and_keyword_is_keyword_only(
    mod: str, fn: str, positional: list[str], keyword: str
) -> None:
    sig = inspect.signature(getattr(core(mod), fn))
    pos = [
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert pos == positional, f"{mod}.{fn} positional parameters changed: {pos}"
    p = sig.parameters.get(keyword)
    if p is None:
        pytest.fail(f"T-0072 not implemented: {mod}.{fn} has no keyword-only {keyword!r}")
    assert p.kind is p.KEYWORD_ONLY and p.default is None, (keyword, p.kind, p.default)


# --------------------------------------------------------------------------- sampler: class_max once per call


def test_sampler_computes_class_max_once_per_slot_across_l_eff_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One constrained + one unconstrained slot, CL 8, through T-0067's stand-in. On CPU the reports are read
    at the start of the next call, so a single ``L_eff`` candidate is the norm; the row widens the candidate
    list by patching ``sampler._l_eff_candidates`` (T-0067's private helper) to add a second, distinct value
    that the device never selects (harmless: selection is ``l_dev == l_eff``). Liveness: some call runs
    ``constrained_map`` twice for the slot."""
    from tests.cd.fakes.diffusion_sampler import Cfg, World, random_live_logits
    from tests.cd.unit._sampler_support import G1, LIVE, Grammar, cd_xargs, patch_everywhere

    sm = core("sampler")
    real_cands = getattr(sm, "_l_eff_candidates", None)
    if real_cands is None:
        pytest.fail("cannot widen the L_eff candidates: sampler._l_eff_candidates (T-0067) is gone")

    def wider(req: Any, valid: int, CL: int) -> list[int]:
        c = list(real_cands(req, valid, CL))
        extra = max(0, min(c) - 1)
        return sorted(set(c) | {extra})

    monkeypatch.setattr(sm, "_l_eff_candidates", wider)
    md = core("mapdecode")
    calls = {"map": 0, "cmax": 0}
    real_map, real_cmax = md.constrained_map, md.class_max

    def count_map(*a: Any, **k: Any) -> Any:
        calls["map"] += 1
        return real_map(*a, **k)

    def count_cmax(*a: Any, **k: Any) -> Any:
        calls["cmax"] += 1
        return real_cmax(*a, **k)

    assert patch_everywhere(monkeypatch, "constrained_map", count_map) >= 1
    assert patch_everywhere(monkeypatch, "class_max", count_cmax) >= 1
    g = Grammar(G1, "t0072-cmax-once")
    w = World(Cfg(CL=8), cd=True, seed=0)
    w.add("c", xargs=cd_xargs(g, "t0072-cmax"))
    w.add("u")
    w.prefill()
    per_call = []
    for _ in range(6):
        calls.update(map=0, cmax=0)
        w.decode(lambda c, r, n: random_live_logits(10 * c + len(r), LIVE, n))
        per_call.append((calls["map"], calls["cmax"]))
    assert any(m >= 2 for m, _ in per_call), (
        f"liveness: no call ran two L_eff candidates: {per_call}"
    )
    assert all(1 <= k <= 1 for m, k in per_call if m), (
        f"class_max computed more than once per constrained slot per call (map, class_max): {per_call}"
    )
