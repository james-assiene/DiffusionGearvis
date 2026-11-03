"""T-0077a: per-step CD on the real states ``[·, S, S]``, not the padded bucket ``[·, B, B]`` (criteria 2-4).

Reference: the frozen pre-change path ``tests/cd/oracle/core_frozen_t0077/`` (e73cfa7, byte-identical; MANIFEST.md),
run through ``cd/steps.py``'s per-candidate composition (``_frozen_t0077.run``) on each path's own tables.

Rows (umbrella ``docs/tasks/T-0077.md`` N3/N4; task criteria 2-4):
- **N3, exact parts** (``torch.equal``): ``class_max``, the max-plus tree, the MAP's tokens/score/``empty``, δ*,
  ``terminal_factor``, the draw's tokens/states/``empty``, the CD generator's state after ``joint_draw``, and the
  leaves ``log_transfer`` returns (no sum-product: an index copy).
- **N4, sum-product parts**: log Z (T-0065 c1 form, 1e-4·max(1, |log Z|)), q (≤ 1e-6), H(q) (≤ 1e-5) on rows
  ``< L_eff``; ``log_a``/``log_b`` and the pass-1 tree (true node = level + scale) at the log-Z form, with an
  identical dead pattern; every compensated product the live pass 2 issues against fp64 on its fp32 inputs (the
  T-0065 gauged bar, see ``test_n4_served_compensated_products``) and T-0073 D0-(c)'s wide bar at the live shapes.
- **Live states (criterion 2)** — red until the trees really run on ``[·, S, S]``: an op-shape spy (every ≥ 4-d
  tensor the tree products touch has its last three dims ≤ S), an argmax spy on the descents (no choice over
  B or B² entries), and padded-state poison (live outputs must not change when the padded block of the tree they
  read is set to log 1).
- **Adversarial**: the only accepting path uses state S−1 (kills a ``[:S-1]`` slice); a poisoned first padded
  state S (kills ``[:S+1]``); S = bucket (``s32_b32``); an empty language (sentinel and ``empty`` identical);
  NFAs with parallel edges; ``L_eff < L`` with the identity tail.

Regime: CPU, torch per ``uv.lock`` (2.13.0+cpu), fp32; instance seeds in ``_frozen_t0077.INSTANCES``; draw seed =
instance seed + 1. Run serially (``OMP_NUM_THREADS=4``): torch threads stall under xdist.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.cd.unit import _frozen_t0077 as fz
from tests.cd.unit._draw_support import Replay

NAMES = list(fz.INSTANCES)
PADDED = [n for n in NAMES if fz.INSTANCES[n]["S"] < fz.INSTANCES[n]["B"]]
#: padded instances whose language is non-empty (the poison and spy rows need live paths)
PADDED_LIVE = [n for n in PADDED if fz.INSTANCES[n]["kind"] != "chain_empty"]
#: the large-S instances where bucket and live shapes differ by more than one state
SPY = ["s31_b64_nfa_leff", "s91_b128_nfa", "s205_b256"]
STOPS = (1,)


@functools.cache
def inst(name: str) -> fz.Inst:
    return fz.build_inst(name)


@functools.cache
def frozen_run(name: str) -> fz.Run:
    return fz.run("frozen", inst(name))


@functools.cache
def live_run(name: str) -> fz.Run:
    return fz.run("live", inst(name))


def _eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def _first_diff(
    got: torch.Tensor, want: torch.Tensor, labels: tuple[str, str] = ("live", "frozen")
) -> str:
    """The orchestrator's adjudication data (N5): how many entries differ, the first value pair and its |Δ|."""
    if got.shape != want.shape or got.dtype != want.dtype:
        return f"shape/dtype {tuple(got.shape)}/{got.dtype} vs {tuple(want.shape)}/{want.dtype}"
    g1, w1 = got.reshape(-1), want.reshape(-1)
    ne = (g1 != w1).nonzero().reshape(-1)
    if ne.numel() == 0:
        return "equal values, different bits (signed zero / NaN)"
    j = int(ne[0])
    idx = tuple(int(i) for i in np.unravel_index(j, tuple(got.shape))) if got.dim() else ()
    g, w = g1[j], w1[j]
    d = (g.double() - w.double()).abs().item() if got.is_floating_point() else "n/a"
    return (
        f"{ne.numel()} entries differ; first at {idx}: {labels[0]} {g.item()} "
        f"{labels[1]} {w.item()} |Δ| {d}"
    )


#: labels of the poison rows' comparisons
POISON = ("poisoned", "clean")


# --------------------------------------------------------------------------- the frozen copy itself


def test_frozen_copies_are_byte_identical_to_e73cfa7() -> None:
    """MANIFEST's sha256 per file, and (where the object exists) the blob at e73cfa7 itself."""
    bad = []
    for name in fz.FROZEN_MODULES:
        got = fz.file_sha256(fz.FROZEN_DIR / f"{name}.py")
        if got != fz.FROZEN_SHA256[name]:
            bad.append(f"{name}.py: sha256 {got} != MANIFEST {fz.FROZEN_SHA256[name]}")
        blob = subprocess.run(
            [
                "git",
                "show",
                f"{fz.FROZEN_COMMIT}:plugin/astra_vllm_plugin/astra_vllm_plugin/cd/{name}.py",
            ],
            cwd=fz.FROZEN_DIR,
            capture_output=True,
        )
        if blob.returncode == 0:
            import hashlib

            if hashlib.sha256(blob.stdout).hexdigest() != fz.FROZEN_SHA256[name]:
                bad.append(f"{name}.py: MANIFEST sha256 is not the blob at {fz.FROZEN_COMMIT}")
    assert not bad, "\n".join(bad)


def test_frozen_modules_load_side_by_side_with_the_live_ones() -> None:
    """The frozen package resolves its relative imports inside itself: no frozen module is (or imports) a live one."""
    for name in fz.FROZEN_MODULES:
        f = fz.frozen(name)
        assert f.__name__ == f"{fz.FROZEN_PKG}.{name}"
        assert f is not fz.live(name)
    assert fz.frozen("draw").LogTree is fz.frozen("tree").LogTree
    assert fz.frozen("draw").LogTree is not fz.live("tree").LogTree
    assert fz.frozen("marginals").log_matmul is fz.frozen("semiring").log_matmul
    leaked = [
        m
        for m in sys.modules
        if m.startswith(fz.FROZEN_PKG)
        and getattr(sys.modules[m], "__file__", "")
        and "plugin/astra_vllm_plugin" in str(sys.modules[m].__file__)
    ]
    assert not leaked, f"frozen package names resolve to plugin files: {leaked}"


# --------------------------------------------------------------------------- premises


def test_instance_premises() -> None:
    """The matrix covers what criterion 2/T-0077a's Tests-to-write name, measured on the frozen path."""
    seen_s = {fz.INSTANCES[n]["S"] for n in NAMES}
    assert {1, 2, 31, 32, 91, 205} <= seen_s
    assert any(fz.INSTANCES[n]["S"] == fz.INSTANCES[n]["B"] for n in NAMES), "no S = bucket case"
    for name in NAMES:
        r, i = frozen_run(name), inst(name)
        S = i.toy.n_states
        assert r.tables.n_states == S and r.tables.bucket == fz.INSTANCES[name]["B"]
        empty = fz.INSTANCES[name]["kind"] == "chain_empty"
        assert bool(r.map.empty) is empty and bool(r.draw.empty) is empty, name
    for name in ("s2_b64", "s31_b64_nfa_leff", "s91_b128_nfa", "s205_b256", "s32_b32"):
        assert not frozen_run(name).tables.single_edge_pairs, f"{name}: no parallel edges"
    for name in ("s31_b64_dfa", "s91_b128_dfa_leff"):
        assert frozen_run(name).tables.is_dfa, f"{name}: not a DFA"
    for name in ("s31_b64_nfa_leff", "s91_b128_dfa_leff", "last_state_s91_b128"):
        assert inst(name).l_eff < inst(name).logits.shape[0], f"{name}: L_eff = L"
    # the last live state carries the draw (so a slice that drops it changes the answer)
    for name in ("s31_b64_dfa", "s91_b128_dfa_leff", "last_state_s31_b64", "last_state_s91_b128"):
        S = inst(name).toy.n_states
        assert int(frozen_run(name).draw.states.max()) == S - 1, f"{name}: state S-1 unused"


# --------------------------------------------------------------------------- N3: exact parts


@pytest.mark.parametrize("name", NAMES)
def test_n3_class_max_bit_identical(name: str) -> None:
    lv, fr = live_run(name), frozen_run(name)
    for k, (g, w) in enumerate(zip(lv.cmax, fr.cmax, strict=True)):
        assert _eq(g, w), f"{name}: class_max[{k}] {_first_diff(g, w)}"


@pytest.mark.parametrize("name", NAMES)
def test_n3_leaves_bit_identical(name: str) -> None:
    """``log_transfer`` keeps its output shape ``[L, bucket, bucket]`` and its bits (an index copy)."""
    lv, fr = live_run(name), frozen_run(name)
    assert _eq(lv.log_m, fr.log_m), f"{name}: log_transfer {_first_diff(lv.log_m, fr.log_m)}"


@pytest.mark.parametrize("name", NAMES)
def test_n3_map_bit_identical(name: str) -> None:
    lv, fr = live_run(name).map, frozen_run(name).map
    for field in ("tokens", "score", "empty"):
        g, w = getattr(lv, field), getattr(fr, field)
        assert _eq(g, w), f"{name}: MAP {field} {_first_diff(g, w)}"


@pytest.mark.parametrize("name", NAMES)
def test_n3_draw_and_generator_bit_identical(name: str) -> None:
    """Same tokens, states and ``empty``; the CD generator consumed the same variates (equal state after)."""
    lv, fr = live_run(name), frozen_run(name)
    assert _eq(lv.gen_state, fr.gen_state), (
        f"{name}: generator state after joint_draw differs (number/shape of variates changed)"
    )
    for field in ("states", "tokens", "empty"):
        g, w = getattr(lv.draw, field), getattr(fr.draw, field)
        assert _eq(g, w), f"{name}: draw {field} {_first_diff(g, w)}"


@pytest.mark.parametrize("name", NAMES)
def test_n3_delta_star_and_terminal_factor_bit_identical(name: str) -> None:
    """δ* of the MAP from A, and ``terminal_factor`` at several budgets, on each path's own tables."""
    i = inst(name)
    budget = fz.live("budget")
    lv, fr = live_run(name), frozen_run(name)
    B = i.bucket
    a = fz.bool_mask(i.start, B)
    ids_l = lv.map.tokens[: i.l_eff].to(torch.int64)
    ids_f = fr.map.tokens[: i.l_eff].to(torch.int64)
    dl = budget.delta_star(lv.tables, a, ids_l, STOPS)
    df = budget.delta_star(fr.tables, a, ids_f, STOPS)
    assert _eq(torch.as_tensor(dl.states), torch.as_tensor(df.states)), f"{name}: δ* states"
    assert _eq(torch.as_tensor(dl.empty), torch.as_tensor(df.empty)), f"{name}: δ* empty"
    for r in (0, 1, 3, 10**6):
        tl, tf = budget.terminal_factor(lv.tables, r), budget.terminal_factor(fr.tables, r)
        assert _eq(tl, tf), f"{name}: terminal_factor(r={r}) {_first_diff(tl, tf)}"


@pytest.mark.parametrize("name", PADDED_LIVE)
def test_n3_maxplus_tree_on_live_states_is_the_bucket_trees_live_block(name: str) -> None:
    """The live ``max_up_sweep`` on the ``[L, S, S]`` leaves equals, bit for bit, the live block of the frozen
    bucket tree (max-plus is exact under any grouping); padding back must not alter it."""
    i = inst(name)
    S = i.toy.n_states
    md_f = fz.frozen("mapdecode")
    md_l = fz.live("mapdecode")
    assert hasattr(md_l, "max_up_sweep"), (
        "mapdecode.max_up_sweep is gone (name the max-plus tree here)"
    )
    leaves = _map_leaves(md_f, name)
    want = md_f.max_up_sweep(leaves)
    got = md_l.max_up_sweep(leaves[:, :S, :S].contiguous())
    assert len(got) == len(want)
    for k, (g, w) in enumerate(zip(got, want, strict=True)):
        g = g[:, :S, :S]
        w = w[:, :S, :S]
        assert _eq(g, w), f"{name}: max-plus level {k} {_first_diff(g, w)}"


def _map_leaves(md: Any, name: str) -> torch.Tensor:
    """The MAP leaves ``constrained_map`` builds (frozen code: max over the edges of a pair, identity tail)."""
    i, r = inst(name), frozen_run(name)
    t = r.tables
    L = i.logits.shape[0]
    B = t.bucket
    wmax, _ = r.cmax
    ew = wmax.index_select(1, t.edge_class)
    pw = torch.full((L, t.n_pairs), md.SENTINEL, dtype=ew.dtype)
    pw.scatter_reduce_(
        1, t.edge_pair.unsqueeze(0).expand(L, -1), ew, reduce="amax", include_self=True
    )
    leaves = torch.full((L, B * B), md.SENTINEL, dtype=ew.dtype)
    leaves.index_copy_(1, t.pair_flat, pw)
    leaves = leaves.view(L, B, B)
    if i.l_eff < L:
        tail = leaves[i.l_eff :]
        tail.fill_(md.SENTINEL)
        tail.diagonal(dim1=-2, dim2=-1).fill_(0.0)
    return leaves


# --------------------------------------------------------------------------- N4: sum-product parts


def _log_close(got: torch.Tensor, want: torch.Tensor, what: str) -> list[str]:
    """Identical dead pattern (≤ -1e29); live entries within T-0065 c1's fp32 log-Z form 1e-4·max(1, |v|)."""
    if got.shape != want.shape:
        return [f"{what}: shape {tuple(got.shape)} vs {tuple(want.shape)}"]
    g, w = got.double(), want.double()
    dg, dw = g <= fz.SENTINEL_MAX, w <= fz.SENTINEL_MAX
    if not torch.equal(dg, dw):
        return [f"{what}: dead pattern differs at {int((dg != dw).sum())} entries"]
    live = ~dw
    if not bool(live.any()):
        return []
    err = (g - w).abs()[live]
    tol = 1e-4 * w.abs()[live].clamp_min(1.0)
    bad = err > tol
    if bool(bad.any()):
        j = int(torch.argmax(err / tol))
        return [f"{what}: |Δ| {err[j].item():.3g} > {tol[j].item():.3g} ({int(bad.sum())} entries)"]
    return []


@pytest.mark.parametrize("name", NAMES)
def test_n4_log_z(name: str) -> None:
    lv, fr = live_run(name).marg, frozen_run(name).marg
    assert not _log_close(lv.log_z.reshape(1), fr.log_z.reshape(1), f"{name}: log Z"), (
        f"{name}: log Z live {float(lv.log_z)} frozen {float(fr.log_z)}"
    )


@pytest.mark.parametrize("name", [n for n in NAMES if fz.INSTANCES[n]["kind"] != "chain_empty"])
def test_n4_q_and_entropy(name: str) -> None:
    """dq ≤ 1e-6 and dH ≤ 1e-5 on rows ``< L_eff`` (T-0065 R6: rows beyond are unspecified)."""
    i = inst(name)
    lv, fr = live_run(name).marg, frozen_run(name).marg
    assert lv.q.shape == fr.q.shape and lv.entropy.shape == fr.entropy.shape
    dq = (lv.q[: i.l_eff].double() - fr.q[: i.l_eff].double()).abs().max().item()
    dh = (lv.entropy[: i.l_eff].double() - fr.entropy[: i.l_eff].double()).abs().max().item()
    assert dq <= 1e-6, f"{name}: max |Δq| {dq:.3g} > 1e-6"
    assert dh <= 1e-5, f"{name}: max |ΔH(q)| {dh:.3g} > 1e-5"


@pytest.mark.parametrize("name", NAMES)
def test_n4_forward_backward_vectors(name: str) -> None:
    """``log_a``/``log_b`` stay ``[L+1, bucket]``; padded states stay dead; live entries within the log-Z form."""
    i = inst(name)
    S = i.toy.n_states
    lv, fr = live_run(name).marg, frozen_run(name).marg
    bad = _log_close(lv.log_a, fr.log_a, f"{name}: log_a") + _log_close(
        lv.log_b, fr.log_b, f"{name}: log_b"
    )
    for v, what in ((lv.log_a, "log_a"), (lv.log_b, "log_b")):
        if v.shape[-1] > S and not bool((v[:, S:] <= fz.SENTINEL_MAX).all()):
            bad.append(f"{name}: {what} has a live padded state")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("name", NAMES)
def test_n4_pass1_tree(name: str) -> None:
    """``up_sweep`` keeps its output shapes (levels ``[n, bucket, bucket]``, scales ``[n]``); true nodes
    (level + scale) within the log-Z form of the frozen tree's, with the same dead pattern."""
    lv, fr = live_run(name).tree, frozen_run(name).tree
    assert len(lv.levels) == len(fr.levels) and len(lv.log_scales) == len(fr.log_scales)
    bad = []
    for k, (gl, wl, gs, ws) in enumerate(
        zip(lv.levels, fr.levels, lv.log_scales, fr.log_scales, strict=True)
    ):
        if gl.shape != wl.shape or gs.shape != ws.shape:
            bad.append(
                f"level {k}: shapes {tuple(gl.shape)}/{tuple(gs.shape)} vs {tuple(wl.shape)}"
            )
            continue
        g = torch.where(
            gl <= fz.SENTINEL_MAX, gl.double(), gl.double() + gs.double()[:, None, None]
        )
        w = torch.where(
            wl <= fz.SENTINEL_MAX, wl.double(), wl.double() + ws.double()[:, None, None]
        )
        bad += _log_close(g, w, f"level {k}")
    assert not bad, f"{name}: " + "; ".join(bad)


class _CompSpy:
    """Every ``log_matmul(..., compensated=True)`` call the live modules make: (a, b, out) cloned."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.orig = fz.live("semiring").log_matmul
        self.saved: list[tuple[Any, str, Any]] = []

    def __call__(self, a: torch.Tensor, b: torch.Tensor, **kw: Any) -> torch.Tensor:
        out = self.orig(a, b, **kw)
        if kw.get("compensated", False):
            self.calls.append((a.detach().clone(), b.detach().clone(), out.detach().clone()))
        return out

    def __enter__(self) -> _CompSpy:
        for mname, mod in list(sys.modules.items()):
            if (
                mname.startswith("astra_vllm_plugin.cd")
                and getattr(mod, "log_matmul", None) is self.orig
            ):
                self.saved.append((mod, "log_matmul", self.orig))
                mod.log_matmul = self
        return self

    def __exit__(self, *exc: Any) -> None:
        for mod, attr, val in self.saved:
            setattr(mod, attr, val)


def _lse64(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """float64 ``log Σ_m exp(A + B)`` of the fp32 operands as received (exact sums of two fp32 values)."""
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    n, m, k = a.shape[-2], a.shape[-1], b.shape[-1]
    a3 = a.double().expand(*batch, n, m).reshape(-1, n, m).numpy()
    b3 = b.double().expand(*batch, m, k).reshape(-1, m, k).numpy()
    out = np.full((a3.shape[0], n, k), -1e30)
    for p in range(a3.shape[0]):
        t = a3[p][:, :, None] + b3[p][None, :, :]
        live = t > -1e29
        mx = np.where(live, t, -np.inf).max(axis=1)
        fin = np.isfinite(mx)
        s = np.where(live, np.exp(t - np.where(fin, mx, 0.0)[:, None, :]), 0.0).sum(axis=1)
        ok = fin & (s > 0)
        out[p] = np.where(ok, mx + np.log(np.where(ok, s, 1.0)), -1e30)
    return out.reshape(*batch, n, k)


@pytest.mark.parametrize("name", [n for n in NAMES if fz.INSTANCES[n]["kind"] != "chain_empty"])
def test_n4_served_compensated_products(name: str) -> None:
    """Every compensated product of the live pass 2, against fp64 on its fp32-rounded inputs (T-0073 D0-(c)'s
    oracle). Bar: T-0065's gauged-regime bar (``test_core_compensated.py``), error in units of
    max(ulp(result), ulp(1.0)): max ≤ 2.5, mean ≤ 0.75. (D0-(c)'s relative ≤ 1 ulp / ≤ 5 % bar is not met by the
    frozen path's own pass-2 products, whose results sit near 0 — measured 2026-10-04 at e73cfa7: relative max
    1.2-6.7 ulp, 2-30 % > 0.51 — so it is applied at the live shapes on its own regime below.)"""
    i = inst(name)
    em = fz.live("emission")
    r = live_run(name)
    x = i.logits.clone()
    B = i.bucket
    cm = em.class_mass(x, r.tables)
    with _CompSpy() as spy:
        fz.live("marginals").constrained_marginals(
            x,
            r.tables,
            r.tree,
            fz.bool_mask(i.start, B),
            fz.bool_mask(i.end, B),
            i.l_eff,
            class_mass=cm,
        )
    assert spy.calls, (
        "no compensated product observed (pass 2 must use log_matmul(compensated=True))"
    )
    errs = []
    for a, b, out in spy.calls:
        want = _lse64(a, b)
        got = out.double().numpy()
        live = want > -1e20
        unit = np.maximum(np.spacing(np.abs(want).astype(np.float32)).astype(np.float64), 2.0**-23)
        errs.append((np.abs(got - want) / unit)[live])
    e = np.concatenate(errs)
    assert e.size, "no live entry in any compensated product"
    assert e.max() <= 2.5 and e.mean() <= 0.75, (
        f"{name}: compensated products max {e.max():.2f}, mean {e.mean():.2f} (bar 2.5 / 0.75)"
    )


class _AllProducts:
    """Every ``log_matmul`` call the live ``cd`` modules make, tagged with the phase it ran in: ``pass1`` until
    ``marginals.up_sweep`` (the pass-2 tree) is entered, ``pass2_tree`` inside it, ``pass2_down`` after it returns."""

    def __init__(self) -> None:
        self.orig = fz.live("semiring").log_matmul
        self.phase = "pass1"
        self.calls: list[tuple[str, bool]] = []
        self.saved: list[tuple[Any, str, Any]] = []

    def __call__(self, a: torch.Tensor, b: torch.Tensor, **kw: Any) -> torch.Tensor:
        self.calls.append((self.phase, bool(kw.get("compensated", False))))
        return self.orig(a, b, **kw)

    def __enter__(self) -> _AllProducts:
        for mname, mod in list(sys.modules.items()):
            if (
                mname.startswith("astra_vllm_plugin.cd")
                and getattr(mod, "log_matmul", None) is self.orig
            ):
                self.saved.append((mod, "log_matmul", self.orig))
                mod.log_matmul = self
        marg = fz.live("marginals")
        assert hasattr(marg, "up_sweep"), "marginals.up_sweep is gone (the pass-2 tree's call site)"
        inner = marg.up_sweep

        def pass2_tree(*args: Any, **kw: Any) -> Any:
            self.phase = "pass2_tree"
            try:
                return inner(*args, **kw)
            finally:
                self.phase = "pass2_down"

        self.saved.append((marg, "up_sweep", inner))
        marg.up_sweep = pass2_tree
        return self

    def __exit__(self, *exc: Any) -> None:
        for mod, attr, val in reversed(self.saved):
            setattr(mod, attr, val)


@pytest.mark.parametrize("name", ["s31_b64_nfa_leff", "s91_b128_nfa", "s205_b256"])
def test_pass2_products_are_all_compensated(name: str) -> None:
    """N6 / T-0065 decision 10 (ruling L7): pass 2 is compensated end to end. Every product issued from the pass-2
    up-sweep on, its tree and its down-sweep, carries ``compensated=True``; pass 1's down-sweep is the plain path.
    The gauged bar alone has no power here (dropping compensation stays under 2.5 / 0.75 on these instances).
    Kills: ``marginals.py`` ``up_sweep(leaves, compensated=True)`` → ``False`` (pass2_tree rows), and the pass-2
    ``_down_sweep(..., compensated=True)`` → ``False`` (pass2_down rows)."""
    i = inst(name)
    r = live_run(name)
    B = i.bucket
    with _AllProducts() as spy:
        fz.live("marginals").constrained_marginals(
            i.logits.clone(),
            r.tables,
            r.tree,
            fz.bool_mask(i.start, B),
            fz.bool_mask(i.end, B),
            i.l_eff,
        )
    by = {p: [c for ph, c in spy.calls if ph == p] for p in ("pass1", "pass2_tree", "pass2_down")}
    assert by["pass2_tree"], f"{name}: no product inside the pass-2 up-sweep (spy dead?)"
    assert by["pass2_down"], f"{name}: no product after the pass-2 up-sweep (no pass-2 down-sweep?)"
    assert by["pass1"] and not any(by["pass1"]), (
        f"{name}: pass 1 must issue plain products (control on the phase split): {by['pass1']}"
    )
    for ph in ("pass2_tree", "pass2_down"):
        n_plain = sum(1 for c in by[ph] if not c)
        assert n_plain == 0, f"{name}: {n_plain} of {len(by[ph])} {ph} products uncompensated"


@pytest.mark.parametrize("S", [91, 205])
def test_n4_compensated_product_wide_bar_at_live_shapes(S: int) -> None:
    """T-0073 D0-(c) on a ``[p, S, S] ⊗ [p, S, S]`` product (the live-state operand shape): entries in [-24, -16],
    30 % sentinels, oracle from the fp32-rounded inputs; max ≤ 1 ulp and ≤ 5 % > 0.51 ulp. The uncompensated
    control on the same input must exceed 5 % (else the input does not discriminate)."""
    sr = fz.live("semiring")
    g = np.random.default_rng(S)
    a = g.uniform(-24, -16, (3, S, S))
    b = g.uniform(-24, -16, (3, S, S))
    a[g.random(a.shape) < 0.3] = sr.SENTINEL
    b[g.random(b.shape) < 0.3] = sr.SENTINEL
    ta, tb = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    want = _lse64(ta, tb)
    ulp = np.spacing(np.abs(want).astype(np.float32)).astype(np.float64)
    live = want > -1e20

    def err(res: torch.Tensor) -> np.ndarray:
        return (np.abs(res.double().numpy() - want) / ulp)[live]

    e_u = err(sr.log_matmul(ta, tb, compensated=False))
    assert float(np.mean(e_u > 0.51)) > 0.05, "control: the input does not discriminate"
    e_c = err(sr.log_matmul(ta, tb, compensated=True))
    assert float(e_c.max()) <= 1.0 and float(np.mean(e_c > 0.51)) <= 0.05, (
        f"S {S}: max {e_c.max():.2f} ulp, {np.mean(e_c > 0.51):.1%} > 0.51 ulp"
    )


# --------------------------------------------------------------------------- criterion 2: live states


class _ShapeSpy(TorchDispatchMode):
    """Largest of the last three dims of every ≥ 4-d tensor an op reads or writes (the [g, r, m, c] temporaries
    of the semiring/max-plus products), and the shape of every argmax/max(dim) input."""

    def __init__(self) -> None:
        super().__init__()
        self.big4: list[tuple[str, tuple[int, ...]]] = []
        self.choices: list[tuple[str, tuple[int, ...]]] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        flat = list(args) + list((kwargs or {}).values())
        flat += list(out) if isinstance(out, (tuple, list)) else [out]
        for t in flat:
            if isinstance(t, torch.Tensor) and t.dim() >= 4:
                self.big4.append((str(func), tuple(t.shape)))
        name = str(func)
        if name.startswith(("aten.argmax", "aten.max.dim")) and isinstance(args[0], torch.Tensor):
            self.choices.append((name, tuple(args[0].shape)))
        return out


def _component(name: str, comp: str) -> _ShapeSpy:
    i = inst(name)
    m = fz.mods("live")
    r = live_run(name)
    x = i.logits.clone()
    B = i.bucket
    a, b = fz.bool_mask(i.start, B), fz.bool_mask(i.end, B)
    cm = m["emission"].class_mass(x, r.tables)
    log_m = m["emission"].log_transfer(x, r.tables, i.l_eff, class_mass=cm)
    spy = _ShapeSpy()
    with spy:
        if comp == "pass1_up_sweep":
            m["tree"].up_sweep(log_m)
        elif comp == "constrained_marginals":
            m["marginals"].constrained_marginals(x, r.tables, r.tree, a, b, i.l_eff, class_mass=cm)
        elif comp == "constrained_map":
            m["mapdecode"].constrained_map(x, r.tables, a, b, i.l_eff, class_max=r.cmax)
        else:
            g = torch.Generator().manual_seed(i.draw_seed)
            m["draw"].joint_draw(x, r.tables, r.tree, a, b, i.l_eff, generator=g)
    return spy


@pytest.mark.parametrize("comp", ["pass1_up_sweep", "constrained_marginals", "constrained_map"])
@pytest.mark.parametrize("name", SPY)
def test_tree_products_run_on_live_states(name: str, comp: str) -> None:
    """Criterion 2: every ≥ 4-d temporary of the pass-1 tree, the down-sweeps + pass-2 tree, and the max-plus
    sweep has its last three dims ≤ S (a bucket-sized product has m = B). Kills: any of the three trees or a
    down-sweep left on ``[·, B, B]``."""
    S = inst(name).toy.n_states
    spy = _component(name, comp)
    assert spy.big4, f"{name}/{comp}: the spy saw no ≥ 4-d tensor (instrument dead?)"
    worst = max(spy.big4, key=lambda kv: max(kv[1][-3:]))
    assert max(worst[1][-3:]) <= S, (
        f"{name}/{comp}: a ≥ 4-d temporary has shape {worst[1]} (op {worst[0]}) > S = {S}, bucket "
        f"{inst(name).bucket}: the product still runs on the padded bucket"
    )


@pytest.mark.parametrize("comp", ["joint_draw", "constrained_map"])
@pytest.mark.parametrize("name", SPY)
def test_descents_choose_among_live_states(name: str, comp: str) -> None:
    """Criterion 2: the boundary pair and the midpoint choices of the draw's descent and the MAP's descent are taken
    over S (or S²) entries, never B (or B²). The draw still draws its uniforms at the padded shapes (generator
    rows above); only the choice is sliced. Kills: a descent left on the bucket."""
    i = inst(name)
    S, B = i.toy.n_states, i.bucket
    E = int(live_run(name).tables.edge_src.numel())
    V = i.logits.shape[1]
    assert len({B, B * B} & {S, S * S, E, V}) == 0, (
        "fixture: B/B² must not coincide with S, S², E or V"
    )
    spy = _component(name, comp)
    lasts = [shape[-1] if shape else 1 for _, shape in spy.choices]
    on_bucket = [c for c, n in zip(spy.choices, lasts, strict=True) if n in (B, B * B)]
    assert not on_bucket, f"{name}/{comp}: choices over the bucket: {on_bucket[:4]}"
    assert any(n in (S, S * S) for n in lasts), (
        f"{name}/{comp}: no choice over S or S² entries seen ({sorted(set(lasts))})"
    )


def _poison(t: torch.Tensor, S: int, which: str) -> torch.Tensor:
    """Set the padded block (``all``) or only padded state S (``first``) to log 1 in both state axes."""
    p = t.clone()
    hi = S + 1 if which == "first" else p.shape[-1]
    p[..., S:hi, :] = 0.0
    p[..., :, S:hi] = 0.0
    return p


@pytest.mark.parametrize("which", ["first", "all"])
@pytest.mark.parametrize("name", PADDED_LIVE)
def test_pass1_tree_ignores_padded_states(name: str, which: str) -> None:
    """``up_sweep(log_transfer(...))`` as ``steps.py`` composes it: padded entries of the leaves set to log 1 in place
    change nothing (the live tree never reads them; outputs padded back with the sentinel). ``first`` kills an
    off-by-one ``[:S+1]`` slice; ``all`` kills the bucket path."""
    i = inst(name)
    S = i.toy.n_states
    m = fz.mods("live")
    r = live_run(name)
    x = i.logits.clone()
    cm = m["emission"].class_mass(x, r.tables)
    log_m = m["emission"].log_transfer(x, r.tables, i.l_eff, class_mass=cm)
    # The clean reference runs on the *same* tensor object, before poisoning: ``clone()`` drops whatever
    # ``log_transfer`` attached to its output (T-0077a carries S as a tensor attribute), so a cloned reference would
    # take the bucket path and differ by summation order alone (found at 9c9c508, |Δ| ≤ 9.5e-7).
    clean_tree = m["tree"].up_sweep(log_m)
    clean = dataclasses.replace(
        clean_tree,
        levels=tuple(t.clone() for t in clean_tree.levels),
        log_scales=tuple(t.clone() for t in clean_tree.log_scales),
    )
    unpoisoned = log_m.clone()
    log_m.copy_(
        _poison(log_m, S, which)
    )  # in place: whatever log_transfer attached to its output stays
    got = m["tree"].up_sweep(log_m)
    # Power control: a live path that reads the poisoned states (``[:S+1]`` for ``first``, the bucket for ``all``),
    # forced through the L2 keyword, must see the poison; else this row could not tell the difference.
    reach = S + 1 if which == "first" else i.bucket
    ctl_clean = m["tree"].up_sweep(unpoisoned, n_states=reach)
    ctl_pois = m["tree"].up_sweep(log_m.clone(), n_states=reach)
    assert not all(_eq(a, b) for a, b in zip(ctl_pois.levels, ctl_clean.levels, strict=True)), (
        f"{name}/{which}: control: a tree on {reach} states does not see the poison"
    )
    bad = [
        f"level {k}: {_first_diff(g, w, POISON)}"
        for k, (g, w) in enumerate(zip(got.levels, clean.levels, strict=True))
        if not _eq(g, w)
    ] + [
        f"scale {k}: {_first_diff(g, w, POISON)}"
        for k, (g, w) in enumerate(zip(got.log_scales, clean.log_scales, strict=True))
        if not _eq(g, w)
    ]
    assert not bad, f"{name}/{which}: the pass-1 tree read padded states: " + "; ".join(bad[:4])


def _poisoned_tree(name: str, which: str) -> Any:
    tree = live_run(name).tree
    S = inst(name).toy.n_states
    return dataclasses.replace(tree, levels=tuple(_poison(lv, S, which) for lv in tree.levels))


@pytest.mark.parametrize("which", ["first", "all"])
@pytest.mark.parametrize("name", PADDED_LIVE)
def test_marginals_ignore_padded_states_of_the_tree(name: str, which: str) -> None:
    """The pass-1 down-sweep reads only the live block of the caller's tree: log Z, q, H, log_a, log_b are
    bit-identical with the tree's padded block set to log 1."""
    i = inst(name)
    r = live_run(name)
    B = i.bucket
    a, b = fz.bool_mask(i.start, B), fz.bool_mask(i.end, B)
    marg = fz.live("marginals")
    clean = marg.constrained_marginals(i.logits.clone(), r.tables, r.tree, a, b, i.l_eff)
    got = marg.constrained_marginals(
        i.logits.clone(), r.tables, _poisoned_tree(name, which), a, b, i.l_eff
    )
    bad = [
        f"{f}: {_first_diff(getattr(got, f), getattr(clean, f), POISON)}"
        for f in ("log_z", "q", "entropy", "log_a", "log_b")
        if not _eq(getattr(got, f), getattr(clean, f))
    ]
    assert not bad, f"{name}/{which}: marginals read padded tree states: " + "; ".join(bad)


@pytest.mark.parametrize("which", ["first", "all"])
@pytest.mark.parametrize("name", PADDED_LIVE)
def test_draw_ignores_padded_states_of_the_tree(name: str, which: str) -> None:
    """The draw's boundary pair and descent read only the live block: same tokens, states, ``empty`` and generator
    state with the tree's padded block set to log 1 (a padded midpoint at log 1 would win every choice)."""
    i = inst(name)
    r = live_run(name)
    B = i.bucket
    a, b = fz.bool_mask(i.start, B), fz.bool_mask(i.end, B)
    dr = fz.live("draw")
    out = []
    for tree in (r.tree, _poisoned_tree(name, which)):
        g = torch.Generator().manual_seed(i.draw_seed)
        d = dr.joint_draw(i.logits.clone(), r.tables, tree, a, b, i.l_eff, generator=g)
        out.append((d, g.get_state()))
    (dc, gc), (dp, gp) = out
    bad = [
        f"{f}: {_first_diff(getattr(dp, f), getattr(dc, f), POISON)}"
        for f in ("states", "tokens", "empty")
        if not _eq(getattr(dp, f), getattr(dc, f))
    ]
    if not _eq(gp, gc):
        bad.append("generator state")
    assert not bad, f"{name}/{which}: the draw read padded tree states: " + "; ".join(bad)


# --------------------------------------------------------------------------- adversarial, against hand-built answers


@pytest.mark.parametrize("name", ["last_state_s31_b64", "last_state_s91_b128"])
def test_only_accepting_path_through_last_state(name: str) -> None:
    """``0 -{1,2}-> S-1 -{3,4}-> S-1``, distractors 1..S-2 never reach F. By hand: the MAP is 2 then 4 on
    ``[1, L_eff)``; the draw's states are S-1 from boundary 1 to ``L_eff`` and its tokens replay into F. A live
    slice that drops state S-1 makes the language empty."""
    i = inst(name)
    S = i.toy.n_states
    _, (first, rest) = fz.last_state_toy(S, i.toy.vocab_size)
    lv = live_run(name)
    want = [first] + [rest] * (i.l_eff - 1)
    assert not bool(lv.map.empty) and not bool(lv.draw.empty), f"{name}: language reported empty"
    assert lv.map.tokens[: i.l_eff].tolist() == want, f"{name}: MAP {lv.map.tokens.tolist()}"
    st = lv.draw.states.tolist()
    assert st[0] == 0 and all(s == S - 1 for s in st[1 : i.l_eff + 1]), f"{name}: states {st}"
    toks = lv.draw.tokens[: i.l_eff].tolist()
    assert Replay(i.toy).accepts({0}, {S - 1}, toks), f"{name}: drawn tokens {toks} not accepted"
    assert set(toks[:1]) <= {1, 2} and set(toks[1:]) <= {3, 4}


def test_empty_language_sentinel_and_flags() -> None:
    """Chain ``0 → … → 30`` with L = 11: no string reaches F. ``empty`` is set by both the draw and the MAP, the MAP
    score is exactly the sentinel, log Z is at or below it (T-0065 R6a); the frozen path agrees bit for bit."""
    name = "empty_s31_b64"
    lv, fr = live_run(name), frozen_run(name)
    sr = fz.live("semiring")
    assert bool(lv.map.empty) and bool(lv.draw.empty)
    assert _eq(lv.map.score, torch.tensor(sr.SENTINEL, dtype=lv.map.score.dtype))
    assert float(lv.marg.log_z) <= fz.SENTINEL_MAX
    assert _eq(lv.map.score, fr.map.score) and _eq(lv.map.empty, fr.map.empty)
    assert _eq(lv.draw.empty, fr.draw.empty)


def test_public_output_shapes_unchanged() -> None:
    """Umbrella criterion 2: output shapes of ``log_transfer``, ``up_sweep``, ``constrained_marginals``,
    ``constrained_map`` and ``joint_draw`` equal the frozen path's (bucket shapes) at S < B."""
    name = "s91_b128_nfa"
    lv, fr = live_run(name), frozen_run(name)
    pairs = [
        ("log_m", lv.log_m, fr.log_m),
        *[
            (f"levels[{k}]", g, w)
            for k, (g, w) in enumerate(zip(lv.tree.levels, fr.tree.levels, strict=False))
        ],
        *[
            (f"log_scales[{k}]", g, w)
            for k, (g, w) in enumerate(zip(lv.tree.log_scales, fr.tree.log_scales, strict=False))
        ],
        *[
            (f"marg.{f}", getattr(lv.marg, f), getattr(fr.marg, f))
            for f in ("log_z", "q", "entropy", "log_a", "log_b")
        ],
        *[
            (f"map.{f}", getattr(lv.map, f), getattr(fr.map, f))
            for f in ("tokens", "score", "empty")
        ],
        *[
            (f"draw.{f}", getattr(lv.draw, f), getattr(fr.draw, f))
            for f in ("tokens", "states", "empty")
        ],
    ]
    bad = [
        f"{n}: {tuple(g.shape)}/{g.dtype} vs {tuple(w.shape)}/{w.dtype}"
        for n, g, w in pairs
        if g.shape != w.shape or g.dtype != w.dtype
    ]
    assert len(lv.tree.levels) == len(fr.tree.levels)
    assert not bad, "; ".join(bad)
    assert math.isfinite(float(lv.marg.log_z))
