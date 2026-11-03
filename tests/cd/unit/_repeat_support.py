"""T-0073 D4's bonus property: constrained decoding outputs are bitwise repeatable on the same device.

Shared by the CPU rows (``tests/cd/unit/test_cd_repeatability.py``, 3 repeats) and the CUDA rows
(``tests/cd/cuda/test_cd_cuda_numerics.py``, 20 repeats inside the pinned container). Each check runs the same
call ``n`` times in one process from identical inputs (fresh generator with the same seed for the draw; a fresh
``World`` with the same global seed for the sampler) and compares every output with the first run **bit for bit**
(NaN-safe ``view`` as integers). Returns a list of human-readable differences; ``[]`` means repeatable.

Instance for the core calls: the (a) instance of ``test_t0065_c2_differential_vs_stock_error_on_cuda`` (L 256,
V 4096, ``random_toy(Random(7), S=12, nfa=True)``, ``Generator(3)`` logits, softcap 30, T 0.4) — on CUDA it is
where launch 2 saw ``index_add_`` in ``class_mass`` differ across repeats (D4).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import torch

from tests.cd.oracle.enumerate import Toy
from tests.cd.unit._core_support import core, random_toy


def a_instance(temp: float = 0.4) -> tuple[Toy, torch.Tensor, int]:
    """(toy, fp32 logits on CPU, L): the c2 row's instance (D2: the fp32 round of its float64 logits)."""
    rng = random.Random(7)
    V, L = 4096, 256
    toy = random_toy(rng, V=V, S=12, nfa=True)
    g = torch.Generator().manual_seed(3)
    raw = torch.randn(L, V, generator=g, dtype=torch.float64) * 40.0
    return toy, ((30.0 * torch.tanh(raw / 30.0)) / temp).to(torch.float32), L


def tables(toy: Toy, device: torch.device) -> Any:
    return core("emission").build_tables(
        n_states=toy.n_states,
        vocab_size=toy.vocab_size,
        class_members=[sorted(s) for s, _ in toy.classes],
        class_is_neg=[n for _, n in toy.classes],
        edge_src=[e[0] for e in toy.edges],
        edge_dst=[e[1] for e in toy.edges],
        edge_class=[e[2] for e in toy.edges],
        finals=sorted(toy.finals),
        device=device,
    )


def vec(idx: Any, n: int, device: torch.device) -> torch.Tensor:
    m = torch.zeros(n, dtype=torch.bool, device=device)
    m[list(idx)] = True
    return m


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype == torch.float64:
        return t.view(torch.int64)
    return t


def compare_runs(runs: list[dict[str, torch.Tensor]]) -> list[str]:
    """Every field of every run vs run 0, bitwise; the differences as strings."""
    bad = []
    for i, r in enumerate(runs[1:], start=1):
        for k, v in r.items():
            if not torch.equal(_bits(runs[0][k]), _bits(v)):
                bad.append(f"repeat {i}: {k}")
    return bad


def _core_setup(
    device: torch.device, temp: float
) -> tuple[Any, torch.Tensor, Any, torch.Tensor, torch.Tensor, int]:
    toy, x, L = a_instance(temp)
    t = tables(toy, device)
    x = x.to(device)
    a, b = vec([0], t.bucket, device), vec(sorted(toy.finals), t.bucket, device)
    return t, x, toy, a, b, L


def marginals_runs(
    device: torch.device, n: int, temp: float = 0.4
) -> list[dict[str, torch.Tensor]]:
    """log_transfer -> up_sweep -> constrained_marginals, ``n`` times."""
    t, x, _toy, a, b, L = _core_setup(device, temp)
    em, tr, mg = core("emission"), core("tree"), core("marginals")
    out = []
    for _ in range(n):
        tree = tr.up_sweep(em.log_transfer(x, t, L))
        res = mg.constrained_marginals(x, t, tree, a, b, L)
        out.append(
            {
                "log_z": res.log_z,
                "q": res.q,
                "entropy": res.entropy,
                "log_a": res.log_a,
                "log_b": res.log_b,
            }
        )
    return out


def draw_runs(
    device: torch.device, n: int, temp: float = 0.4, draws: int = 4
) -> list[dict[str, torch.Tensor]]:
    """``joint_draw`` ``draws`` times per run from a fresh generator with the same seed, ``n`` runs."""
    t, x, _toy, a, b, L = _core_setup(device, temp)
    em, tr, dr = core("emission"), core("tree"), core("draw")
    out = []
    for _ in range(n):
        tree = tr.up_sweep(em.log_transfer(x, t, L))
        g = torch.Generator(device=device).manual_seed(1234)
        run: dict[str, torch.Tensor] = {}
        for k in range(draws):
            d = dr.joint_draw(x, t, tree, a, b, L, generator=g)
            run[f"draw{k}.tokens"] = d.tokens
            run[f"draw{k}.states"] = d.states
            run[f"draw{k}.empty"] = d.empty
        out.append(run)
    return out


def map_runs(device: torch.device, n: int, temp: float = 0.4) -> list[dict[str, torch.Tensor]]:
    t, x, _toy, a, b, L = _core_setup(device, temp)
    md = core("mapdecode")
    out = []
    for _ in range(n):
        m = md.constrained_map(x, t, a, b, L)
        out.append({"tokens": m.tokens, "score": m.score, "empty": m.empty})
    return out


def sampler_runs(n: int, calls: int = 4) -> list[dict[str, torch.Tensor]]:
    """One constrained request (G1, hand-built wire payload) plus one unconstrained, through the per-step
    ``ConstrainedDiffusionSampler`` of the fake engine (``tests/cd/fakes/diffusion_sampler.py``, device from
    ``ASTRA_CD_DEVICE``): a fresh ``World`` (same global seed) per run, ``calls`` decode calls each; every state
    row, ``draft_tokens``, output row and the global RNG state after each call are compared."""
    from tests.cd.fakes.diffusion_sampler import STATE_ROWS, Cfg, World, random_live_logits
    from tests.cd.unit._sampler_support import G1, LIVE, Grammar, cd_xargs

    g1 = Grammar(G1, "t0073-repeat-g1")

    def fn(call: int, req: str, cl: int) -> torch.Tensor:
        # per request, not per call: the canvas stabilises, so converge and commit steps are covered
        return random_live_logits(1 if req == "c" else 2, LIVE, cl)

    out = []
    for i in range(n):
        w = World(Cfg(CL=8, stability_threshold=1), cd=True, seed=0)
        w.add("c", xargs=cd_xargs(g1, f"t0073-repeat-{i}"))
        w.add("u")
        w.prefill()
        run: dict[str, torch.Tensor] = {}
        for c in range(calls):
            s = w.decode(fn)
            for r in ("c", "u"):
                for name in STATE_ROWS:
                    run[f"call{c}.{r}.{name}"] = s.row(name, r)
                for name in s.out:
                    run[f"call{c}.{r}.out.{name}"] = s.out_row(name, r)
            run[f"call{c}.draft_tokens"] = s.draft_tokens
            run[f"call{c}.rng_after"] = s.rng_after
        out.append(run)
    return out


CHECKS: dict[str, Callable[[torch.device, int], list[dict[str, torch.Tensor]]]] = {
    "constrained_marginals": marginals_runs,
    "joint_draw": draw_runs,
    "constrained_map": map_runs,
}
