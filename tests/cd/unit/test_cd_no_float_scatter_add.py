"""T-0073 D4, the CPU guard: no floating-point scatter-add on the per-step marginals path.

D4 located the CUDA failure (a) in ``index_add_`` inside ``class_mass``: CUDA scatter-adds are atomics, so the order
of the fp32 additions (and the result) changes from run to run. CPU ``index_add_`` is order-stable, so no CPU value
row can see a regression back to it; this row pins the *ops* instead. A ``TorchDispatchMode`` records every aten
call made by ``emission.class_mass``, ``emission.log_transfer`` (with and without the class-mass pass-through),
``tree.up_sweep``, ``marginals.constrained_marginals`` and ``marginals._token_marginals`` and fails on any of these
with a **floating** destination:

- ``index_add`` / ``index_add_``; ``scatter_add`` / ``scatter_add_``;
- ``scatter_reduce`` / ``scatter_reduce_`` with ``reduce`` in {sum, mean, prod};
- ``scatter`` / ``scatter_`` with ``reduce`` in {add, multiply};
- ``index_put`` / ``index_put_`` / ``_index_put_impl_`` / ``put_`` with ``accumulate=True``.

Allowed: integer and bool destinations (``draw.py``'s int32 ``index_add_``, count tensors) and ``amax`` / ``amin``
scatters (order-independent). ``budget.py`` (0/1 counts) and ``mapdecode.py`` (amax/amin) are outside this path.
The spy's own rows check it flags each forbidden form and passes each allowed one.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.cd.unit._core_support import core, mask, random_logits, random_toy, tables_for

_SUM_REDUCES = {"sum", "mean", "prod"}
_SCATTER_REDUCES = {"add", "multiply"}


def _arg(args: tuple[Any, ...], kwargs: dict[str, Any], i: int, name: str, default: Any) -> Any:
    if len(args) > i:
        return args[i]
    return kwargs.get(name, default)


class FloatScatterAddSpy(TorchDispatchMode):
    """Records ``(op, dtype)`` for every floating-destination scatter-add-like aten call."""

    def __init__(self) -> None:
        super().__init__()
        self.hits: list[str] = []
        self.ops: set[str] = set()

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        name = func.overloadpacket.__name__
        self.ops.add(name)
        dest = args[0] if args and isinstance(args[0], torch.Tensor) else None
        if dest is not None and dest.is_floating_point():
            base = name.rstrip("_")
            bad = False
            if base in ("index_add", "scatter_add"):
                bad = True
            elif base == "scatter_reduce":
                bad = _arg(args, kwargs, 4, "reduce", "") in _SUM_REDUCES
            elif base == "scatter":
                bad = (
                    kwargs.get("reduce", _arg(args, kwargs, 4, "reduce", None)) in _SCATTER_REDUCES
                )
            elif base in ("index_put", "_index_put_impl", "put"):
                bad = bool(_arg(args, kwargs, 3, "accumulate", False))
            if bad:
                self.hits.append(f"{func} on {dest.dtype}")
        return func(*args, **kwargs)


# --------------------------------------------------------------------------- the spy itself (both directions)


def _spy(fn: Any) -> FloatScatterAddSpy:
    s = FloatScatterAddSpy()
    with s:
        fn()
    return s


@pytest.mark.filterwarnings("ignore:The reduce argument of torch.scatter:UserWarning")
def test_spy_flags_every_forbidden_form() -> None:
    f = torch.zeros(2, 4)
    idx = torch.tensor([0, 1, 1, 3])
    src = torch.ones(2, 4)
    forms = {
        "index_add_": lambda: f.clone().index_add_(1, idx, src),
        "index_add": lambda: f.index_add(1, idx, src),
        "scatter_add_": lambda: f.clone().scatter_add_(1, idx.expand(2, 4), src),
        "scatter_reduce_ sum": lambda: f.clone().scatter_reduce_(1, idx.expand(2, 4), src, "sum"),
        "scatter_reduce mean": lambda: f.scatter_reduce(1, idx.expand(2, 4), src, "mean"),
        "scatter_ add": lambda: f.clone().scatter_(1, idx.expand(2, 4), src, reduce="add"),
        "index_put_ accumulate": lambda: f.clone().index_put_(
            (torch.tensor([0, 0]),), torch.ones(2, 4), True
        ),
        "put_ accumulate": lambda: f.clone().put_(
            torch.tensor([0, 0]), torch.ones(2), accumulate=True
        ),
    }
    missed = [k for k, fn in forms.items() if not _spy(fn).hits]
    assert missed == [], f"the spy does not flag: {missed}"


def test_spy_allows_integer_and_order_independent_forms() -> None:
    i32 = torch.zeros(2, 4, dtype=torch.int32)
    f = torch.zeros(2, 4)
    idx = torch.tensor([0, 1, 1, 3])
    forms = {
        "int32 index_add_ (draw.py)": lambda: i32.clone().index_add_(
            1, idx, torch.ones(2, 4, dtype=torch.int32)
        ),
        "int64 index_put_ accumulate": lambda: torch.zeros(4, dtype=torch.int64).index_put_(
            (torch.tensor([0, 0]),), torch.ones(2, dtype=torch.int64), True
        ),
        "float scatter_reduce_ amax": lambda: f.clone().scatter_reduce_(
            1, idx.expand(2, 4), torch.ones(2, 4), "amax"
        ),
        "float scatter_reduce_ amin": lambda: f.clone().scatter_reduce_(
            1, idx.expand(2, 4), torch.ones(2, 4), "amin"
        ),
        "float index_put_ (no accumulate)": lambda: f.clone().index_put_(
            (torch.tensor([0, 1]),), torch.ones(2, 4)
        ),
        "float index_copy_": lambda: f.clone().index_copy_(
            1, torch.tensor([0, 1, 2, 3]), torch.ones(2, 4)
        ),
    }
    flagged = {k: s.hits for k, fn in forms.items() if (s := _spy(fn)).hits}
    assert flagged == {}, f"the spy flags allowed forms: {flagged}"


# --------------------------------------------------------------------------- the path


def _cases() -> list[tuple[str, Any, torch.Tensor, list[int], list[int], int]]:
    """(name, toy, logits, start, end, l_eff): NFA with negated classes and parallel edges (has_outside and not),
    the (a)-shaped instance at a small size, and an ``l_eff < L`` case."""
    out = []
    rng = random.Random(73)
    toy = random_toy(rng, V=64, S=5, nfa=True)
    x = torch.tensor(random_logits(rng, 8, 64, span=8.0))
    out.append(("nfa_v64", toy, x, [0, 1], sorted(toy.finals), 8))
    out.append(("nfa_v64_leff5", toy, x, [0], sorted(toy.finals), 5))
    rng = random.Random(7)
    toy = random_toy(rng, V=512, S=12, nfa=True)
    g = torch.Generator().manual_seed(3)
    x = (30.0 * torch.tanh(torch.randn(32, 512, generator=g) * 40.0 / 30.0)) / 0.4
    out.append(("a_shaped_v512", toy, x, [0], sorted(toy.finals), 32))
    return out


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c[0])
def test_marginals_path_has_no_floating_scatter_add(case: Any) -> None:
    _name, toy, x, start, end, l_eff = case
    em, tr, mg = core("emission"), core("tree"), core("marginals")
    t = tables_for(toy)
    a, b = mask(start, t.bucket), mask(end, t.bucket)
    spy = FloatScatterAddSpy()
    with spy:
        cm = em.class_mass(x, t)
        tree = tr.up_sweep(em.log_transfer(x, t, l_eff))
        em.log_transfer(x, t, l_eff, class_mass=cm)
        mg.constrained_marginals(x, t, tree, a, b, l_eff)
        mg.constrained_marginals(x, t, tree, a, b, l_eff, class_mass=cm)
        coef = torch.rand(x.shape[0], t.n_classes, generator=torch.Generator().manual_seed(5))
        mg._token_marginals(x, t, cm, coef)
    assert spy.ops, "the spy recorded no aten op: it is not intercepting"
    assert spy.hits == [], (
        f"floating scatter-add on the marginals path (non-deterministic on CUDA, T-0073 D4): "
        f"{sorted(set(spy.hits))} [emission from {em.__file__}]"
    )
