"""T-0065 criterion 8 (negative, hygiene): no vllm import, no global RNG, no host sync on the
per-step path, allocation cap and retained-tree size at L = 256, bucket 512.

The sync spy is a pair of torch modes rather than a tensor subclass: a ``TorchFunctionMode`` records
the Python-level sync methods (``.item``, ``.tolist``, ``.cpu``, ``.numpy``, ``bool``/``int``/``float``
/``__index__`` on a tensor, ``nonzero``/``argwhere``/``masked_select``/``unique``) and a
``TorchDispatchMode`` records the ATen ops every one of those lowers to (``_local_scalar_dense``,
``nonzero``, boolean-mask ``index``), so a sync reached through any Python spelling is seen.
"""

from __future__ import annotations

import ast
import random

import pytest

from tests.cd.unit._core_support import CD_DIR, CORE_MODULES, core, mask, random_toy, tables_for

MIB = 1 << 20


def _module_source(name: str) -> str:
    path = CD_DIR / f"{name}.py"
    if not path.exists():
        pytest.fail(f"T-0065 not implemented: {path} is missing")
    return path.read_text()


@pytest.mark.parametrize("name", CORE_MODULES)
def test_no_vllm_import(name):
    tree = ast.parse(_module_source(name))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name == "vllm" or a.name.startswith("vllm.")]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "vllm" or node.module.startswith("vllm."):
                bad.append(node.module)
        elif isinstance(node, ast.Call) and getattr(
            node.func, "attr", getattr(node.func, "id", "")
        ) in (
            "import_module",
            "__import__",
        ):
            bad += [
                a.value
                for a in node.args
                if isinstance(a, ast.Constant)
                and isinstance(a.value, str)
                and a.value.split(".")[0] == "vllm"
            ]
    assert not bad, f"{name}.py imports vllm: {bad}"


def _instance(L: int = 16, V: int = 32, S: int = 5, bucket: int = 8):
    import torch

    rng = random.Random(4)
    toy = random_toy(rng, V=V, S=S, nfa=True)
    logits = torch.tensor(
        [[rng.uniform(-30, 0) for _ in range(V)] for _ in range(L)], dtype=torch.float32
    )
    tables = tables_for(toy, bucket=bucket)
    return toy, logits, tables, mask({0, 1}, bucket), mask(sorted(toy.finals), bucket)


def _pipeline(logits, tables, a, b, l_eff):
    em, tr, mg = core("emission"), core("tree"), core("marginals")
    em.class_log_weights(logits, tables)
    logM = em.log_transfer(logits, tables, l_eff)
    t = tr.up_sweep(logM)
    core("semiring").log_matmul(logM[:2], logM[2:4])
    return mg.constrained_marginals(logits, tables, t, a, b, l_eff)


def test_no_global_rng_use():
    import numpy as np
    import torch

    _, logits, tables, a, b = _instance()
    before = (
        torch.random.get_rng_state().clone(),
        np.random.get_state()[1].copy(),
        random.getstate(),
    )
    _pipeline(logits, tables, a, b, 12)
    bud = core("budget")
    bud.terminal_factor(tables, 3)
    bud.delta_star(tables, a, torch.tensor([0, 1, 2], dtype=torch.int64), [3])
    after = (torch.random.get_rng_state(), np.random.get_state()[1], random.getstate())
    assert torch.equal(before[0], after[0]), (
        "torch global RNG consumed (this half is deterministic)"
    )
    assert (before[1] == after[1]).all(), "numpy global RNG consumed"
    assert before[2] == after[2], "python global RNG consumed"


def _sync_spies():
    import torch
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    seen: list[str] = []
    fn_names = {
        "item",
        "tolist",
        "cpu",
        "numpy",
        "__bool__",
        "__int__",
        "__float__",
        "__index__",
        "nonzero",
        "argwhere",
        "masked_select",
        "unique",
        "unique_consecutive",
    }

    class FnSpy(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            name = getattr(func, "__name__", str(func))
            if name in fn_names:
                seen.append(f"function:{name}")
            return func(*args, **(kwargs or {}))

    class AtenSpy(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            if "_local_scalar_dense" in name or name.startswith("aten.nonzero"):
                seen.append(f"aten:{name}")
            if name.startswith("aten.index.Tensor"):
                idx = args[1] if len(args) > 1 else []
                if any(i is not None and i.dtype == torch.bool for i in idx):
                    seen.append("aten:index with a boolean mask")
            return func(*args, **(kwargs or {}))

    return seen, FnSpy, AtenSpy


def test_no_host_sync_on_the_per_step_path(monkeypatch):
    import torch

    _, logits, tables, a, b = _instance()
    seen, FnSpy, AtenSpy = _sync_spies()
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *a, **k: seen.append("torch.cuda.synchronize")
    )
    em, tr, mg, sr = core("emission"), core("tree"), core("marginals"), core("semiring")
    with FnSpy(), AtenSpy():
        em.class_log_weights(logits, tables)
        logM = em.log_transfer(logits, tables, 12)
        sr.log_matmul(logM[:2], logM[2:4])
        t = tr.up_sweep(logM)
        res = mg.constrained_marginals(logits, tables, t, a, b, 12)
    assert not seen, f"synchronizing calls on the per-step path: {sorted(set(seen))}"
    # the spy is live: the same mode sees a sync when one happens (guards a vacuous pass)
    with FnSpy(), AtenSpy():
        res.log_z.item()
    assert seen, "spy failed to see .item()"


@pytest.mark.slow
def test_allocation_cap_and_retained_tree_size_at_bucket_512():
    """L = 256, bucket 512 (300 live states), V = 4096: no single op output > 256 MiB, so the
    [L/2, S, S, S] broadcast (64 GiB here) must be chunked; retained levels ≤ (2L−1)·S²·4 B."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    L, V, bucket = 256, 4096, 512
    rng = random.Random(9)
    toy = random_toy(rng, V=V, S=300, nfa=False)
    logits = torch.randn(L, V, generator=torch.Generator().manual_seed(9)) * 10
    tables = tables_for(toy, bucket=bucket)
    a, b = mask({0}, bucket), mask(sorted(toy.finals), bucket)
    worst: list[tuple[int, str]] = [(0, "")]

    class AllocSpy(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            outs = out if isinstance(out, (tuple, list)) else [out]
            for o in outs:
                if isinstance(o, torch.Tensor):
                    n = o.numel() * o.element_size()
                    if n > worst[0][0]:
                        worst[0] = (n, str(func))
            return out

    em, tr, mg = core("emission"), core("tree"), core("marginals")
    with AllocSpy():
        logM = em.log_transfer(logits, tables, L)
        t = tr.up_sweep(logM)
        mg.constrained_marginals(logits, tables, t, a, b, L)
    n, op = worst[0]
    assert n <= 256 * MIB, f"largest single allocation {n / MIB:.0f} MiB from {op}"
    retained = sum(lv.numel() * lv.element_size() for lv in t.levels)
    assert retained <= (2 * L - 1) * bucket * bucket * 4, (
        f"retained tree levels {retained / MIB:.0f} MiB"
    )
    assert tuple(t.levels[0].shape) == (L, bucket, bucket)
    assert tuple(t.levels[-1].shape) == (1, bucket, bucket)
