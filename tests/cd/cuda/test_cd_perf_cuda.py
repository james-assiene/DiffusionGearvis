"""T-0072 criterion 7 (tier 2): criterion 1's rows on CUDA, and the stop-id cache without host-to-device copies.

Runs **inside the pinned container** (no engine boot), driven by ``tests/contract/test_vllm_cd_perf_gpu.py``,
with ``torch.set_float32_matmul_precision("high")`` set globally for the module (the scan may use a float
matmul; TF32 must not change a bit of a 0/1 product). The T-0063 cut automata cannot be compiled inside the
image, so the host exports their arrays (``ASTRA_CD_CUT_NPZ``: one ``.npz`` per automaton, written by the
contract test from T-0063's compile cache) and this module rebuilds tables on CUDA from them.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import tests.cd.unit.test_delta_star_scan as S
from tests.cd.unit._core_support import core
from tests.cd.unit._perf_support import STOPS, identical, scan, seq, walk_ids

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda")


@pytest.fixture(autouse=True, scope="module")
def tf32_high() -> Iterator[None]:
    if not torch.cuda.is_available():
        why = "CUDA rows run inside the pinned container (tests/contract/test_vllm_cd_perf_gpu.py)"
        if os.environ.get("ASTRA_CD_REQUIRE_CUDA") == "1":
            pytest.fail(f"ASTRA_CD_REQUIRE_CUDA=1 but CUDA is not available: {why}")
        pytest.skip(why)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    print(f"ASTRA_TIER2_MATMUL_PRECISION {torch.get_float32_matmul_precision()}")
    yield
    torch.set_float32_matmul_precision(prev)


def _toy_tables(toy: Any, bucket: int | None) -> Any:
    return core("emission").build_tables(
        n_states=toy.n_states,
        vocab_size=toy.vocab_size,
        class_members=[sorted(s) for s, _ in toy.classes],
        class_is_neg=[n for _, n in toy.classes],
        edge_src=[e[0] for e in toy.edges],
        edge_dst=[e[1] for e in toy.edges],
        edge_class=[e[2] for e in toy.edges],
        finals=sorted(toy.finals),
        bucket=bucket,
        device=DEV,
    )


def _mask(idx: Any, n: int) -> torch.Tensor:
    m = torch.zeros(n, dtype=torch.bool, device=DEV)
    for i in idx:
        m[i] = True
    return m


def _check(tables: Any, a: torch.Tensor, ids: list[int], stops: Any, what: str) -> Any:
    t = torch.tensor(ids, dtype=torch.int64, device=DEV)
    want = seq(tables, a, t, stops)
    got = scan(tables, a.clone(), t, stops)
    bad = identical(got, want)
    assert not bad, f"{what} (cuda): {bad}"
    assert got.states.device.type == "cuda"
    return got


@pytest.mark.parametrize("bucket", [None, 64, 128, 512])
@pytest.mark.parametrize("L", S.LS)
def test_hand_built_nfas_on_cuda(L: int, bucket: int | None) -> None:
    for name, toy in S._toys():
        tables = _toy_tables(toy, bucket)
        for start in S._starts(toy):
            for kind, ids in S._canvases(toy, L, seed=L * 7 + len(name)).items():
                got = _check(
                    tables, _mask(start, tables.bucket), ids, S.TOY_STOPS, f"{name} {kind} L={L}"
                )
                want = S._expected_delta(toy, start, ids, S.TOY_STOPS)
                assert {i for i, on in enumerate(got.states.cpu().tolist()) if on} == want


@pytest.mark.parametrize("L", [63, 64, 65, 255, 256])
def test_odd_carry_and_order_on_cuda(L: int) -> None:
    tables = _toy_tables(S.MARKER, 512)
    for p in range(L):
        ids = [2] * p + [3] + [4] * (L - p - 1)
        got = _check(tables, _mask({0}, 512), ids, STOPS, f"marker L={L} p={p}")
        assert bool(got.states[1]) and not bool(got.states[0])
    tables = _toy_tables(S.CHAIN, 128)
    ids = [5] * L
    ids[0], ids[L - 1] = 3, 4
    got = _check(tables, _mask({0}, 128), ids, STOPS, f"chain L={L}")
    assert bool(got.states[2])


def _npz_dir() -> Path:
    d = os.environ.get("ASTRA_CD_CUT_NPZ")
    if not d:
        pytest.fail(
            "ASTRA_CD_CUT_NPZ is not set: run through tests/contract/test_vllm_cd_perf_gpu.py"
        )
    return Path(d)


def _load(path: Path) -> Any:
    z = np.load(path)
    return SimpleNamespace(
        n_states=int(z["n_states"]),
        n_states_bucket=int(z["n_states_bucket"]),
        vocab_size=int(z["vocab_size"]),
        edge_src=z["edge_src"],
        edge_dst=z["edge_dst"],
        edge_class=z["edge_class"],
        is_final=z["is_final"],
        start_vector=z["start_vector"],
        tables=SimpleNamespace(
            sum_indptr=z["sum_indptr"], sum_indices=z["sum_indices"], sum_is_neg=z["sum_is_neg"]
        ),
    )


def test_every_cut_automaton_on_cuda() -> None:
    from tests.cd.unit._perf_support import start_of, tables_from_compiled

    files = sorted(_npz_dir().glob("*.npz"))
    # exactly the automata the host exported: the same source (S._cut_ids, i.e. every T-0063 cut minus the 17
    # OVER_LADDER ones with bucket > 512) as tests/contract/test_vllm_cd_perf_gpu.py's CUT_AUTOMATA (793)
    want = {f"{kind}__{cut}__{rid}.npz" for kind, cut, rid in S._cut_ids()}
    got = {f.name for f in files}
    assert got == want, (
        f"cut automata: got {len(got)}, expected {len(want)}; missing {sorted(want - got)[:5]}, "
        f"unexpected {sorted(got - want)[:5]}"
    )
    for f in files:
        a = _load(f)
        seed = sum(map(ord, f.stem))
        for b in [b for b in S.BUCKETS if b >= a.n_states_bucket]:
            tables = tables_from_compiled(a, b, device="cuda")
            a0 = start_of(a, b).to(DEV)
            for L in S.LS:
                _check(tables, a0, walk_ids(a, L, seed + L), STOPS, f"{f.stem} B={b} L={L}")


class _H2D(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.hits: list[str] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        name = str(func)
        if "lift_fresh" in name:
            self.hits.append(name)
        if (
            ("_to_copy" in name or "copy_" in name)
            and isinstance(out, torch.Tensor)
            and out.device.type == "cuda"
        ):
            srcs = [x for x in args if isinstance(x, torch.Tensor)]
            src = srcs[-1] if "copy_" in name else srcs[0]
            if src.device.type == "cpu":
                self.hits.append(f"{name} (host -> device)")
        return out


def test_stop_id_cache_no_h2d_after_first_call_on_cuda() -> None:
    tables = _toy_tables(S.NFA_OVERLAP, 64)
    a0 = _mask({0}, 64)
    ids = torch.tensor([2, 9, 9, 1, 5], dtype=torch.int64, device=DEV)
    scan(tables, a0, ids, STOPS)
    spy = _H2D()
    with spy:
        scan(tables, a0, ids, STOPS)
    assert not spy.hits, f"δ* copied host data to the device after the first call: {spy.hits}"
    live = _H2D()
    with live:
        torch.as_tensor([1, 2], device=DEV)
    assert live.hits, "liveness: the H2D spy saw nothing"
