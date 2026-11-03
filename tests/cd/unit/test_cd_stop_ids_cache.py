"""T-0072 criterion 3: the stop-id tensor is cached per (device, stop-id tuple).

After the first ``delta_star`` call on a (device, stop-id tuple), a call builds no tensor from host data: no
``torch.tensor`` / ``as_tensor`` / ``asarray`` / ``from_numpy`` / ``new_tensor`` of Python data (a
``TorchFunctionMode`` sees the Python spelling, a ``TorchDispatchMode`` sees ``aten.lift_fresh*``, which every
one of them lowers to) and no host-to-device copy (CUDA only: tier 2, ``tests/contract/test_vllm_cd_perf_gpu.py``).
A different stop-id tuple on the same device gets its own tensor, never a stale one: ``(1,)`` then
``(1, 50, 106)`` and the reverse, on ids where the two tuples give different δ* (checked against the frozen
sequential δ* and the enumeration oracle).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from tests.cd.oracle.enumerate import Toy, advance
from tests.cd.unit._core_support import mask, tables_for
from tests.cd.unit._perf_support import identical, scan, seq

HOST_BUILDERS = {"tensor", "as_tensor", "asarray", "from_numpy", "frombuffer", "new_tensor"}


class _HostData:
    """Records every tensor built from host (Python / numpy) data while active."""

    def __init__(self) -> None:
        self.hits: list[str] = []
        outer = self

        class Fn(TorchFunctionMode):
            def __torch_function__(
                self, func: Any, types: Any, args: Any = (), kwargs: Any = None
            ) -> Any:
                if getattr(func, "__name__", "") in HOST_BUILDERS:
                    outer.hits.append(f"function:{func.__name__}")
                return func(*args, **(kwargs or {}))

        class Aten(TorchDispatchMode):
            def __torch_dispatch__(
                self, func: Any, types: Any, args: Any = (), kwargs: Any = None
            ) -> Any:
                if "lift_fresh" in str(func):
                    outer.hits.append(f"aten:{func}")
                return func(*args, **(kwargs or {}))

        self._fn, self._aten = Fn(), Aten()

    def __enter__(self) -> _HostData:
        self._fn.__enter__()
        self._aten.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._aten.__exit__(*exc)
        self._fn.__exit__(*exc)


#: V = 128: 0 loops on {2, 50}; 5 kills. With stops (1,) the 50 is consumed and the 5 after it kills the set;
#: with stops (1, 50, 106) consumption ends at the 50 and the set survives.
TOY = Toy(
    2,
    128,
    ((frozenset({2, 50}), False), (frozenset({3}), False)),
    ((0, 0, 0), (0, 1, 1)),
    frozenset({1}),
)
IDS = [2, 2, 50, 5, 2]


def _ids() -> torch.Tensor:
    return torch.tensor(IDS, dtype=torch.int64)


def test_spy_is_live() -> None:
    with _HostData() as spy:
        torch.tensor([1, 50, 106])
    assert spy.hits, "the host-data spy saw nothing"
    with _HostData() as spy:
        torch.zeros(3, dtype=torch.int64)
        torch.arange(4)
    assert not spy.hits, f"the spy must not count factories without host data: {spy.hits}"


def test_second_call_builds_no_tensor_from_host_data() -> None:
    tables = tables_for(TOY, bucket=8)
    a0, ids = mask({0}, 8), _ids()
    scan(tables, a0, ids, (1, 50, 106))  # first call on (cpu, (1, 50, 106)): may build
    for _ in range(2):
        with _HostData() as spy:
            scan(tables, a0, ids, (1, 50, 106))
        assert not spy.hits, f"δ* built tensors from host data after the first call: {spy.hits}"


def test_second_call_with_an_equal_list_reuses_the_cache() -> None:
    """The key is the tuple of ids, so a list equal to a cached tuple is a hit (the sampler passes a tuple;
    T-0065 R1 types ``stop_ids`` as ``Sequence[int]``)."""
    tables = tables_for(TOY, bucket=8)
    a0, ids = mask({0}, 8), _ids()
    scan(tables, a0, ids, (1, 106))
    with _HostData() as spy:
        scan(tables, a0, ids, [1, 106])
    assert not spy.hits, spy.hits


def _results(order: list[tuple[int, ...]]) -> list[Any]:
    tables = tables_for(TOY, bucket=8)
    out = []
    for stops in order:
        got = scan(tables, mask({0}, 8), _ids(), stops)
        want = seq(tables, mask({0}, 8), _ids(), stops)
        assert not identical(got, want), (stops, identical(got, want))
        out.append(got)
    return out


def test_fixture_discriminates_the_two_tuples() -> None:
    assert advance(TOY, {0}, IDS[:3]) == {0}
    assert advance(TOY, {0}, IDS) == set()


def test_a_different_tuple_is_never_served_a_stale_tensor() -> None:
    a, b, a2 = _results([(1,), (1, 50, 106), (1,)])
    assert bool(a.empty) and not bool(b.empty) and bool(a2.empty)
    assert {i for i, on in enumerate(b.states.tolist()) if on} == {0}


def test_reverse_order_is_never_stale_either() -> None:
    b, a, b2 = _results([(1, 50, 106), (1,), (1, 50, 106)])
    assert not bool(b.empty) and bool(a.empty) and not bool(b2.empty)


def test_empty_stop_tuple_after_a_cached_one() -> None:
    """No stop ids: every id is consumed (T-0065 R1); a cache must not substitute the previous tuple."""
    _results([(1, 50, 106), ()])
