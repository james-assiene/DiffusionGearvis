"""T-0081 contract: the seed hook recognises prefill completion exactly as stock's ``_finish_prefills`` does.

Ruling SD5: ``astra_vllm_plugin.cd.seed.completing_prefill_slots(input_batch) -> np.ndarray`` returns the slots (batch
order) whose prefill completes in this call. The oracle is the vendored v0.29.0 sampler itself, run on the same batch:
a slot completes iff stock flips its ``is_encoder_phase`` True -> False in that call (decode rows enter in the denoise
phase, so they can only flip False -> True). The table covers the prefill-only path, chunked prefill, mixed decode
calls, decode rows that satisfy the completion inequality without being prefill rows, and a non-identity slot mapping
(slots are returned, not batch indices). The vendored functions' source is pinned by sha256, so a vendor update fails
here loudly and the coupling is re-examined.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

import numpy as np
import pytest

from tests.cd.fakes.diffusion_sampler import Cfg, World, stock
from tests.cd.unit import _seed_support as sd
from tests.cd.unit._sampler_support import LIVE

pytestmark = pytest.mark.contract

SOURCE_SHA = {
    "_finish_prefills": "e1c5797e9cbd327b6546cf92ff3d8696af927310cf285aabdb9109de5f5e9457",
    "_handle_prefill": "58b05527effdec592682e84c9e9aecf369144b56156540ccc6e33317c03a7285",
    "__call__": "01ad2347f3e331454d8b6b46695a4de9c703a9e3fd7eb3b1920b6caa1707eab2",
}
LOGITS = sd.fixed_logits([{t: 1.0 for t in LIVE}])

#: name -> (requests to admit and prefill first (decode rows), rows of the call under test, slot of each request)
TABLE: dict[str, tuple[list[str], list[sd.Row]]] = {
    "prefill_only_all_complete": ([], [sd.Row("a", True), sd.Row("b", True), sd.Row("c", True)]),
    "prefill_only_chunked": (
        [],
        [
            sd.Row("a", True, computed=0, scheduled=2),
            sd.Row("b", True, computed=2, scheduled=3),
            sd.Row("c", True, computed=0, scheduled=6),
        ],
    ),
    "prefill_only_all_mid_chunk": (
        [],
        [sd.Row("a", True, computed=0, scheduled=1), sd.Row("b", True, computed=1, scheduled=3)],
    ),
    "mixed_complete_and_mid_chunk": (
        ["d1", "d2"],
        [sd.Row("d1"), sd.Row("a", True), sd.Row("d2"), sd.Row("b", True, computed=0, scheduled=4)],
    ),
    "decode_rows_only": (["d1", "d2"], [sd.Row("d1"), sd.Row("d2")]),
    "mixed_prefill_rows_first": (
        ["d1"],
        [
            sd.Row("b", True, computed=4, scheduled=1),
            sd.Row("a", True, computed=0, scheduled=1),
            sd.Row("d1"),
        ],
    ),
}
#: slots chosen out of order so a hook returning batch indices fails
SLOTS = {"a": 6, "b": 2, "c": 4, "d1": 5, "d2": 0}


@pytest.mark.parametrize("name", sorted(SOURCE_SHA))
def test_vendored_source_is_the_pinned_one(name: str) -> None:
    src = inspect.getsource(getattr(stock.DiffusionSampler, name))
    got = hashlib.sha256(src.encode()).hexdigest()
    assert got == SOURCE_SHA[name], (
        f"vendored DiffusionSampler.{name} changed ({got}): re-check the seed hook's completion rule against it"
    )


def _oracle(pre: list[str], rows: list[sd.Row]) -> tuple[list[int], Any]:
    w = World(Cfg(CL=4), cd=False, seed=0)
    for r in pre:
        w.add(r, slot=SLOTS[r])
    if pre:
        w.prefill()
    for r in rows:
        if r.req_id not in w.reqs:
            w.add(r.req_id, slot=SLOTS[r.req_id])
    st = w.ms.diffusion_states
    batch, _, _ = sd.batch_for_rows(w, rows)
    before = st.is_encoder_phase.clone()
    for r in pre:
        assert not bool(before[SLOTS[r]]), "fixture: decode rows must enter in the denoise phase"
    sd.call(w, rows, LOGITS)
    after = st.is_encoder_phase
    done = [
        SLOTS[r.req_id]
        for r in rows
        if bool(before[SLOTS[r.req_id]]) and not bool(after[SLOTS[r.req_id]])
    ]
    return done, batch


@pytest.mark.parametrize("name", sorted(TABLE))
def test_completion_set_equals_stock_finish_prefills(name: str) -> None:
    pre, rows = TABLE[name]
    want, batch = _oracle(pre, rows)
    fn = getattr(sd.seed_module(), "completing_prefill_slots", None)
    assert callable(fn), "cd/seed.py has no completing_prefill_slots(input_batch) (ruling SD5)"
    got = [int(x) for x in np.asarray(fn(batch)).reshape(-1)]
    assert got == want, f"{name}: hook says {got}, stock's _finish_prefills completed {want}"


def test_the_table_is_not_degenerate() -> None:
    outcomes = {name: _oracle(*TABLE[name])[0] for name in TABLE}
    assert outcomes["prefill_only_all_complete"] == [6, 2, 4]
    assert outcomes["prefill_only_chunked"] == [2, 4]
    assert outcomes["prefill_only_all_mid_chunk"] == []
    assert outcomes["mixed_complete_and_mid_chunk"] == [6]
    assert outcomes["decode_rows_only"] == []
    assert outcomes["mixed_prefill_rows_first"] == [2]
    sd.seed_module()  # the hook's module must exist for this contract to mean anything
