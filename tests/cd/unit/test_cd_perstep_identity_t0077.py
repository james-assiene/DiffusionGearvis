"""T-0077 N5 (T-0077a criterion 5): on the CPU stock world, every arm's outputs and records equal the frozen path's.

Each (arm, scenario) runs twice through the T-0067 CPU stand-in (vendored v0.29.0 stock step) with
``_golden_support.trace``: once on the live plugin, once with every call the not-owned callers (``steps``,
``sampler``, ``requests``) make into the T-0077a-owned modules routed to the frozen e73cfa7 copies
(``_frozen_t0077.frozen_world``). Compared per call, bit for bit: ``canvas``, history rows and length,
``argmax_canvas``, ``sc_embeds``, phase, ``confident``, ``step``, ``draft_tokens[:CL]``, committed ids, the device
report bytes, the CD generator's state, the global RNG state; and every ``ASTRA_CD`` record line. The op digest is
**not** compared (live states legitimately change the op sequence; T-0077a criterion 6 reports the golden module).

Arms: ``end_projection``, ``perstep_mf``, ``perstep_mar``, ``perstep_mar`` + constrained SC, ``perstep_mar_qargmax``
and its SC twin; scenarios ``budget`` (4th canvas ``L_eff`` 5 < CL 8) and ``pad`` (valid length 5 < CL 8). The
unconstrained companion is compared with a stock-only run of the same batch (``_scenario.run_pair``).

Per N5 a difference fails the row; the failure names the first differing value and its |Δ| for the orchestrator's
ruling (within N4 or not). The row is never edited to pass.

Regime: CPU fp32, ``max_denoising_steps`` 6, ST 1, T 0.4-0.8, bound 1.0, threshold 1e9, seeds 11 / 23 (the T-0076
golden scenarios). Run serially.
"""

from __future__ import annotations

import functools
import inspect
import sys
from typing import Any

import pytest
import torch

from tests.cd.fakes.diffusion_sampler import bits, rows_differ
from tests.cd.unit import _frozen_t0077 as fz
from tests.cd.unit import _golden_support as gs
from tests.cd.unit._sampler_support import Grammar
from tests.cd.unit._scenario import Spec, canvas_logits, run_pair

#: every served arm as (mode, self_conditioning)
N5_ARMS: dict[str, tuple[str, str]] = {
    "end_projection": ("end_projection", "model"),
    "perstep_mf": ("perstep_mf", "model"),
    "perstep_mar": ("perstep_mar", "model"),
    "perstep_mar_sc": ("perstep_mar", "constrained"),
    "perstep_mar_qargmax": ("perstep_mar_qargmax", "model"),
    "perstep_mar_qargmax_sc": ("perstep_mar_qargmax", "constrained"),
}
KEYS = [f"{arm}/{scen}" for arm in N5_ARMS for scen in gs.SCENARIOS]
OWNED = {f"astra_vllm_plugin.cd.{m}" for m in fz.FROZEN_MODULES}


@functools.cache
def traced(key: str, path: str) -> dict[str, Any]:
    arm, scen = key.split("/")
    mode, sc = N5_ARMS[arm]
    if path == "frozen":
        with fz.frozen_world():
            return gs.trace(arm, scen, mode=mode, sc=sc)
    return gs.trace(arm, scen, mode=mode, sc=sc)


def _value_diff(name: str, g: torch.Tensor, w: torch.Tensor) -> str:
    if g.shape != w.shape or g.dtype != w.dtype:
        return f"{name}: shape/dtype {tuple(g.shape)}/{g.dtype} vs {tuple(w.shape)}/{w.dtype}"
    ne = (bits(g).reshape(-1) != bits(w).reshape(-1)).nonzero().reshape(-1)
    j = int(ne[0])
    gv, wv = g.reshape(-1)[j], w.reshape(-1)[j]
    d = (gv.double() - wv.double()).abs().item() if g.is_floating_point() else "n/a"
    return f"{name}: {ne.numel()} entries differ; first flat {j}: live {gv.item()} frozen {wv.item()} |Δ| {d}"


def test_frozen_world_routes_every_core_call() -> None:
    """Non-vacuity of the frozen path: every function of a T-0077a-owned module bound in a not-owned ``cd`` module
    is in ``CALL_SITES``; inside ``frozen_world`` each binding is the frozen function, and it is restored on exit."""
    gs_mods = [fz.live(m) for m in ("steps", "sampler", "requests", "budget")]
    missing = []
    for mod in gs_mods:
        for attr, val in vars(mod).items():
            if inspect.isfunction(val) and getattr(val, "__module__", "") in OWNED:
                short = mod.__name__.rsplit(".", 1)[-1]
                if not any(s == short and a == attr for s, a, _, _ in fz.CALL_SITES):
                    missing.append(f"{mod.__name__}.{attr}")
    assert not missing, f"core functions the frozen world does not route: {missing}"
    before = {(m, a): getattr(fz.live(m), a) for m, a, _, _ in fz.CALL_SITES}
    with fz.frozen_world():
        for m, a, fm, fa in fz.CALL_SITES:
            assert getattr(fz.live(m), a) is getattr(fz.frozen(fm), fa), f"{m}.{a} not routed"
    for (m, a), val in before.items():
        assert getattr(fz.live(m), a) is val, f"{m}.{a} not restored"
    assert all(not fz.live(m).__name__.startswith(fz.FROZEN_PKG) for m in ("steps", "sampler"))
    assert sys.modules["astra_vllm_plugin.cd.steps"] is fz.live("steps")


@pytest.mark.parametrize("key", KEYS)
def test_n5_records_equal_frozen(key: str) -> None:
    """Every ``ASTRA_CD`` record line of the live run equals the frozen run's, in order."""
    live, froz = traced(key, "live"), traced(key, "frozen")
    assert froz["records"], f"{key}: the frozen run wrote no record (scenario did not commit)"
    if live["records"] != froz["records"]:
        diffs = [
            f"line {i}:\n  live   {a}\n  frozen {b}"
            for i, (a, b) in enumerate(zip(live["records"], froz["records"], strict=False))
            if a != b
        ]
        pytest.fail(
            f"{key}: record lines differ ({len(live['records'])} live, {len(froz['records'])} frozen)\n"
            + "\n".join(diffs[:3])
        )


@pytest.mark.parametrize("arm", [a for a in N5_ARMS if a != "end_projection"])
def test_served_pass1_tree_runs_on_live_states(arm: str) -> None:
    """Ruling L5/L7: on the served path (``steps.prepare`` inside the stock world), the tree ``steps.up_sweep`` returns
    ran on the request's live states: ``tree.n_states == tables.n_states < bucket`` for every call, where ``tables`` is
    what ``steps.log_transfer`` received just before. Kills: the live-states tag lost between ``log_transfer`` and
    ``up_sweep`` on the served path (``log_transfer`` not tagging, a copy/``clone``/``contiguous`` in between, or
    ``up_sweep`` ignoring the tag) — the tree silently falls back to the bucket and N5 still holds."""
    steps = fz.live("steps")
    seen: list[tuple[int, int, Any]] = []
    pending: list[Any] = []
    orig_lt, orig_up = steps.log_transfer, steps.up_sweep

    def lt(logits: Any, tables: Any, *a: Any, **kw: Any) -> Any:
        pending.append(tables)
        return orig_lt(logits, tables, *a, **kw)

    def up(log_m: Any, *a: Any, **kw: Any) -> Any:
        tree = orig_up(log_m, *a, **kw)
        tables = pending.pop()
        seen.append((int(tables.n_states), int(tables.bucket), getattr(tree, "n_states", None)))
        return tree

    steps.log_transfer, steps.up_sweep = lt, up
    try:
        mode, sc = N5_ARMS[arm]
        gs.trace(arm, "budget", mode=mode, sc=sc)
    finally:
        steps.log_transfer, steps.up_sweep = orig_lt, orig_up
    assert seen, f"{arm}: steps.up_sweep was never called on the served path"
    assert all(s < b for s, b, _ in seen), (
        f"{arm}: premise: the stock-world automaton fills its bucket {seen[:2]}"
    )
    bad = [(s, b, n) for s, b, n in seen if n != s]
    assert not bad, (
        f"{arm}: {len(bad)} of {len(seen)} served pass-1 trees not on the live states "
        f"(n_states, bucket, tree.n_states) e.g. {bad[:3]}"
    )


@pytest.mark.parametrize("key", KEYS)
def test_n5_outputs_equal_frozen(key: str) -> None:
    """Every traced tensor of every call (op digest excluded) is bit-identical to the frozen run's."""
    live, froz = traced(key, "live"), traced(key, "frozen")
    assert len(live["calls"]) == len(froz["calls"]), (
        f"{key}: {len(live['calls'])} calls live, {len(froz['calls'])} frozen"
    )
    bad: list[str] = []
    for i, (g, w) in enumerate(zip(live["calls"], froz["calls"], strict=True)):
        for name, wv in w.items():
            if name == "ops":
                continue
            gv = g[name]
            if gv.shape != wv.shape or gv.dtype != wv.dtype or not torch.equal(bits(gv), bits(wv)):
                bad.append(f"call {i}: " + _value_diff(name, gv, wv))
    assert not bad, f"{key}: not bit-identical to the frozen path:\n" + "\n".join(bad[:12])


@pytest.mark.parametrize("arm", list(N5_ARMS))
def test_n5_unconstrained_companion_bit_identical_to_stock(arm: str) -> None:
    """The unconstrained slot beside each arm equals a stock-only run of the same batch, call by call (state rows,
    draft tokens, output rows, global RNG)."""
    from tests.cd import _wire_support as ws

    mode, sc = N5_ARMS[arm]
    g = Grammar(gs.GRAMMAR_SPEC, f"t0077a-n5-{arm}")
    xargs = ws.raw_xargs(g.payload, mode=mode, self_conditioning=sc, record_tag=f"t0077a-n5-{arm}")
    specs = [Spec("c", xargs=xargs), Spec("u")]
    fn = canvas_logits({"c": 31, "u": 32}, period=3)
    a, b = run_pair(gs._cfg(8), specs, fn, n_calls=14)
    for i, (sa, sb) in enumerate(zip(a, b, strict=True)):
        bad = rows_differ(sa, sb, "u")
        assert bad == [], f"{arm}: call {i}: unconstrained slot differs from stock in {bad}"
        assert torch.equal(sa.rng_after, sb.rng_after), f"{arm}: call {i}: global RNG differs"
