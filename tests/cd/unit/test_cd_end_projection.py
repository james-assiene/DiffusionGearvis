"""T-0067 criterion 6: ``end_projection`` = stock trajectory, the constrained MAP committed in place of the argmax.

Expected canvases come from an independent brute-force MAP (``_sampler_support.brute_map``: fp64 enumeration
over the live ids, hand-written DFA replay with the stop augmentation), never from T-0066's code. CL = 4.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tests.cd.fakes.diffusion_sampler import STATE_ROWS, Cfg, bits
from tests.cd.unit._sampler_support import G1, Grammar, argmax_ids, brute_map, cd_xargs
from tests.cd.unit._scenario import Spec, canvas_logits, commit_calls, convergence_calls, run_pair

BIG = 10**6


@pytest.fixture(scope="module")
def g1() -> Grammar:
    return Grammar(G1, "t0067-ep-g1")


def _check_run(
    g: Grammar,
    cfg: Cfg,
    fn,  # type: ignore[no-untyped-def]
    n_calls: int,
    *,
    max_tokens: int = 4096,
    canvases: int = 2,
    cd_ctx=None,  # type: ignore[no-untyped-def]
):  # type: ignore[no-untyped-def]
    CL = cfg.CL
    specs = [Spec("c", xargs=cd_xargs(g, "ep-c"), params={"max_tokens": max_tokens}), Spec("u")]
    a, b = run_pair(cfg, specs, fn, n_calls=n_calls, cd_ctx=cd_ctx)
    conv_a, conv_b = convergence_calls(a, "c"), convergence_calls(b, "c")
    assert conv_a[:canvases] == conv_b[:canvases], "convergence must fire on the stock step"
    commits = commit_calls(a, "c")
    assert len(commits) >= canvases
    states: set = {0}
    emitted = 0
    for k in range(canvases):
        conv, com = conv_a[k], commits[k]
        assert com == conv + 1
        l_eff = min(CL, max_tokens - emitted)
        r_after = max_tokens - emitted - l_eff
        # the logits of decode snapshot i were produced by fn(a[i].call, ...) (the prefill is call 0)
        want, _score, gap = brute_map(g, fn(a[conv].call, "c", CL), states, l_eff, r_after)
        assert gap > 1e-6, "fixture: tie in the brute-force MAP"
        got = a[com].committed("c")
        assert got is not None and got[:l_eff] == want, (
            f"canvas {k}: committed {got} != constrained MAP {want} of the converging step's logits"
        )
        # canvas and argmax_canvas replaced at the converging call; draft_tokens re-synced
        slot = a[conv].slots["c"]
        assert a[conv].state["argmax_canvas"][slot][:l_eff].tolist() == want
        assert a[conv].state["canvas"][slot][:l_eff].tolist() == want, (
            "canvas must hold the MAP (KV write)"
        )
        assert a[conv].draft_tokens[slot, :CL].tolist() == a[conv].state["canvas"][slot].tolist()
        # before convergence the trajectory is stock's
        start = commits[k - 1] + 1 if k else 0
        for i in range(start, conv):
            for name in STATE_ROWS:
                assert torch.equal(bits(a[i].row(name, "c")), bits(b[i].row(name, "c"))), (
                    f"canvas {k}, call {i}: {name} of the constrained slot differs from stock before convergence"
                )
        states = g.run(states, got[:l_eff])
        emitted += l_eff
    return a, b


def test_stability_convergence_commits_the_map(g1: Grammar) -> None:
    fn = canvas_logits({"c": 11, "u": 12}, period=3)
    assert not g1.accepts({0}, argmax_ids(fn(1, "c", 4)), BIG), "fixture: argmax must be outside G1"
    _check_run(g1, Cfg(CL=4, stability_threshold=1), fn, n_calls=9)


def test_argmax_in_language_is_committed_unchanged(g1: Grammar) -> None:
    """Projection identity end to end: a valid argmax is committed as-is and the slot stays bit-stock."""
    CL = 4

    def fn(call: int, req: str, n: int) -> torch.Tensor:
        x = canvas_logits({"u": 12})(call, req, n)
        if req == "c":
            x = torch.full_like(x, -30.0)
            for i, t in enumerate([1000, 1002, 1003, 1004][:n]):
                x[i, 1000:1008] = torch.linspace(0.0, 1.0, 8)
                x[i, t] = 5.0
        return x

    specs = [Spec("c", xargs=cd_xargs(g1, "ep-id")), Spec("u")]
    # 3 decode calls = canvas 0 only (denoise, converge, commit): canvas 1 starts from A_1, where this
    # argmax is no longer in the language
    a, b = run_pair(Cfg(CL=CL, stability_threshold=1), specs, fn, n_calls=3)
    for i, (sa, sb) in enumerate(zip(a, b, strict=True)):
        for name in STATE_ROWS:
            assert torch.equal(bits(sa.row(name, "c")), bits(sb.row(name, "c"))), (i, name)
    assert [s.committed("c") for s in a] == [s.committed("c") for s in b]


def test_step_cap_convergence_commits_the_map_of_that_step(g1: Grammar) -> None:
    cfg = Cfg(CL=4, confidence_threshold=0.0, max_denoising_steps=6)
    fn = canvas_logits({"c": 21, "u": 22}, period=1)  # logits change every call
    conv = 5  # step counter reaches the cap on the 6th denoise call
    m_conv = brute_map(g1, fn(conv, "c", 4), {0}, 4, BIG)[0]
    assert m_conv != brute_map(g1, fn(conv - 1, "c", 4), {0}, 4, BIG)[0], (
        "fixture: MAPs must differ by step"
    )
    _check_run(g1, cfg, fn, n_calls=15)


def test_48_step_cap(g1: Grammar) -> None:
    cfg = Cfg(CL=4, confidence_threshold=0.0, max_denoising_steps=48)
    fn = canvas_logits({"c": 31, "u": 32}, period=1)
    _check_run(g1, cfg, fn, n_calls=50, canvases=1)


def test_state_set_threads_across_canvases(g1: Grammar) -> None:
    """Canvas 1's MAP starts from δ*(A_0, canvas 0), not from A_0."""
    fn = canvas_logits({"c": 41, "u": 42}, period=3)
    a0 = brute_map(g1, fn(1, "c", 4), {0}, 4, BIG)[0]
    a1 = g1.run({0}, a0)
    assert (
        brute_map(g1, fn(4, "c", 4), a1, 4, BIG)[0] != brute_map(g1, fn(4, "c", 4), {0}, 4, BIG)[0]
    ), "fixture: canvas 1's MAP must depend on the threaded state set"
    _check_run(g1, Cfg(CL=4, stability_threshold=1), fn, n_calls=9)


def test_emitted_count_drives_the_budget(g1: Grammar) -> None:
    """max_tokens = CL + 2: canvas 0 must end within 2 tokens of F; canvas 1 has L_eff = 2 and must end in F."""
    fn = canvas_logits({"c": 51, "u": 52}, period=3)
    a0 = brute_map(g1, fn(1, "c", 4), {0}, 4, 2)[0]
    a1 = g1.run({0}, a0)
    tight = brute_map(g1, fn(4, "c", 4), a1, 2, 0)[0]
    loose = brute_map(g1, fn(4, "c", 4), a1, 4, BIG)[0][:2]
    assert tight != loose, "fixture: the budget must change canvas 1's MAP"
    _check_run(g1, Cfg(CL=4, stability_threshold=1), fn, n_calls=9, max_tokens=6)


class _LateEvent:
    """Stands in for a CUDA event that completes ``late`` host queries later (tier-1 lag injection)."""

    def __init__(self, late: int) -> None:
        self.left = late

    def query(self) -> bool:
        if self.left > 0:
            self.left -= 1
            return False
        return True


def _lagging(w: Any) -> Any:
    """After every sampler call, make each newly queued report's event answer "not ready" twice, so ≥ 2
    reports are unread when the budget is decided (the CUDA path; review, 2026-09-30). Reaches the
    implementation's report queue ``_pending`` (items with an ``event``), the only way to model an
    incomplete event on CPU; if that queue is renamed, this row says so instead of passing."""
    import contextlib

    seen: set[int] = set()

    @contextlib.contextmanager
    def ctx() -> Any:
        yield
        q = getattr(w.sampler, "_pending", None)
        assert q is not None, (
            "lag injection needs the wrapper's report queue `_pending` (see docstring)"
        )
        for p in q:
            if id(p) not in seen:
                seen.add(id(p))
                p.event = _LateEvent(2)

    return ctx


def test_budget_with_unread_reports(g1: Grammar) -> None:
    """max_tokens = CL + 2 with every report read ≥ 2 calls late: canvas 1 must still get L_eff = 2, R = 0."""
    fn = canvas_logits({"c": 51, "u": 52}, period=3)
    _check_run(g1, Cfg(CL=4, stability_threshold=1), fn, n_calls=9, max_tokens=6, cd_ctx=_lagging)


def test_truncated_canvas_commits_the_map_with_no_room_after() -> None:
    """valid_canvas_len = 3 < CL = 4 (a canvas truncated near max_model_len): L_eff = 3 and R_after = 0, so the
    committed 3 ids are the MAP that reaches F within them."""
    from tests.cd.fakes.diffusion_sampler import World

    g = Grammar(G1, "t0067-ep-trunc")
    fn = canvas_logits({"c": 61}, period=3)
    w = World(Cfg(CL=4, stability_threshold=1), cd=True)
    w.add("c", xargs=cd_xargs(g, "ep-trunc"))
    w.reqs["c"].valid_len = 3
    w.prefill()
    snaps = [w.decode(fn) for _ in range(3)]
    com = next(i for i, s in enumerate(snaps) if s.committed("c") is not None)
    conv_call = snaps[com - 1].call
    want = brute_map(g, fn(conv_call, "c", 3), {0}, 3, 0)[0]
    loose = brute_map(g, fn(conv_call, "c", 3), {0}, 3, BIG)[0]
    assert want != loose, "fixture: R_after = 0 must change the MAP"
    assert snaps[com].committed("c") == want
