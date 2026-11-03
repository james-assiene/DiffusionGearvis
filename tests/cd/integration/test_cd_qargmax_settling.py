"""T-0076 criterion 6: convergence speed of ``perstep_mar_qargmax`` against ``perstep_mar`` (CPU, vendored stock world).

(a) **Gated.** Language {ac, bc, bd} (a b c d = 1000..1003) then a stop, CL 32. p₁ = (c .51, d .49) fixed;
p₀(a) alternates .55 / .45 between calls (p₀(b) = 1 − p₀(a)). Hand computation:

    p₀(a) = .55: ac .2805, bc .2295, bd .2205 -> MAP ac;  q₀(a) = .2805 / .7305 = .384 -> argmax b
    p₀(a) = .45: ac .2295, bc .2805, bd .2695 -> MAP bc;  q₀(a) = .2295 / .7795 = .294 -> argmax b
    q₁(c) = .51 / (.51 + p₀(b)·.49) = .698 / .654 -> argmax c

so the MAP history flips every step (``perstep_mar`` can only stop at the step cap, 16), while the argmax-of-q
history is ``bc`` every step (``perstep_mar_qargmax`` converges as soon as ST rows agree and it is confident).
Positions 2-31 are fixed and peaked (a stop at 2, id 1007 after), so they never flip. Regime: CPU fp32, torch per
``uv.lock``, vendored v0.29.0 stock step, CL 32, ``max_denoising_steps`` 16, stock ST (``stability_threshold`` 1 in
``tests/fixtures/cd/model/generation_config.json`` -> 2 rows), bound 1.0, confident threshold 0.1 (mean H(q) over CL
≈ 0.04 here, asserted from the fp64 oracle), T = 1 (t_min = t_max = 1), CD generator seed ``steps.GENERATOR_SEED``.

(b) **Reported, not gated** (``slow``). T-0063's Sudoku automaton (first test-cut puzzle) with seeded noisy logits:
``x = 3·N(0,1)[seed] + NOISE·N(0,1)[seed, call]`` (NOISE 0.1) over all of V, n = 200 pre-registered seeds 7600-7799 (override with
``ASTRA_CD_SETTLING_N``), always confident (threshold 1e9, so convergence = a stable history), one canvas per
run; mean / median / p90 denoise steps of the first canvas for both arms,
printed with the regime. The only assertion is that every run committed without an error record.
"""

from __future__ import annotations

import math
import os
import statistics
from typing import Any

import pytest
import torch

from tests.cd import _wire_support as ws
from tests.cd.fakes.diffusion_sampler import Cfg, V, World
from tests.cd.unit import _perstep_support as ps
from tests.cd.unit._sampler_support import Grammar, Records, brute_map

pytestmark = pytest.mark.integration

A, B, C, D = 1000, 1001, 1002, 1003
CL, CAP, ST = 32, 16, 2
THRESHOLD = 0.1
ARM_MODE = {"perstep_mar": "perstep_mar", "perstep_mar_qargmax": "perstep_mar_qargmax"}


def grammar() -> Grammar:
    return Grammar((4, [(0, A, 1), (0, B, 2), (1, C, 3), (2, C, 3), (2, D, 3)], {3}), "t0076-6a")


def rows_6a(pa: float) -> torch.Tensor:
    rows: list[dict[int, float]] = [
        {A: math.log(pa), B: math.log(1 - pa)},
        {C: math.log(0.51), D: math.log(0.49)},
        {1: 5.0},
    ]
    rows += [{1007: 5.0}] * (CL - 3)
    return ps.logits_rows(rows)


def fn_6a(call: int, req: str, n: int) -> torch.Tensor:
    return rows_6a(0.55 if call % 2 else 0.45)[:n].clone()


def cfg_6a() -> Cfg:
    return Cfg(
        CL=CL,
        t_min=1.0,
        t_max=1.0,
        stability_threshold=1,
        max_denoising_steps=CAP,
        entropy_bound=1.0,
        confidence_threshold=THRESHOLD,
        max_num_reqs=2,
    )


def test_6a_premise_by_oracle() -> None:
    g = grammar()
    for pa, want_map in ((0.55, [A, C]), (0.45, [B, C])):
        x = rows_6a(pa)
        tokens, _, gap = brute_map(g, x[:2], [0], 2, 4000)
        assert tokens == want_map and gap > 0.01, (pa, tokens, gap)
        q = ps.brute_q(g, x[:2], [0], 2, 4000)
        assert ps.argmax_q(q, min_gap=0.05) == [B, C], (pa, q)
        hq = [ps.entropy64(_dense(qi)) for qi in q]
        assert sum(hq) / CL < THRESHOLD / 2, (
            f"premise: mean H(q) {sum(hq) / CL} not well below {THRESHOLD}"
        )


def _dense(qi: dict[int, float]) -> Any:
    import numpy as np

    return np.asarray(list(qi.values()), dtype=np.float64)


def _first_canvas(mode: str, tag: str, records: Records) -> dict[str, Any]:
    g = grammar()
    w = World(cfg_6a(), cd=True, seed=0)
    w.add("c", xargs=ws.raw_xargs(g.payload, mode=mode, self_conditioning="model", record_tag=tag))
    w.prefill()
    committed = None
    denoise = 0
    for _ in range(CAP + 4):
        s = w.decode(fn_6a)
        committed = s.committed("c")
        if committed is not None:
            break
        denoise += 1
    w.decode(fn_6a)  # flush the record
    recs = [r for r in records.tagged(tag) if r["canvas_idx"] == 0]
    return {"committed": committed, "denoise": denoise, "records": recs, "grammar": g}


def test_6a_mar_reaches_the_cap_and_qargmax_converges_within_st_plus_one(
    caplog: Any, capsys: Any
) -> None:
    records = Records(caplog, capsys)
    mar = _first_canvas("perstep_mar", "t76-6a-mar", records)
    qa = _first_canvas("perstep_mar_qargmax", "t76-6a-qa", records)
    # perstep_mar: the MAP history flips every call -> only the cap ends the canvas
    assert mar["records"] and mar["records"][0]["error"] is None, mar["records"]
    assert mar["records"][0]["steps"] == CAP, (
        f"perstep_mar steps {mar['records'][0]['steps']} != cap {CAP}"
    )
    # perstep_mar_qargmax: argmax-of-q is bc on every call -> stable after ST rows
    assert qa["records"], "no record for the qargmax request"
    rq = qa["records"][0]
    assert rq["error"] is None, f"T-0076 not implemented or loud: {rq}"
    assert rq["arm"] == "perstep_mar_qargmax", rq
    assert rq["steps"] <= ST + 1, (
        f"perstep_mar_qargmax took {rq['steps']} steps (> ST + 1 = {ST + 1})"
    )
    ids = qa["committed"]
    assert ids is not None and ids[:2] in ([A, C], [B, C]) and ids[2] in (1, 50, 106), ids
    assert qa["grammar"].accepts([0], ids, 4096 - CL), (
        f"committed {ids[:4]} rejected by host replay"
    )


# --------------------------------------------------------------------------- 6b (reported)

#: always confident: over V with noisy logits nothing reaches a small mean-entropy threshold, so 6b measures the
#: settling rule alone (stability of the history), which is what differs between the two arms
THRESHOLD_6B = 1e9
#: per-step noise scale (the step-to-step jitter of a model that has nearly settled)
NOISE_6B = 0.1
SEEDS_6B = range(7600, 7600 + int(os.environ.get("ASTRA_CD_SETTLING_N", "200")))


def _noisy(seed: int) -> Any:
    def fn(call: int, req: str, n: int) -> torch.Tensor:
        base = torch.randn(n, V, generator=torch.Generator().manual_seed(seed)) * 3.0
        noise = torch.randn(n, V, generator=torch.Generator().manual_seed(seed * 1009 + call))
        return base + NOISE_6B * noise

    return fn


def _p90(xs: list[int]) -> float:
    s = sorted(xs)
    return float(s[min(len(s) - 1, math.ceil(0.9 * len(s)) - 1)])


@pytest.mark.slow
def test_6b_sudoku_noisy_logits_steps_per_canvas_report(caplog: Any, capsys: Any) -> None:
    from tests.cd.integration.test_cd_multicanvas_guarantee import oracle

    records = Records(caplog, capsys)
    o = oracle("sudoku")
    cfg = Cfg(
        CL=CL,
        t_min=0.4,
        t_max=0.8,
        stability_threshold=1,
        max_denoising_steps=CAP,
        entropy_bound=1.0,
        confidence_threshold=THRESHOLD_6B,
        max_num_reqs=2,
    )
    steps: dict[str, list[int]] = {"perstep_mar": [], "perstep_mar_qargmax": []}
    for mode in steps:
        for seed in SEEDS_6B:
            tag = f"t76-6b-{mode}-{seed}"
            w = World(cfg, cd=True, seed=seed)
            w.add(
                "c",
                xargs=ws.raw_xargs(o.payload, mode=mode, self_conditioning="model", record_tag=tag),
                max_tokens=512,
            )
            w.prefill()
            fn = _noisy(seed)
            for _ in range(CAP + 2):
                if w.decode(fn).committed("c") is not None:
                    break
            w.decode(fn)
            recs = [r for r in records.tagged(tag) if r["canvas_idx"] == 0]
            assert len(recs) == 1 and recs[0]["error"] is None, f"{mode} seed {seed}: {recs}"
            steps[mode].append(int(recs[0]["steps"]))
    with capsys.disabled():
        print(
            "\n[T-0076 criterion 6b] Sudoku (T-0063, first test-cut puzzle), CPU fp32, vendored v0.29.0 stock "
            f"step, CL {CL}, max_denoising_steps {CAP}, ST {ST}, bound 1.0, threshold {THRESHOLD_6B:g} (always "
            "confident: convergence = stable history), T 0.4-0.8, "
            f"logits 3·N(0,1)[seed] + {NOISE_6B}·N(0,1)[seed,call], logits independent of the fed-back canvas "
            f"(cannot show feedback-driven MAP flipping), n = {len(SEEDS_6B)} seeds "
            f"{SEEDS_6B.start}-{SEEDS_6B.stop - 1}, first canvas, max_tokens 512"
        )
        for mode, xs in steps.items():
            print(
                f"  {mode}: mean {statistics.mean(xs):.2f}  median {statistics.median(xs):.1f}  "
                f"p90 {_p90(xs):.0f}  max {max(xs)}  (n {len(xs)})"
            )
    assert all(len(v) == len(SEEDS_6B) for v in steps.values())
