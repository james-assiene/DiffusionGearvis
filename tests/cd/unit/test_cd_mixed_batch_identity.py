"""T-0067 criterion 4: in a mixed batch the unconstrained slots are bit-identical to the stock run.

Batch {constrained (end_projection), unconstrained, unconstrained}; reference = the vendored stock sampler on
the same batch with the constrained request's ``astra_cd_*`` keys removed; fixed global torch seed. Compared
per call (every call here schedules the same set in both runs): state rows, ``draft_tokens[:, :CL]``, output
rows, and the global RNG state after the call. Covers denoise, converging and commit steps, the step cap, and
top-k / top-p on an unconstrained slot. Non-vacuity: the constrained slot's committed canvas must differ
from the stock one at least once (CD acted), else identity of the others proves nothing.
"""

from __future__ import annotations

import pytest
import torch

from tests.cd.fakes.diffusion_sampler import Cfg, rows_differ
from tests.cd.unit._sampler_support import G1, Grammar, cd_xargs
from tests.cd.unit._scenario import Spec, canvas_logits, run_pair

CASES = {
    "stability": (Cfg(CL=4, stability_threshold=1), {}, {}),
    "cap": (Cfg(CL=4, confidence_threshold=0.0, max_denoising_steps=5), {}, {}),
    # top-k/top-p put -inf in the logits, so stock entropy is NaN and never "confident": these slots
    # converge only at the step cap, which is set low so commit steps are covered within the run
    "topk_topp_on_unconstrained": (
        Cfg(CL=4, stability_threshold=1, max_denoising_steps=5),
        {"top_k": 3},
        {"top_p": 0.7},
    ),
}


@pytest.fixture(scope="module")
def g1() -> Grammar:
    return Grammar(G1, "t0067-mixed-g1")


@pytest.mark.parametrize("case", sorted(CASES))
def test_unconstrained_slots_bit_identical_to_stock(case: str, g1: Grammar) -> None:
    cfg, p1, p2 = CASES[case]
    specs = [
        Spec("c", xargs=cd_xargs(g1, "mixed-c")),
        Spec("u1", params=p1),
        Spec("u2", params=p2),
    ]
    fn = canvas_logits({"c": 11, "u1": 12, "u2": 13}, period=3)
    first = fn(0, "c", cfg.CL).argmax(dim=-1).tolist()
    assert not g1.accepts({0}, first, 10**6), (
        "fixture precondition: canvas 0's argmax must be outside G1"
    )
    a, b = run_pair(cfg, specs, fn, n_calls=14)
    phases: set[str] = set()
    for i, (sa, sb) in enumerate(zip(a, b, strict=True)):
        assert sa.order == sb.order
        for r in ("u1", "u2"):
            bad = rows_differ(sa, sb, r)
            assert bad == [], f"call {i}: unconstrained slot {r} differs from stock in {bad}"
            phases.add("commit" if sa.entered_commit[r] else "denoise")
        assert torch.equal(sa.rng_after, sb.rng_after), (
            f"call {i}: global RNG state differs from stock"
        )
    assert phases == {"commit", "denoise"}, f"fixture: phases covered {phases}"
    cd_commits = [s.committed("c") for s in a if s.committed("c") is not None]
    st_commits = [s.committed("c") for s in b if s.committed("c") is not None]
    assert cd_commits, "the constrained request never committed"
    assert cd_commits != st_commits, (
        "non-vacuity: the constrained slot committed exactly what stock did on every canvas, so the "
        "identity of the other slots shows nothing (the fixture's argmax is outside G1)"
    )
