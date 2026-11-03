"""T-0068 criterion 2: Mar is not Mf, in both directions (tests.md: a guard needs a row for each side).

Forward: where the grammar forces the token (H(p) high, H(q) = 0), Mar accepts and Mf does not.
Reverse: when every per-position support is the full vocabulary, q = p exactly, so the two accept masks are
identical (an implementation that, say, treated every constrained position as forced, or took H(q) of a
wrongly-normalised q, would split them).

Accept masks are read from the fed-back canvas: an accepted position holds the joint draw, which sits in the
position's (small) high-mass set; a rejected one holds a uniform renoise id outside the live and stop ids.
CPU stand-in, T = 1, CL = 4, V = 262 144, bound 0.5 (the inner sampler's), threshold never reached.
"""

from __future__ import annotations

import math

import pytest

from tests.cd.unit import _perstep_support as ps
from tests.cd.unit.test_cd_arms_semantics import canvas, const, run_cd, t1

FORCED = [1000, 1001, 1002]
#: position 0 free over the 8 live ids (peaked on 1005: low H(p), low H(q)); positions 1-3 forced to
#: 1000/1001/1002 under logits flat over 8 / 6 / 5 live ids
FWD_SETS = [list(range(1000, 1008)), [FORCED[0]], [FORCED[1]], [FORCED[2]]]
FWD_ROWS = [
    {**ps.flat(range(1000, 1008)), 1005: 12.0},
    ps.flat(range(1000, 1008)),
    ps.flat(range(1000, 1006)),
    ps.flat(range(1000, 1005)),
]


def _accept_mask(snap_canvas: list[int]) -> list[bool]:
    return [not ps.renoised(t) for t in snap_canvas]


def test_forward_premise() -> None:
    x = ps.logits_rows(FWD_ROWS)
    hp = [ps.entropy64(ps.p64(x[i])) for i in range(4)]
    hq = [ps.entropy64(ps.q_product64(x[i], FWD_SETS[i])) for i in range(4)]
    assert hq[1:] == [0.0, 0.0, 0.0] and hq[0] < 1e-3
    assert min(hp[1:]) > math.log(5) - 1e-3
    assert ps.stock_rule(hq, 0.5) == [True] * 4 and ps.margin(hq, 0.5) > 0.05
    # Mf: position 0 and the lowest-H(p) forced position (3: ln 5) are accepted; 1 and 2 are not
    assert ps.stock_rule(hp, 0.5) == [True, False, False, True] and ps.margin(hp, 0.5) > 0.05


@pytest.mark.parametrize("arm", ["mar", "mf"])
def test_forced_positions_mar_accepts_mf_does_not(arm: str) -> None:
    g = ps.product_grammar(FWD_SETS, "t0068-fwd")
    x = ps.logits_rows(FWD_ROWS)
    xargs = ps.ps_xargs(g, f"fwd-{arm}", arm, confidence_threshold=ps.NEVER_CONFIDENT)
    _, snaps = run_cd(t1(entropy_bound=0.5), xargs, const(x), 3)
    for k, s in enumerate(snaps):
        c = canvas(s)
        if arm == "mar":
            assert c[1:] == FORCED, (
                f"mar call {k}: forced positions (H(q) = 0) must be accepted: {c}"
            )
        else:
            assert ps.renoised(c[1]) and ps.renoised(c[2]), (
                f"mf call {k}: positions 1, 2 have H(p) = ln 8, ln 6 and must be rejected: {c}"
            )
            assert c[3] == FORCED[2], f"mf call {k}: position 3 is accepted and forced: {c}"
        assert c[0] in FWD_SETS[0]


#: full-vocabulary support: positions 0 and 3 peaked (low H), 1 flat over 8 live ids, 2 flat over 4
REV_ROWS = [
    {**ps.flat(range(1000, 1008)), 1003: 12.0},
    ps.flat(range(1000, 1008)),
    ps.flat(range(1000, 1004)),
    {**ps.flat(range(1000, 1008)), 1006: 12.0},
]
REV_MASK = [True, False, True, True]


def test_reverse_premise() -> None:
    x = ps.logits_rows(REV_ROWS)
    hp = [ps.entropy64(ps.p64(x[i])) for i in range(4)]
    assert ps.stock_rule(hp, 0.5) == REV_MASK and ps.margin(hp, 0.5) > 0.05


@pytest.fixture(scope="module")
def full_v() -> ps.Grammar:
    return ps.full_vocab_grammar("t0068-fullv")


@pytest.mark.parametrize("arm", ["mar", "mf", "mar_sc"])
def test_full_vocabulary_support_gives_identical_masks(arm: str, full_v: ps.Grammar) -> None:
    """q = p, so H(q) = H(p): both arms accept {0, 2, 3} and reject 1 on every call. Accepted flat position 2
    holds one of its 4 high-mass ids; position 1 (rejected) a uniform id."""
    x = ps.logits_rows(REV_ROWS)
    xargs = ps.ps_xargs(full_v, f"rev-{arm}", arm, confidence_threshold=ps.NEVER_CONFIDENT)
    _, snaps = run_cd(t1(entropy_bound=0.5), xargs, const(x), 4)
    for k, s in enumerate(snaps):
        c = canvas(s)
        assert _accept_mask(c) == REV_MASK, (
            f"{arm} call {k}: accept mask {_accept_mask(c)} != {REV_MASK}: {c}"
        )
        assert c[0] == 1003 and c[3] == 1006 and c[2] in range(1000, 1004), c


def test_masks_equal_across_arms_on_full_support(full_v: ps.Grammar) -> None:
    """The same row, arm against arm in one batch: an mf and a mar slot side by side on the same logits."""
    from tests.cd.fakes.diffusion_sampler import World

    x = ps.logits_rows(REV_ROWS)
    w = World(t1(entropy_bound=0.5), cd=True, seed=3)
    for arm in ("mf", "mar"):
        w.add(
            arm,
            xargs=ps.ps_xargs(full_v, f"pair-{arm}", arm, confidence_threshold=ps.NEVER_CONFIDENT),
        )
    w.prefill()
    for k in range(4):
        s = w.decode(lambda call, req, n: x[:n].clone())
        masks = {r: _accept_mask([int(t) for t in s.row("canvas", r)]) for r in ("mf", "mar")}
        assert masks["mf"] == masks["mar"] == REV_MASK, f"call {k}: {masks}"
