"""T-0081 criterion 3: the CD guarantee holds for seeded requests (CPU, integration).

1. **Out-of-language draft (adversarial).** A product language ``OPEN · {A, B} · CLOSE · stop`` (ids 1003, {1000, 1001},
   1004) and drafts that violate it -- an out-of-enum value (1002 where A/B is required) and a stop id inside the
   "string" (EOS at the value position) -- seeded at noise 0 under a *copy model* (the logits echo the input canvas
   confidently, as a model that agrees with its own draft would). Every arm must commit a canvas the host replay
   accepts (CS = 1), never the draft, with no loud error.
2. **In-language draft.** The same language, draft ``1003 1001 1004 EOS EOS EOS EOS EOS``: with the copy model it
   commits after <= 2 denoise steps and equals the draft (``end_projection``, ``perstep_mar``, ``perstep_mar_qargmax``).
   Non-vacuity: the same request **unseeded** needs more than 2 denoise steps (the copy model is unsure on noise).
3. **Matrix.** T-0068 c3's multi-canvas guarantee (host replay viability after every commit, acceptance, zero loud
   errors, termination by ``max_tokens``) with seeded requests at noise {0, 0.5} x start step {0, mid = 1 of 3}, over
   the six arms; the automaton kind (BFCL ``live_simple_2-2-0`` JSON whitespace, Sudoku test 0, two synthetic NFA
   families) and ``max_tokens`` rotate per run. Drafts alternate between the unconstrained argmax of the run's first
   logits (the model's own draft; out of language almost surely) and a random in-language walk of the automaton.
   ``ASTRA_CD_GUARANTEE_RUNS`` sets the run count (routine 48 = 2 per cell), ``ASTRA_CD_GUARANTEE_FULL=1`` 2 048.

Regime: CPU fp32, vendored v0.29.0 stock step, V = 262 144. Rows 1-2: CL 8, 16 max steps, ST 2, bound 1.0, threshold
0.1, t 0.4-0.8. Row 3: T-0068 c3's regime (CL 32, bound 0.1, threshold 0.005, 3 max steps).
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Callable
from typing import Any

import pytest
import torch

from tests.cd import _wire_support as ws
from tests.cd.fakes.diffusion_sampler import STOP_IDS, Cfg, World
from tests.cd.integration import test_cd_multicanvas_guarantee as mg
from tests.cd.unit import _perstep_support as ps
from tests.cd.unit import _seed_support as sd
from tests.cd.unit._sampler_support import error_count

pytestmark = pytest.mark.integration

OPEN, A, B, C, CLOSE = 1003, 1000, 1001, 1002, 1004
CL = 8
ARMS6 = ("end_projection", "mf", "mar", "mar_sc", "qa", "qa_sc")
DRAFT_OK = (OPEN, B, CLOSE, 1, 1, 1, 1, 1)
DRAFTS_BAD = {
    "out_of_enum": (OPEN, C, CLOSE, 1, 1, 1, 1, 1),
    "stop_inside_string": (OPEN, 1, CLOSE, 1, 1, 1, 1, 1),
}


def enum_grammar() -> Any:
    return ps.product_grammar([[OPEN], [A, B], [CLOSE]], "t0081-enum")


def small_cfg() -> Cfg:
    return Cfg(
        CL=CL,
        max_denoising_steps=16,
        stability_threshold=1,
        entropy_bound=1.0,
        confidence_threshold=0.1,
        max_num_reqs=2,
    )


def arm_xargs(g: Any, arm: str, tag: str) -> dict[str, Any]:
    if arm == "end_projection":
        return ws.raw_xargs(
            g.payload, mode="end_projection", self_conditioning="model", record_tag=tag
        )
    mode, sc = ps.ALL_ARMS[arm]
    return ws.raw_xargs(g.payload, mode=mode, self_conditioning=sc, record_tag=tag)


def first_commit(x: dict[str, Any], *, limit: int = 20) -> tuple[list[int], int, list[int]]:
    """Run one request under the copy model until its first commit: (committed ids, denoise calls, seeded canvas)."""
    w = World(small_cfg(), cd=True, seed=0)
    w.add("c", xargs=x, max_tokens=64)
    pre = w.prefill()
    fn = sd.copy_model(w)
    for _ in range(limit):
        s = w.decode(fn)
        ids = s.committed("c")
        if ids is not None:
            w.remove("c")
            return ids, s.call - pre.call - 1, sd.row(pre, "canvas", "c")
    raise AssertionError(f"no commit within {limit} calls")


def until_stop(ids: list[int]) -> list[int]:
    k = next((i for i, t in enumerate(ids) if t in STOP_IDS), None)
    return ids if k is None else ids[: k + 1]


# --------------------------------------------------------------------------- 1. out-of-language drafts


@pytest.mark.parametrize("arm", ARMS6)
@pytest.mark.parametrize("kind", sorted(DRAFTS_BAD))
def test_an_out_of_language_draft_is_repaired_never_committed(arm: str, kind: str) -> None:
    g = enum_grammar()
    draft = DRAFTS_BAD[kind]
    assert not g.accepts([0], until_stop(list(draft)), 10**6), (
        "premise: the draft is out of the language"
    )
    errors0 = error_count()
    ids, _, seeded_canvas = first_commit(
        sd.with_seed(arm_xargs(g, arm, f"t0081-bad-{kind}-{arm}"[:64]), draft)
    )
    assert seeded_canvas == list(draft), "premise: the request was seeded with the draft"
    out = until_stop(ids)
    assert g.accepts([0], out, 10**6), (
        f"{arm}/{kind}: committed {ids} is not accepted by host replay (CS != 1)"
    )
    assert ids != list(draft), f"{arm}/{kind}: the out-of-language draft was committed"
    assert error_count() == errors0, f"{arm}/{kind}: a loud error on a feasible instance"


# --------------------------------------------------------------------------- 2. in-language confident draft


@pytest.mark.parametrize("arm", ["end_projection", "mar", "qa"])
def test_an_in_language_confident_draft_commits_within_two_steps_and_equals_the_draft(
    arm: str,
) -> None:
    g = enum_grammar()
    assert g.accepts([0], until_stop(list(DRAFT_OK)), 10**6), (
        "premise: the draft is in the language"
    )
    ids, steps, seeded_canvas = first_commit(
        sd.with_seed(arm_xargs(g, arm, f"t0081-ok-{arm}"), DRAFT_OK)
    )
    assert seeded_canvas == list(DRAFT_OK), "premise: the request was seeded"
    assert steps <= 2, f"{arm}: {steps} denoise steps before the commit (expected <= 2)"
    assert ids == list(DRAFT_OK), f"{arm}: committed {ids}, the draft was {list(DRAFT_OK)}"
    plain, plain_steps, _ = first_commit(arm_xargs(g, arm, f"t0081-ok-plain-{arm}"))
    assert plain_steps > 2, f"non-vacuity: unseeded {arm} also committed within {plain_steps} steps"
    assert g.accepts([0], until_stop(plain), 10**6)


# --------------------------------------------------------------------------- 3. the seeded guarantee matrix

ROUTINE_RUNS, FULL_RUNS = 48, 2048
N_RUNS = (
    FULL_RUNS
    if os.environ.get("ASTRA_CD_GUARANTEE_FULL") == "1"
    else int(os.environ.get("ASTRA_CD_GUARANTEE_RUNS", str(ROUTINE_RUNS)))
)
MID = 1  # of max_denoising_steps 3 (T-0068 c3's regime)
SEEDS = ((0.0, 0), (0.0, MID), (0.5, 0), (0.5, MID))
CELLS = [(a, n, k) for a in mg.ARMS4 for n, k in SEEDS]
PER_CELL = max(1, math.ceil(N_RUNS / len(CELLS)))


def in_language_walk(o: mg.Oracle, n: int, rng: random.Random) -> list[int]:
    """A random path of ``n`` ids from the start states, staying on states that can still reach a final state."""
    V = o.toy.vocab_size
    states = set(o.start)
    out: list[int] = []
    for _ in range(n):
        opts = [(s, d, c) for s in states for d, c in o.replay.out.get(s, ()) if o.d[d] < math.inf]
        if not opts:
            break
        _s, _d, c = rng.choice(sorted(opts))
        stored, neg = o.toy.classes[c]
        if neg:
            t = rng.randrange(V)
            while t in stored or t in STOP_IDS:
                t = rng.randrange(V)
        else:
            t = rng.choice(sorted(stored))
        out.append(t)
        states = o.replay.run(states, [t])
    return out or [1]


def run_seeded(
    o: mg.Oracle,
    arm: str,
    max_tokens: int,
    fn: Callable[[int, str, int], torch.Tensor],
    draft: list[int],
    noise: float,
    start: int,
    tag: str,
) -> None:
    c = mg.cfg()
    w = World(c, cd=True, seed=0)
    errors0 = error_count()
    w.add(
        "c",
        xargs=sd.with_seed(mg.xargs_for(o, arm, tag), draft, noise=noise, start=start),
        max_tokens=max_tokens,
    )
    pre = w.prefill()
    if noise == 0.0:
        assert sd.row(pre, "canvas", "c")[: len(draft)] == draft, "premise: the request was seeded"
    assert sd.row(pre, "step", "c") == [start], "premise: the start step was applied"
    A = set(o.start)
    out: list[int] = []
    for _ in range((max_tokens + 2) * (c.max_denoising_steps + 2)):
        ids = w.decode(fn).committed("c")
        if ids is None:
            continue
        ids = ids[: max_tokens - len(out)]
        stop_at = next((i for i, t in enumerate(ids) if t in STOP_IDS), None)
        if stop_at is not None:
            ids = ids[: stop_at + 1]
        if error_count() != errors0:
            raise mg.Violation("loud", "a loud error on a feasible instance")
        A = o.replay.run(A, ids)
        out += ids
        if not A:
            raise mg.Violation("viability", f"δ* empty after {len(out)} ids")
        if o.min_d(A) > max_tokens - len(out):
            raise mg.Violation("viability", f"min d {o.min_d(A)} > budget after {len(out)} ids")
        if stop_at is not None or len(out) >= max_tokens:
            w.remove("c")
            if error_count() != errors0:
                raise mg.Violation("loud", "a loud error record on a feasible instance")
            if not o.replay.accepts(o.start, o.toy.finals, out):
                raise mg.Violation(
                    "acceptance", f"output of {len(out)} ids not accepted by host replay"
                )
            return
    raise mg.Violation("termination", f"no termination ({len(out)} ids)")


@pytest.mark.slow
@pytest.mark.parametrize(
    ("arm", "noise", "start"), CELLS, ids=[f"{a}-n{n}-k{k}" for a, n, k in CELLS]
)
def test_seeded_guarantee_matrix(arm: str, noise: float, start: int) -> None:
    fails: list[str] = []
    cell = CELLS.index((arm, noise, start))
    for r in range(PER_CELL):
        kind = mg.KINDS[(r + cell) % len(mg.KINDS)]
        max_tokens = mg.MAX_TOKENS[(r // len(mg.KINDS) + cell) % len(mg.MAX_TOKENS)]
        seed = 81_000_000 + cell * 1009 + r
        o = mg.oracle(kind, seed % 16) if kind.startswith("nfa") else mg.oracle(kind)
        fn = (
            mg.adversarial_logits(seed)
            if r % 4 == 3
            else mg.random_logits(seed, stop_averse=r % 4 == 2)
        )
        rng = random.Random(seed)
        if r % 2 == 0:
            draft = [int(t) for t in fn(0, "c", mg.CL).argmax(dim=-1)]
        else:
            draft = in_language_walk(o, rng.randint(1, mg.CL), rng)
        try:
            run_seeded(o, arm, max_tokens, fn, draft, noise, start, f"sg-{kind}-{cell}-{r}")
        except mg.Violation as v:
            fails.append(f"run {r} ({kind}, max_tokens {max_tokens}, seed {seed}): {v}")
            if len(fails) >= 3:
                break
    assert fails == [], f"{arm} noise {noise} start {start}: " + "; ".join(fails)
