"""T-0068 criteria 3-4: the multi-canvas guarantee (SPEC §3.1b proposition), CPU simulation. Integration, slow.

The vendored v0.29.0 stock step, the CD sampler (T-0067 wrapper, per-step arms from T-0068) and a fake scheduler
that trims each canvas at its ``valid_canvas_len`` (``num_sampled``), the whole output at ``max_tokens``, and at
the first stop id (inclusive) -- which ends the request. Every check is a host replay over the compiled automaton
(``_draw_support.Replay``: independent set-of-states replay; distances by reverse BFS in ``oracle/enumerate.py``),
never the plugin's δ*.

Required, per run (criterion 3):
  * after every commit ``k``: ``A_{k+1} = δ*(A_k, emitted ids of canvas k) ≠ ∅`` and ``min_{s ∈ A_{k+1}} d(s)`` ≤
    the remaining budget ``max_tokens − emitted`` (a viable prefix within budget);
  * the run terminates (a stop id is emitted) by ``max_tokens`` at the latest, and the full output id sequence is
    accepted by host replay (reaches the stop-augmented final state);
  * zero loud errors (the per-process error counter does not move) on these feasible instances.

Matrix: automata {BFCL ``live_simple_2-2-0`` (JSON whitespace), the first Sudoku test puzzle, two families of
seeded synthetic NFAs} × arms {``end_projection``, ``perstep_mf``, ``perstep_mar``, ``perstep_mar`` + constrained SC, and (T-0076
criterion 5) ``perstep_mar_qargmax`` ± constrained SC} × ``max_tokens`` {256, 300, 512, 768} = 96 cells ×
ceil(N / 96) seeded runs. Run count (ruling P11): routine default N = 288 (3 per cell: one random, one adversarial, one stop-averse; the first also truncates
``valid_canvas_len``), which keeps ``make cd-test`` tractable and covers every cell; ``ASTRA_CD_GUARANTEE_RUNS=<n>``
overrides it; ``ASTRA_CD_GUARANTEE_FULL=1`` is the explicit opt-in for the criterion's full matrix (N = 2048, i.e.
the 2000 runs rounded up to 32 per cell), run once at verification. Seeds alternate random logits (N(0, 3²) over all of V; every other one stop-averse) and adversarial logits
(criterion 4: EOS preferred at position 0, ``<turn|>`` = 106 preferred everywhere including inside strings, PAD = 0
preferred at every position after the first). Every third seed also sets ``max_model_len`` just above
``prompt + max_tokens`` so the last canvas is truncated (``valid_canvas_len < CL``) without shrinking the budget.
Regime: CL 32, stock bound 0.1 / threshold 0.005, ``max_denoising_steps`` 3 (keeps runs short; convergence by
the cap is the common case under random logits), stability_threshold 1, T schedule 0.4-0.8, torch CPU.

The adversarial core (task file): a budget one canvas too generous must produce a detectable violation --
``test_harness_detects_a_budget_off_by_one_canvas`` injects exactly that and requires the harness to flag it.
"""

from __future__ import annotations

import functools
import math
import os
import random
import statistics
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from tests.cd import _compile_support as cs
from tests.cd import _wire_support as ws
from tests.cd.fakes.diffusion_sampler import EOS, STOP_IDS, Cfg, V, World
from tests.cd.oracle.enumerate import Toy, distance_to_final
from tests.cd.unit import _perstep_support as ps
from tests.cd.unit._draw_support import Replay
from tests.cd.unit._sampler_support import LIVE, Records, error_count, patch_everywhere

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: ruling P11: routine 192 (3 per cell); ASTRA_CD_GUARANTEE_FULL=1 opts in to the full 2048-run matrix
#: T-0076 criterion 5 adds the two qargmax arms (96 cells): routine 288 keeps 3 per cell; the full opt-in is
#: ceil(2048 / 96) = 22 per cell = 2 112 runs (≥ the criterion's 2 000)
ROUTINE_RUNS, FULL_RUNS = 288, 2048
N_RUNS = (
    FULL_RUNS
    if os.environ.get("ASTRA_CD_GUARANTEE_FULL") == "1"
    else int(os.environ.get("ASTRA_CD_GUARANTEE_RUNS", str(ROUTINE_RUNS)))
)
KINDS = ("bfcl", "sudoku", "nfa_chain", "nfa_dense")
ARMS4 = (
    "end_projection",
    "mf",
    "mar",
    "mar_sc",
    "qa",
    "qa_sc",
)  # T-0076: + perstep_mar_qargmax (+ SC)
MAX_TOKENS = (256, 300, 512, 768)
N_CELLS = len(KINDS) * len(ARMS4) * len(MAX_TOKENS)
PER_CELL = max(1, math.ceil(N_RUNS / N_CELLS))
CL = 32
PAD, TURN_CLOSE = 0, 106
BFCL_ID = "live_simple_2-2-0"


def cfg(**kw: Any) -> Cfg:
    base: dict[str, Any] = dict(
        CL=CL,
        stability_threshold=1,
        entropy_bound=0.1,
        confidence_threshold=0.005,
        max_denoising_steps=3,
        max_num_reqs=2,
    )
    base.update(kw)
    return Cfg(**base)


# --------------------------------------------------------------------------- automata and the host oracle


class Oracle:
    """Host replay and distance-to-final over a compiled (stop-augmented) automaton."""

    def __init__(self, compiled: Any) -> None:
        t = compiled.tables
        indptr, indices, neg = (
            np.asarray(t.sum_indptr),
            np.asarray(t.sum_indices),
            np.asarray(t.sum_is_neg),
        )
        classes = tuple(
            (frozenset(int(v) for v in indices[int(indptr[c]) : int(indptr[c + 1])]), bool(neg[c]))
            for c in range(len(indptr) - 1)
        )
        n_edges = int(compiled.n_edges)
        edges = tuple(
            (int(s), int(d), int(c))
            for s, d, c in zip(
                np.asarray(compiled.edge_src)[:n_edges],
                np.asarray(compiled.edge_dst)[:n_edges],
                np.asarray(compiled.edge_class)[:n_edges],
                strict=True,
            )
        )
        n = int(compiled.n_states)
        finals = frozenset(int(s) for s in np.flatnonzero(np.asarray(compiled.is_final)[:n]))
        self.toy = Toy(
            n_states=n,
            vocab_size=int(compiled.vocab_size),
            classes=classes,
            edges=edges,
            finals=finals,
        )
        self.replay = Replay(self.toy)
        self.d = distance_to_final(self.toy)
        self.start = {int(s) for s in np.flatnonzero(np.asarray(compiled.start_vector)[:n])}
        self.payload = ws.encode(compiled)

    def min_d(self, states: set[int]) -> float:
        return min((self.d[s] for s in states), default=math.inf)


def _synthetic(kind: str, seed: int) -> Any:
    rng = random.Random(seed)
    ids = list(range(1000, 1012))
    if kind == "nfa_chain":
        n = rng.randint(5, 9)
        t = [(i, rng.choice(ids), i + 1) for i in range(n - 1)]
        t += [(rng.randrange(n), rng.choice(ids), rng.randrange(n)) for _ in range(2 * n)]
    else:
        n = rng.randint(3, 6)
        t = [(i, rng.choice(ids), i + 1) for i in range(n - 1)]
        t += [(s, v, rng.randrange(n)) for s in range(n) for v in rng.sample(ids, 4)]
    finals = {n - 1} | {s for s in range(n) if rng.random() < 0.2}
    return ws.hand_built(n, sorted(set(t)), finals, name=f"t0068-{kind}-{seed}")


@functools.cache
def oracle(kind: str, seed: int = 0) -> Oracle:
    if kind == "bfcl":
        return Oracle(cs.grammars().compile_bfcl(cs.bfcl_function(BFCL_ID), whitespace="json"))
    if kind == "sudoku":
        return Oracle(cs.compiled_sudoku(cs.sudoku_cut("test")[0][1]))
    return Oracle(_synthetic(kind, seed))


# --------------------------------------------------------------------------- logits


def random_logits(
    seed: int, *, stop_averse: bool = False
) -> Callable[[int, str, int], torch.Tensor]:
    """N(0, 3²) over all of V. ``stop_averse``: the stop ids sit 12 nats lower, so the model never wants to end
    and only the budget can make it (the case where a budget off by one canvas overruns ``max_tokens``)."""

    def fn(call: int, req: str, n: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed * 100_003 + call)
        x = torch.randn(n, V, generator=g) * 3.0
        if stop_averse:
            x[:, list(STOP_IDS)] -= 12.0
        return x

    return fn


def adversarial_logits(seed: int) -> Callable[[int, str, int], torch.Tensor]:
    def fn(call: int, req: str, n: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed * 100_003 + call)
        x = torch.randn(n, V, generator=g)
        x[:, TURN_CLOSE] = 12.0  # <turn|> everywhere, including inside strings
        x[1:, PAD] = 14.0  # PAD after the first position
        x[0, EOS] = 15.0  # EOS first
        return x

    return fn


# --------------------------------------------------------------------------- the run and its checks


class Violation(AssertionError):
    def __init__(self, kind: str, msg: str) -> None:
        super().__init__(f"[{kind}] {msg}")
        self.kind = kind


def xargs_for(o: Oracle, arm: str, tag: str) -> dict[str, Any]:
    if arm == "end_projection":
        return ws.raw_xargs(
            o.payload, mode="end_projection", self_conditioning="model", record_tag=tag
        )
    mode, sc = ps.ALL_ARMS[arm]
    return ws.raw_xargs(o.payload, mode=mode, self_conditioning=sc, record_tag=tag)


def run_once(
    o: Oracle,
    arm: str,
    max_tokens: int,
    fn: Callable[[int, str, int], torch.Tensor],
    *,
    truncate_extra: int | None = None,
    tag: str = "guar",
) -> dict[str, Any]:
    """Drive one request to its end under the fake scheduler; raise ``Violation`` on the first broken
    requirement. Returns the output ids and the commit count."""
    c = cfg()
    w = World(c, cd=True, seed=0)
    errors0 = error_count()
    w.add("c", xargs=xargs_for(o, arm, tag), max_tokens=max_tokens)
    w.prefill()
    req = w.reqs["c"]
    out: list[int] = []
    A = set(o.start)
    max_calls = (max_tokens + 2) * (c.max_denoising_steps + 2)
    for _call in range(max_calls):
        if truncate_extra is not None:
            room = max_tokens + truncate_extra - len(out)  # max_model_len − prompt − emitted
            req.valid_len = max(1, min(CL, room))
        snap = w.decode(fn)
        ids = snap.committed("c")
        if ids is None:
            continue
        remaining = max_tokens - len(out)
        ids = ids[:remaining]  # the scheduler trims at max_tokens
        stop_at = next((i for i, t in enumerate(ids) if t in STOP_IDS), None)
        if stop_at is not None:
            ids = ids[: stop_at + 1]
        if error_count() != errors0:
            raise Violation("loud", f"a loud error on a feasible instance (commit {ids[:8]}...)")
        A = o.replay.run(A, ids)
        out += ids
        if not A:
            raise Violation(
                "viability", f"δ* empty after commit {len(out)} ids: {out[-len(ids) :]}"
            )
        budget = max_tokens - len(out)
        if o.min_d(A) > budget:
            raise Violation(
                "viability",
                f"min d = {o.min_d(A)} > remaining budget {budget} after {len(out)} ids",
            )
        if stop_at is not None or len(out) >= max_tokens:
            w.remove("c")  # the engine frees the request; T-0067 S11 flushes its records on release
            if error_count() != errors0:
                raise Violation(
                    "loud", f"a loud error record on a feasible instance ({len(out)} ids)"
                )
            if not o.replay.accepts(o.start, o.toy.finals, out):
                raise Violation(
                    "acceptance", f"output of {len(out)} ids is not accepted by host replay"
                )
            return {"ids": out}
    raise Violation(
        "termination", f"no termination within {max_calls} calls ({len(out)} ids emitted)"
    )


CELLS = [(k, a, m) for k in KINDS for a in ARMS4 for m in MAX_TOKENS]


@pytest.mark.parametrize(
    ("kind", "arm", "max_tokens"), CELLS, ids=[f"{k}-{a}-{m}" for k, a, m in CELLS]
)
def test_every_commit_is_viable_and_every_output_accepted(
    kind: str, arm: str, max_tokens: int
) -> None:
    fails: list[str] = []
    for r in range(PER_CELL):
        seed = (
            MAX_TOKENS.index(max_tokens) * 1_000_003
            + ARMS4.index(arm) * 10_007
            + KINDS.index(kind) * 101
            + r
        )
        o = oracle(kind, seed % 16) if kind.startswith("nfa") else oracle(kind)
        fn = adversarial_logits(seed) if r % 2 else random_logits(seed, stop_averse=r % 4 == 2)
        extra = (seed % CL) if r % 3 == 0 else None
        try:
            run_once(o, arm, max_tokens, fn, truncate_extra=extra, tag=f"g-{kind}-{r}")
        except Violation as v:
            fails.append(f"run {r} (seed {seed}, {'adv' if r % 2 else 'rand'}): {v}")
            if len(fails) >= 3:
                break
    assert fails == [], f"{kind}/{arm}/max_tokens={max_tokens}: " + "; ".join(fails)


def test_harness_detects_a_budget_off_by_one_canvas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Liveness of the checks above: make ``terminal_factor`` believe one more canvas remains than does. The
    last canvas is then not forced to end, and the harness must report a viability, termination or acceptance
    violation (not a loud error) within 4 runs on BFCL (stop-averse logits), for ``end_projection`` and for ``perstep_mar``."""
    from tests.cd.fakes.diffusion_sampler import cd_module

    real = cd_module("budget").terminal_factor
    calls = {"n": 0}

    def generous(tables: Any, r_after: Any) -> Any:
        calls["n"] += 1
        return real(tables, r_after + CL)

    ps.steps_module()  # per-step code must be loaded so its namespace is patched too
    assert patch_everywhere(monkeypatch, "terminal_factor", generous) > 0
    o = oracle("bfcl")
    for arm in ("end_projection", "mar"):
        kinds: list[str] = []
        for r in range(4):
            try:
                run_once(o, arm, 256 + (r % 4) * 3, random_logits(90_000 + r, stop_averse=True))
            except Violation as v:
                kinds.append(v.kind)
                break
        assert calls["n"] > 0, (
            "cannot inject: terminal_factor was never called through a patched namespace"
        )
        assert kinds and kinds[0] in ("viability", "termination", "acceptance"), (
            f"{arm}: a budget one canvas too generous went undetected (violations: {kinds})"
        )


# --------------------------------------------------------------------------- criterion 4: stop position


def _ends_at(seed: int, e: int) -> Callable[[int, str, int], torch.Tensor]:
    """The model ends at position ``e``: live ids before it, EOS at ``e``, and live text *after* it (a MAP that
    scored the post-stop tail, or forced a PAD tail, would pay for an early stop and push it to CL − 1)."""
    g = torch.Generator().manual_seed(seed)
    top = torch.randint(0, len(LIVE), (256,), generator=g)

    def fn(call: int, req: str, n: int) -> torch.Tensor:
        x = torch.randn(n, V, generator=torch.Generator().manual_seed(seed + call)) * 0.5
        for i in range(n):
            x[i, LIVE[int(top[i])]] = 8.0
        if e < n:
            x[e, EOS] = 10.0
        return x

    return fn


@pytest.mark.parametrize("arm", ARMS4)
def test_stop_position_follows_the_model(arm: str, caplog: Any, capsys: Any) -> None:
    """Loop language over the live ids, CL 64: the model ends at e ~ U[5, 40]. Median recorded stop_pos ≤ 40 over
    15 runs (the stop can come later only if the budget or the language forced it; neither does here)."""
    records = Records(caplog, capsys)
    g = ps.loop_grammar(LIVE, "t0068-stop")
    o_payload = g.payload
    stops: list[int] = []
    for r in range(15):
        rng = random.Random(4000 + r)
        e = rng.randint(5, 40)
        tag = f"stop-{arm}-{r}"
        if arm == "end_projection":
            x = ws.raw_xargs(
                o_payload, mode="end_projection", self_conditioning="model", record_tag=tag
            )
        else:
            mode, sc = ps.ALL_ARMS[arm]
            x = ws.raw_xargs(o_payload, mode=mode, self_conditioning=sc, record_tag=tag)
        w = World(cfg(CL=64), cd=True, seed=r)
        w.add("c", xargs=x, max_tokens=4096)
        w.prefill()
        fn = _ends_at(4000 + r, e)
        for _ in range(8):
            s = w.decode(fn)
            if s.committed("c") is not None:
                break
        w.decode(fn)
        w.decode(fn)  # records lag ≤ 2 calls
        recs = [rec for rec in records.tagged(tag) if rec.get("canvas_idx") == 0]
        assert len(recs) == 1, f"{arm} run {r}: records {recs}"
        assert recs[0]["error"] is None, recs[0]
        assert recs[0]["stop_pos"] is not None, (
            f"{arm} run {r}: no stop in canvas 0 (model ends at {e})"
        )
        stops.append(int(recs[0]["stop_pos"]))
    med = statistics.median(stops)
    assert med <= 40, (
        f"{arm}: median stop_pos {med} > 40 (stops {stops}); the MAP's stop drifts to CL − 1"
    )
