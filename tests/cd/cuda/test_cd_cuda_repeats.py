"""T-0067 criterion 12 (tier 2): the T-0065/T-0066 core criteria repeated on CUDA fp32 with
``torch.set_float32_matmul_precision("high")`` set globally (the TF32 landmine), the compensated-arithmetic rows
on the served path (eager and ``torch.compile``d: a reassociating backend silently removes TwoSum), and the peak
added device memory of an ``end_projection`` convergence step at CL 256, bucket ≤ 256.

Runs **inside the pinned container** (no engine boot), driven by ``tests/contract/test_vllm_cd_cuda_gpu.py``,
which mounts this tree read-only. Oracles are the float64/``Fraction`` ones under ``tests/cd/oracle`` (numpy and
stdlib only). Instance sizes are reduced from the CPU rows where the CPU oracle would dominate: criterion 2's
differential runs at L = 256, V = 4096 (not 262 144), stated per row.
"""

from __future__ import annotations

import itertools
import math
import os
import random
from collections import Counter
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch

from tests.cd._quiet_logs import quiet_torch_logs
from tests.cd.oracle import dense64, viterbi64
from tests.cd.oracle import enumerate as en
from tests.cd.oracle.enumerate import Toy
from tests.cd.unit._core_support import SENTINEL_MAX, core, random_logits, random_toy

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def quiet_torch_trace_per_row() -> Iterator[None]:
    """T-0077c: keep torch.__trace DEBUG lines (torch.compile) out of pytest's captured logs, so a red row's message
    survives in the wrapper's evidence; restored after each row (``tests/cd/_quiet_logs.py``)."""
    with quiet_torch_logs():
        yield


@pytest.fixture(autouse=True, scope="module")
def tf32_high() -> Iterator[None]:
    if not torch.cuda.is_available():
        why = "CUDA rows run inside the pinned container (tests/contract/test_vllm_cd_cuda_gpu.py)"
        if os.environ.get("ASTRA_CD_REQUIRE_CUDA") == "1":
            pytest.fail(f"ASTRA_CD_REQUIRE_CUDA=1 but CUDA is not available: {why}")
        pytest.skip(why)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    # evidence for the host row: the precision really in force while these rows run
    print(f"ASTRA_TIER2_MATMUL_PRECISION {torch.get_float32_matmul_precision()}")
    yield
    torch.set_float32_matmul_precision(prev)


DEV = torch.device("cuda")


def tables(toy: Toy, bucket: int | None = None) -> Any:
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


def vec(idx: Any, n: int) -> torch.Tensor:
    m = torch.zeros(n, dtype=torch.bool, device=DEV)
    m[list(idx)] = True
    return m


def pipeline(
    logits: torch.Tensor, toy: Toy, start: Any, end: Any, l_eff: int
) -> tuple[Any, Any, Any]:
    t = tables(toy)
    tree = core("tree").up_sweep(core("emission").log_transfer(logits, t, l_eff))
    res = core("marginals").constrained_marginals(
        logits, t, tree, vec(start, t.bucket), vec(end, t.bucket), l_eff
    )
    return res, t, tree


def instances(n: int, seed: int) -> list[tuple[Toy, list[list[float]], list[int], int]]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        V, L, S = rng.choice([4, 5, 6]), rng.choice([4, 6, 8]), rng.choice([3, 4, 5])
        toy = random_toy(rng, V=V, S=S, nfa=len(out) % 2 == 1)
        logits = random_logits(rng, L, V, span=8.0)
        start = [0] if len(out) % 3 else [0, 1]
        oracle = en.enumerate_log(en.log_softmax64(logits), toy, start, sorted(toy.finals), L)
        if math.isfinite(oracle.log_z):
            out.append((toy, logits, start, L))
    return out


# --------------------------------------------------------------------------- T-0065 criteria 1-5


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["fp64", "fp32"])
def test_t0065_c1_exactness_on_cuda(dtype: torch.dtype) -> None:
    for toy, logits, start, L in instances(12, seed=1):
        o = en.enumerate_log(en.log_softmax64(logits), toy, start, sorted(toy.finals), L)
        res, _t, _tree = pipeline(
            torch.tensor(logits, dtype=dtype, device=DEV), toy, start, sorted(toy.finals), L
        )
        dz = abs(float(res.log_z) - o.log_z)
        dq = float(np.max(np.abs(res.q.double().cpu().numpy()[:L] - np.array(o.q))))
        dh = float(np.max(np.abs(res.entropy.double().cpu().numpy()[:L] - np.array(o.entropy))))
        if dtype == torch.float64:
            assert dz <= 1e-10 and dq <= 1e-12, (dz, dq)
        else:
            assert dz <= 1e-4 * max(1.0, abs(o.log_z)) and dq <= 1e-6 and dh <= 1e-5, (dz, dq, dh)


@pytest.mark.parametrize("temp", [0.8, 0.4])
def test_t0065_c2_differential_vs_stock_error_on_cuda(temp: float) -> None:
    """L = 256, V = 4096 (reduced from 262 144), softcapped ×30 at T: fp32 H(q) error ≤ 10× the stock fp32
    H(p) error on the same logits + 1e-6 (and q likewise), against dense64."""
    rng = random.Random(7)
    V, L = 4096, 256
    toy = random_toy(rng, V=V, S=12, nfa=True)
    g = torch.Generator().manual_seed(3)
    raw = torch.randn(L, V, generator=g, dtype=torch.float64) * 40.0
    # T-0073 D2 (human): the oracle and the bar's fp64 reference both use the fp32-rounded inputs the core and
    # the stock fp32 H(p) receive (from the unrounded logits the oracle moved by up to 1.58e-6 in H at T 0.4)
    x64 = ((30.0 * torch.tanh(raw / 30.0)) / temp).to(torch.float32).to(torch.float64)
    start, end = np.zeros(toy.n_states, bool), np.zeros(toy.n_states, bool)
    start[0] = True
    end[sorted(toy.finals)] = True
    o = dense64.forward_backward64(x64.numpy(), toy, start, end, L)
    if not math.isfinite(o.log_z):
        pytest.fail("fixture: empty language on this instance")
    x32 = x64.to(torch.float32).to(DEV)
    res, _t, _tree = pipeline(x32, toy, [0], sorted(toy.finals), L)
    lp = torch.log_softmax(x32, dim=-1)
    hp32 = -(lp.exp() * lp).sum(-1).double().cpu().numpy()
    lp64 = torch.log_softmax(x64, dim=-1)
    hp64 = -(lp64.exp() * lp64).sum(-1).numpy()
    bar = 10.0 * np.abs(hp32 - hp64) + 1e-6
    err = np.abs(res.entropy.double().cpu().numpy()[:L] - o.entropy)
    assert np.all(err <= bar), f"worst H(q) error / bar = {float(np.max(err / bar)):.2f}"


def test_t0065_c3_non_subtractive_complement_on_cuda() -> None:
    V = 4096
    toy = Toy(
        n_states=2,
        vocab_size=V,
        classes=((frozenset(range(10)), True), (frozenset({20}), False)),
        edges=((0, 1, 0), (0, 1, 1)),
        finals=frozenset({1}),
    )
    x = torch.zeros(1, V, dtype=torch.float64)
    x[0, :10] = -40.0
    w = core("emission").class_log_weights(x.to(torch.float32).to(DEV), tables(toy))
    lp = en.log_softmax64(x.tolist())[0]
    members = [v for v in range(V) if v >= 10]
    m = max(lp[v] for v in members)
    want = m + math.log(math.fsum(math.exp(lp[v] - m) for v in members))
    got = float(w[0, 0])
    assert abs(got - want) <= 1e-4 * max(abs(want), 1e-7) or abs(got - want) <= 1e-9, (got, want)


def test_t0065_c4_two_band_no_spurious_zero_on_cuda() -> None:
    """Allowed ids sit ~106 nats under a forbidden argmax at each of 8 positions (≈ 850 nats in total)."""
    V, L = 64, 8
    toy = Toy(
        n_states=2,
        vocab_size=V,
        classes=((frozenset({5, 6}), False),),
        edges=((0, 1, 0), (1, 1, 0)),
        finals=frozenset({1}),
    )
    x = torch.full((L, V), -200.0, dtype=torch.float64)
    x[:, 0] = 0.0
    x[:, 5] = -106.0
    x[:, 6] = -106.5
    start, end = np.array([True, False]), np.array([False, True])
    o = dense64.forward_backward64(x.numpy(), toy, start, end, L)
    res, _t, _tree = pipeline(x.to(torch.float32).to(DEV), toy, [0], [1], L)
    lz = float(res.log_z)
    assert math.isfinite(lz), "spurious Z = 0"
    tol = 1e-4 * max(1.0, abs(o.log_z))
    assert abs(lz - o.log_z) <= tol
    la, lb = res.log_a.double().cpu(), res.log_b.double().cpu()
    per_i = torch.logsumexp(la + lb, dim=-1)[: L + 1]
    assert float((per_i - o.log_z).abs().max()) <= tol, per_i


#: T-0073 D0: the c5 fixture's seed. Seed 5 (the T-0067 original) has an EMPTY language under its −inf mask (0 of
#: 729 strings over the finite ids is accepted), so the row passed on q ≡ 0 vacuously and then failed on H. Seed 13
#: keeps the builder and the −inf / −1e20 stress with a non-empty language: 378 accepted strings over the finite
#: ids, 189 through the −1e20 id (live but negligible) and 189 avoiding it.
C5_SEED = 13
#: the original seed: the empty-language row's fixture
C5_EMPTY_SEED = 5


def c5_fixture(seed: int) -> tuple[Toy, torch.Tensor, int, int]:
    """(toy, top-3-filtered logits with a −1e20 sentinel at row 0's argmax, L, the sentinel's id)."""
    rng = random.Random(seed)
    toy = random_toy(rng, V=6, S=4, nfa=True)
    L = 6
    x = torch.tensor(random_logits(rng, L, 6, span=8.0), dtype=torch.float32)
    keep = x.topk(3, dim=-1).indices
    filt = torch.full_like(x, float("-inf")).scatter(1, keep, x.gather(1, keep))
    sid = int(keep[0, 0])
    filt[0, sid] = -1e20
    return toy, filt, L, sid


def accepted_over_finite_ids(
    toy: Toy, filt: torch.Tensor, start: list[int], L: int
) -> list[tuple[int, ...]]:
    """Independent of the enumerator's arithmetic: every string over the finite (unfiltered) ids, replayed."""
    allowed = [
        [v for v in range(filt.shape[1]) if math.isfinite(float(filt[i, v]))] for i in range(L)
    ]
    return [
        s
        for s in itertools.product(*allowed)
        if en.accepts(toy, start, sorted(toy.finals), list(s))
    ]


def test_t0065_c5_no_nan_with_neg_inf_and_sentinels_on_cuda() -> None:
    toy, filt, L, sid = c5_fixture(C5_SEED)
    start = [0]
    accepted = accepted_over_finite_ids(toy, filt, start, L)
    assert accepted, "fixture: empty language under the −inf mask (the row would pass on q ≡ 0)"
    assert any(s[0] == sid for s in accepted) and any(s[0] != sid for s in accepted), (
        "fixture: the −1e20 id must lie on some accepted strings and be avoidable by others"
    )
    # exact oracle: log_softmax64 keeps −inf, so an empty language gives log Z = −inf (no −1e300 clamp)
    o = en.enumerate_log(
        en.log_softmax64(filt.double().tolist()), toy, start, sorted(toy.finals), L
    )
    if not math.isfinite(o.log_z):
        pytest.fail("fixture: empty language")
    res, _t, _tree = pipeline(filt.to(DEV), toy, start, sorted(toy.finals), L)
    q, h = res.q.cpu(), res.entropy.cpu()[:L]
    assert torch.isfinite(res.log_z.cpu()) and torch.isfinite(q).all() and torch.isfinite(h).all()
    assert (q[torch.isinf(filt)] == 0).all(), "forbidden ids must have q exactly 0"
    assert (h >= 0).all()


def test_t0065_c5_empty_language_is_detected_on_cuda() -> None:
    """The original c5 fixture (seed 5): no string over the finite ids is accepted. T-0073 D0: the empty language
    is detected — ``constrained_marginals``' log Z at the sentinel, and the served Z = 0 signal
    (``constrained_map(...).empty``, from which the sampler raises ``z0``) fires with ``score`` at the sentinel.
    Nothing is asserted about q or H(q) (unspecified when Z = 0, T-0065 R6)."""
    toy, filt, L, _sid = c5_fixture(C5_EMPTY_SEED)
    start = [0]
    assert accepted_over_finite_ids(toy, filt, start, L) == [], (
        "fixture: the language must be empty"
    )
    o = en.enumerate_log(
        en.log_softmax64(filt.double().tolist()), toy, start, sorted(toy.finals), L
    )
    assert o.log_z == -math.inf, o.log_z
    res, _t, _tree = pipeline(filt.to(DEV), toy, start, sorted(toy.finals), L)
    lz = float(res.log_z.cpu())
    assert not math.isnan(lz) and lz <= SENTINEL_MAX, (
        f"empty language not at the sentinel: log Z = {lz}"
    )
    m = _map(filt.to(DEV), toy, start, L)
    assert bool(m.empty.cpu()), "empty language not detected: constrained_map(...).empty is False"
    assert float(m.score.cpu()) <= SENTINEL_MAX, float(m.score.cpu())


# --------------------------------------------------------------------------- T-0066 criteria 1, 4, 5


def _chi2_p(counts: Counter[Any], probs: dict[Any, float], n: int) -> float:
    cells = sorted(probs.items(), key=lambda kv: kv[1])
    pooled, acc_o, acc_e = [], 0, 0.0
    for key, p in cells:
        acc_o += counts.get(key, 0)
        acc_e += p * n
        if acc_e >= 5:
            pooled.append((acc_o, acc_e))
            acc_o, acc_e = 0, 0.0
    if acc_e and pooled:
        o_, e_ = pooled.pop()
        pooled.append((o_ + acc_o, e_ + acc_e))
    stat = sum((o_ - e_) ** 2 / e_ for o_, e_ in pooled)
    df = len(pooled) - 1
    return float(
        torch.special.gammaincc(
            torch.tensor(df / 2.0, dtype=torch.float64),
            torch.tensor(stat / 2.0, dtype=torch.float64),
        )
    )


def test_t0066_c1_draw_exactness_and_cuda_rng_discipline() -> None:
    """n = 20 000 draws per instance (2 instances: DFA, NFA), χ² with cells pooled to ≥ 5 expected,
    Bonferroni α = 1e-3 over the 2 instances; the CUDA and CPU global RNG states are unchanged."""
    n, alpha = 20_000, 1e-3 / 2
    for seed, toy_nfa in ((11, False), (12, True)):
        rng = random.Random(seed)
        toy = random_toy(rng, V=4, S=3, nfa=toy_nfa)
        L = 4
        logits = random_logits(rng, L, 4, span=3.0)
        post = en.string_posterior(en.log_softmax64(logits), toy, [0], sorted(toy.finals), L)
        if not post:
            pytest.fail("fixture: empty language")
        x = torch.tensor(logits, dtype=torch.float32, device=DEV)
        t = tables(toy)
        tree = core("tree").up_sweep(core("emission").log_transfer(x, t, L))
        a, b = vec([0], t.bucket), vec(sorted(toy.finals), t.bucket)
        g = torch.Generator(device=DEV).manual_seed(seed)
        cpu0, cuda0 = torch.get_rng_state(), torch.cuda.get_rng_state()
        draw = core("draw").joint_draw
        counts: Counter[Any] = Counter()
        for _ in range(n):
            d = draw(x, t, tree, a, b, L, generator=g)
            counts[tuple(int(v) for v in d.tokens[:L].cpu())] += 1
        assert torch.equal(torch.get_rng_state(), cpu0) and torch.equal(
            torch.cuda.get_rng_state(), cuda0
        )
        outside = [k for k in counts if k not in post]
        assert outside == [], f"draws outside the support: {outside[:5]}"
        p = _chi2_p(counts, post, n)
        assert p >= alpha, f"χ² p = {p:.2e} < {alpha:.1e} (seed {seed})"


def _map(x: torch.Tensor, toy: Toy, start: list[int], L: int) -> Any:
    t = tables(toy)
    return core("mapdecode").constrained_map(
        x, t, vec(start, t.bucket), vec(sorted(toy.finals), t.bucket), L
    )


def test_t0066_c4_map_exactness_and_temperature_invariance_on_cuda() -> None:
    checked = 0
    for toy, logits, start, L in instances(10, seed=21):
        x = torch.tensor(logits, dtype=torch.float32, device=DEV)
        m = _map(x, toy, start, L)
        toks = [int(v) for v in m.tokens[:L].cpu()]
        best = viterbi64.map_score64(np.array(logits), toy, start, sorted(toy.finals), L)
        assert en.accepts(toy, start, sorted(toy.finals), toks)
        assert abs(viterbi64.string_score64(np.array(logits), toks, L) - best) <= 1e-5 * max(
            1.0, abs(best)
        )
        for temp in (0.8, 0.4):
            m_t = _map(x / temp, toy, start, L)
            assert [int(v) for v in m_t.tokens[:L].cpu()] == toks, f"MAP changed at T = {temp}"
        checked += 1
    assert checked == 10


def test_t0066_c5_projection_identity_on_cuda() -> None:
    hit_in, hit_out = 0, 0
    for toy, logits, start, L in instances(16, seed=31):
        x = torch.tensor(logits, dtype=torch.float32, device=DEV)
        xstar = [int(v) for v in x.argmax(-1).cpu()]
        toks = [int(v) for v in _map(x, toy, start, L).tokens[:L].cpu()]
        if en.accepts(toy, start, sorted(toy.finals), xstar):
            assert toks == xstar
            hit_in += 1
        else:
            assert toks != xstar and en.accepts(toy, start, sorted(toy.finals), toks)
            hit_out += 1
    # a valid argmax planted on one instance so both branches are exercised
    toy, logits, start, L = instances(1, seed=41)[0]
    path = next(
        iter(en.string_posterior(en.log_softmax64(logits), toy, start, sorted(toy.finals), L))
    )
    x = torch.tensor(logits, dtype=torch.float32)
    for i, tok in enumerate(path):
        x[i, tok] = float(x[i].max()) + 1.0
    assert [int(v) for v in _map(x.to(DEV), toy, start, L).tokens[:L].cpu()] == list(path)
    assert hit_out > 0, "fixture: no instance with an invalid argmax"


# --------------------------------------------------------------------------- compensated arithmetic, served path


def _log_matmul64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = a[..., :, :, None] + b[..., None, :, :]
    m = s.max(axis=-2)
    return m + np.log(np.exp(s - m[..., None, :]).sum(axis=-2))


@pytest.mark.parametrize("path", ["eager", "compiled"])
def test_compensated_log_matmul_on_the_served_path(path: str) -> None:
    """T-0065 ``test_core_compensated`` ``wide`` regime on CUDA: entries in [-24, -16], 30 % sentinels,
    inner 128. Compensated: max ≤ 1.0 ulp and ≤ 5 % of entries > 0.51 ulp; the uncompensated control on the
    same input must exceed 5 % (else the input does not discriminate)."""
    semiring = core("semiring")
    g = np.random.default_rng(0)
    a = g.uniform(-24, -16, (64, 128))
    b = g.uniform(-24, -16, (128, 64))
    a[g.random(a.shape) < 0.3] = semiring.SENTINEL
    b[g.random(b.shape) < 0.3] = semiring.SENTINEL
    ta = torch.tensor(a, dtype=torch.float32, device=DEV)
    tb = torch.tensor(b, dtype=torch.float32, device=DEV)
    # T-0073 D0: the oracle is built from the fp32-rounded inputs the core receives (as test_core_compensated.py
    # does). Built from the unrounded fp64 inputs, input rounding alone is ~0.5 ulp of the result and the row
    # failed identically on CPU (1.13 ulp / 9.5 %).
    want = _log_matmul64(ta.double().cpu().numpy(), tb.double().cpu().numpy())
    fn = semiring.log_matmul if path == "eager" else torch.compile(semiring.log_matmul)
    ulp = np.spacing(np.abs(want).astype(np.float32)).astype(np.float64)
    live = want > -1e20

    def err(res: torch.Tensor) -> np.ndarray:
        return (np.abs(res.double().cpu().numpy() - want) / ulp)[live]

    e_c = err(fn(ta, tb, compensated=True))
    # ruling P7 (T-0077c O9): on CUDA compensated=False runs the compensated kernel; the control is the torch path
    e_u = err(semiring.log_matmul(ta, tb, compensated=False, kernels=False))
    assert float(np.mean(e_u > 0.51)) > 0.05, (
        "control: the input does not discriminate on this device"
    )
    assert float(np.max(e_c)) <= 1.0 and float(np.mean(e_c > 0.51)) <= 0.05, (
        f"{path}: compensation lost (max {np.max(e_c):.2f} ulp, {np.mean(e_c > 0.51):.1%} > 0.51 ulp)"
    )


@pytest.mark.parametrize("path", ["eager", "compiled"])
def test_compensated_sum_on_the_served_path(path: str) -> None:
    semiring = core("semiring")
    g = np.random.default_rng(1)
    big = g.uniform(1e3, 1e4, 4096)
    x = np.concatenate([big, -big, g.uniform(-1, 1, 4096)])
    g.shuffle(x)
    x32 = x.astype(np.float32)
    want = math.fsum(float(v) for v in x32)
    fn = semiring.compensated_sum if path == "eager" else torch.compile(semiring.compensated_sum)
    got = float(fn(torch.tensor(x32, device=DEV)))
    naive = float(torch.tensor(x32, device=DEV).sum())
    assert abs(got - want) <= 4 * np.spacing(np.float32(max(abs(want), 1.0))), (got, want)
    assert abs(naive - want) > abs(got - want), (
        "control: plain sum is no worse here; input does not discriminate"
    )


# --------------------------------------------------------------------------- memory (T-0067 criterion 12)


def test_end_projection_convergence_step_peak_added_memory() -> None:
    """CL 256, one constrained request (a hand-built chain, bucket 128), stability_threshold 1: the peak of the
    converging sampler call under CD minus the peak of the same call under the stock sampler ≤ 1.5 GiB."""
    from tests.cd import _wire_support as ws
    from tests.cd.fakes.diffusion_sampler import DEVICE, Cfg, World, random_live_logits
    from tests.cd.unit._sampler_support import cd_xargs

    assert DEVICE.type == "cuda", "run with ASTRA_CD_DEVICE=cuda"

    class _G:
        pass

    g = _G()
    compiled = ws.with_bucket(128)
    g.payload = ws.encode(compiled)  # type: ignore[attr-defined]
    cfg = Cfg(CL=256, max_num_reqs=4, stability_threshold=1)
    live = list(range(1000, 1128))

    def fn(call: int, req: str, n: int) -> torch.Tensor:
        return random_live_logits(5, live, n)

    peaks = {}
    for cd in (False, True):
        w = World(cfg, cd=cd, seed=0)
        w.add("c", xargs=cd_xargs(g, "mem-c") if cd else None, max_tokens=4096)  # type: ignore[arg-type]
        w.prefill()
        w.decode(fn)  # step 1 (denoise)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        snap = w.decode(fn)  # converging step (history stable, always confident)
        torch.cuda.synchronize()
        assert snap.row("is_encoder_phase", "c").item(), "fixture: this call must converge"
        peaks[cd] = torch.cuda.max_memory_allocated() - base
    added = peaks[True] - peaks[False]
    assert added <= 1.5 * (1 << 30), (
        f"added peak {added / (1 << 30):.2f} GiB > 1.5 GiB (CL 256, bucket 128)"
    )


# --------------------------------------------------------------------------- the CUDA lag path (review, 2026-09-30)


def _warm_up_stock_compiled_step(cfg: Any, logits_fn: Any, calls: int = 12) -> None:
    """Compile the image's ``torch.compile``d stock sample step once, with sync-debug **off**, before any measured call.

    T-0067 tier-2 re-run bdcb230: with ``set_sync_debug_mode("error")`` already on at the first sampler call, inductor
    compiled ``_compiled_sample_step`` inside the measured call and its graph passes synchronise, so the rows died
    with ``BackendCompilerFailed`` before measuring anything. A real engine compiles at warm-up.

    The warm-up is a **stock** engine (``cd=False``) with no constrained request: no CD code runs here, so no CD-side
    first-call work (caches, uploads) is warmed, and every CD-side sync on the measured calls still raises. It has
    the measured runs' ``Cfg`` and batch (three requests, same logits function, enough calls to cross denoise,
    converge and commit), so the measured calls reuse the compiled graph; a recompile inside a measured call
    would still raise under sync-debug "error" — loud, never masked."""
    from tests.cd.fakes.diffusion_sampler import World

    assert torch.cuda.get_sync_debug_mode() == 0, "warm-up must run with sync-debug off"
    w = World(cfg, cd=False, seed=0)
    for r in ("c1", "c2", "u"):
        w.add(r)
    w.prefill()
    for _ in range(calls):
        w.decode(logits_fn)
    torch.cuda.synchronize()


def test_lagged_reports_give_the_same_records_as_a_synced_run(caplog: Any, capsys: Any) -> None:
    """~50 calls with no host sync between them (``set_sync_debug_mode("error")`` around every sampler call),
    near max_tokens so the budget depends on unread reports; the records (after a final drain) equal those of
    an identical run that synchronises after every call."""
    import contextlib

    from tests.cd.fakes.diffusion_sampler import DEVICE, Cfg, World
    from tests.cd.unit._sampler_support import G1, Grammar, Records, cd_xargs
    from tests.cd.unit._scenario import canvas_logits

    assert DEVICE.type == "cuda", "run with ASTRA_CD_DEVICE=cuda"
    g = Grammar(G1, "t0067-cuda-lag")
    fn = canvas_logits({"c1": 91, "c2": 92, "u": 93}, period=3)

    def run(synced: bool, tag: str) -> list[dict[str, Any]]:
        records = Records(caplog, capsys)
        w = World(Cfg(CL=4, stability_threshold=1), cd=True, seed=0)

        @contextlib.contextmanager
        def strict() -> Any:
            torch.cuda.set_sync_debug_mode("error")
            try:
                yield
            finally:
                torch.cuda.set_sync_debug_mode(0)

        if not synced:
            w.call_ctx = strict
        w.add("c1", xargs=cd_xargs(g, f"{tag}-1"), max_tokens=10)
        w.add("c2", xargs=cd_xargs(g, f"{tag}-2"), max_tokens=4096)
        w.add("u")
        w.prefill()
        for _ in range(50):
            w.decode(fn)
            if synced:
                torch.cuda.synchronize()
        w.call_ctx = contextlib.nullcontext
        torch.cuda.synchronize()
        for _ in range(3):  # drain
            w.decode(fn)
            torch.cuda.synchronize()
        out = []
        for r in records.all():
            t = r.get("record_tag") or ""
            if t.startswith(tag):
                out.append({**r, "record_tag": t[len(tag) :]})
        return out

    _warm_up_stock_compiled_step(Cfg(CL=4, stability_threshold=1), fn)
    lagged = run(False, "lag")
    synced = run(True, "syn")
    assert not [r for r in lagged if r.get("error") == "internal"], (
        "a CD-side host sync raised under sync-debug 'error' and S9 ended the request 'internal' "
        f"(acceptance 8ca4a53: sampler.py forced-EOS write-back): {[r for r in lagged if r.get('error') == 'internal'][:2]}"
    )
    assert lagged and lagged == synced


#: ~35-50 ms on a 5090 per call: far longer than the host takes to issue the next call, so the stream falls
#: further behind at every call and the report copy of call N is still queued when call N+1 flushes.
_LAG_SLEEP_CYCLES = 100_000_000


def test_lagged_reports_really_lag_and_give_the_same_records(
    caplog: Any, capsys: Any, monkeypatch: Any
) -> None:
    """Ruling S14: the lag row must actually lag. ``World.decode_unsynced`` makes no host read of device
    memory between calls (no snapshot; entry phases stay on the device; pinned non-blocking uploads) and a
    ``torch.cuda._sleep`` is enqueued after every sampler call, so the device-to-host report copy is still in
    flight when the next call flushes. Every ``torch.cuda.Event`` is spied: at least one ``query()`` must
    return False (otherwise the lag path was never exercised). The records (after a synchronised drain and
    the releases) equal those of the identical run that synchronises after every call."""
    import contextlib

    from tests.cd.fakes.diffusion_sampler import DEVICE, Cfg, World
    from tests.cd.unit._sampler_support import G1, Grammar, Records, cd_xargs
    from tests.cd.unit._scenario import canvas_logits

    assert DEVICE.type == "cuda", "run with ASTRA_CD_DEVICE=cuda"
    g = Grammar(G1, "t0067-cuda-reallag")
    fn = canvas_logits({"c1": 91, "c2": 92, "u": 93}, period=3)

    queries: list[bool] = []
    real_event = torch.cuda.Event

    class SpyEvent(real_event):  # type: ignore[misc,valid-type]
        def query(self) -> bool:
            done = bool(super().query())
            queries.append(done)
            return done

    monkeypatch.setattr(torch.cuda, "Event", SpyEvent)

    def run(lag: bool, tag: str) -> list[dict[str, Any]]:
        records = Records(caplog, capsys)
        w = World(Cfg(CL=4, stability_threshold=1), cd=True, seed=0)

        @contextlib.contextmanager
        def lagging() -> Any:
            torch.cuda.set_sync_debug_mode("error")
            try:
                yield
            finally:
                torch.cuda.set_sync_debug_mode(0)
            torch.cuda._sleep(_LAG_SLEEP_CYCLES)  # after the call: keeps the stream behind the host

        w.add("c1", xargs=cd_xargs(g, f"{tag}-1"), max_tokens=10)
        w.add("c2", xargs=cd_xargs(g, f"{tag}-2"), max_tokens=4096)
        w.add("u")
        w.prefill()
        torch.cuda.synchronize()
        if lag:
            w.call_ctx = lagging
        for _ in range(50):
            w.decode_unsynced(fn)
            if not lag:
                torch.cuda.synchronize()
        w.call_ctx = contextlib.nullcontext
        torch.cuda.synchronize()
        for _ in range(3):  # drain
            w.decode_unsynced(fn)
            torch.cuda.synchronize()
        for r in list(w.reqs):
            w.remove(r)  # S11: release flushes
        out = []
        for r in records.all():
            t = r.get("record_tag") or ""
            if t.startswith(tag):
                out.append({**r, "record_tag": t[len(tag) :]})
        return out

    _warm_up_stock_compiled_step(Cfg(CL=4, stability_threshold=1), fn)
    queries.clear()  # only the measured (lagged) run's event queries count as the lag proof
    lagged = run(True, "lag")
    lag_queries = list(queries)
    assert lag_queries, (
        "no torch.cuda.Event.query() call at all: the report path is not event-gated"
    )
    assert False in lag_queries, (
        f"all {len(lag_queries)} event queries returned True: the reports never lagged, the row proves nothing"
    )
    synced = run(False, "syn")
    assert lagged, "no records from the lagged run"
    assert not [r for r in lagged if r.get("error") == "internal"], (
        "a CD-side host sync raised under sync-debug 'error' and S9 ended the request 'internal' "
        f"(acceptance 8ca4a53: sampler.py forced-EOS write-back): {[r for r in lagged if r.get('error') == 'internal'][:2]}"
    )
    assert lagged == synced
