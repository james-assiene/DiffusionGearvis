"""T-0077c on CUDA (tier 2, ``-m gpu``, ``/launch``): the fused ``log_matmul`` kernel, plain and compensated.

Run **inside the pinned container** through T-0067's runner, twice, by ``tests/contract/test_vllm_cd_logmatmul_gpu.py``
(T-0077b wrapper form: ``/repo:ro``, ``--network none``, ``--noconftest -o addopts= -p no:cacheprovider``,
``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``, ``ASTRA_CD_REQUIRE_CUDA=1``, the four dev automata compiled on the host and
mounted at ``$ASTRA_CD_PAYLOADS``). Without CUDA the rows skip, and fail under ``ASTRA_CD_REQUIRE_CUDA=1``. API: the
ruling in ``tests/cd/unit/_frozen_t0077c.py``. "Dispatched" is observed (a Python frame of
``cd/kernels/logmatmul.py``), never assumed.

Rows (``LOGMATMUL_CUDA_PASSED`` = K_c = 43):
- criterion 2 (2): at ``[128, 400, 400] ⊗ [128, 400, 400]`` the peak added by one entry call minus the output's bytes
  is < one ``CHUNK_ELEMS`` fp32 chunk (16 MiB), plain and compensated;
- criterion 3a (8): entry and dispatched ``semiring.log_matmul`` (O9: the compensated spec whatever ``compensated`` says;
  every semiring call reaches the entry with ``compensated=True``) bit-identical (fp32 patterns, NaN positions) to
  ``spec_log_matmul`` run on CUDA with torch's CUDA exp/log; groups ``inner`` (m ∈ {1, 2, 3, 31, 32, 33, 91, 205, 400,
  512}), ``core_shapes`` (strided live-block views, vec-mat, mat-vec, broadcast batch), ``dev_automata`` (level-0
  products ``[L/2, S, S] ⊗ [L/2, S, S]`` of the real normalised leaves and vec-mats ``[L, 1, S] ⊗ [L, S, S]``),
  ``adversarial`` (regimes, two-band, all-sentinel, NaN/±inf, near-DEAD, near-tie pins) × plain/compensated. A
  difference names the instance, the count of differing entries and the max ulp: no tolerance;
- criterion 3b (2): the same bits under every ``(block_n, block_k)`` tried;
- criterion 4 (8): compensated wide and gauged cells at inner {32, 91, 205, 400, 512}, outer 64, seeds 0/1: the max passes at
  the bar **or** at torch's own CUDA max on the same input (O8, human 2026-10-06), the gauged mean / wide share > 0.51 ulp
  stays gated at the bar; semiring plain on CUDA (the compensated kernel under O9) max ≤ the torch CUDA plain path on both
  regimes, mean ≤ on gauged only, wide mean printed (O6); two-band; all-sentinel; dead set ``torch.equal`` to the torch
  path's on every instance; the kernel entry's ``compensated=False`` fails the compensated bar on T-0065's inputs (power
  control, entry-level since O9). Error statistics kernel vs torch are printed (criterion 10);
- criterion 5 (15): live (kernel) vs ``core_frozen_t0077c``, core instances (11) and dev automata (4): max|dq| ≤ 1e-6,
  max|dH| ≤ 1e-5 on rows < L_eff, dead sets of ``log_a``/``log_b`` equal, |Δ log Z| and max|Δ| of live log_a/log_b
  printed; N3 ``torch.equal``: ``class_max``, max-plus tree, MAP tokens/score/empty, δ*, ``terminal_factor``,
  generator state after ``joint_draw``; draw tokens/states/empty equal (O5: a flip fails the row, never edited);
- criterion 7 (4): 20 repeats (entry and whole ``constrained_marginals``); sync debug "error" in steady state (K5);
  ``ASTRA_T0077C_DIGEST`` lines over ``DIGEST_CASES`` (host compares two processes); ``torch.compile`` == eager;
- criterion 8 (4): ``kernels=False`` runs the torch path, equal to frozen; a triton-blocked subprocess (K9 blocker)
  runs the torch path, equal to frozen; a fresh process (log_matmul, then marginals, then fp64 operands) logs exactly
  one decision record naming ``log_matmul``, "triton" and "3.7.1", plus one layout-refusal record; the regime line.

Regime: image torch 2.13.0+cu130 / triton 3.7.1 / CUDA 13.0, RTX 5090, fp32, TF32 "high" set globally (vLLM's); CL 256,
V 262 144 for the dev automata (seed 0, logits ``randn·4``, L_eff = CL); T-0065 seeds 0/1, outer 64; 20 repeats.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tests.cd.unit import _frozen_t0077 as fz
from tests.cd.unit import _frozen_t0077b as kb
from tests.cd.unit import _frozen_t0077c as c

pytestmark = pytest.mark.gpu

LOGMATMUL_CUDA_PASSED = 43
REPO = Path(__file__).resolve().parents[3]
DEV = torch.device("cuda")
BUCKETS = (64, 128, 256, 512)
CL, V = 256, 262_144
REPEATS = 20
PAYLOADS_ENV = "ASTRA_CD_PAYLOADS"
DIGEST = "ASTRA_T0077C_DIGEST"
REGIME = "ASTRA_T0077C_REGIME"
VARIANTS = {"comp": True, "plain": False}
INNER_SWEEP = (1, 2, 3, 31, 32, 33, 91, 205, 400, 512)
TILES = ((None, None), (1, 1), (2, 8), (16, 16), (32, 32), (64, 16), (128, 128))
CORE = tuple(kb.INSTANCES)
MARGINAL_FIELDS = kb.MARGINAL_FIELDS
#: digest names, in print order: entry outputs (``lm:``) and whole marginals fields (``cm:``)
LM_DIGEST_CASES = [f"lm:{r}{i}-{v}" for r in c.REGIMES for i in c.INNERS for v in VARIANTS] + [
    f"lm:dev:b{b}-{v}" for b in BUCKETS for v in VARIANTS
]
CM_DIGEST_CASES = [f"cm:core:{n}.{f}" for n in CORE for f in MARGINAL_FIELDS] + [
    f"cm:dev:b{b}.{f}" for b in BUCKETS for f in MARGINAL_FIELDS
]
DIGEST_CASES = LM_DIGEST_CASES + CM_DIGEST_CASES


@pytest.fixture(autouse=True, scope="module")
def tf32_high() -> Iterator[None]:
    if not torch.cuda.is_available():
        why = "CUDA rows run inside the pinned container (tests/contract/test_vllm_cd_logmatmul_gpu.py)"
        if os.environ.get("ASTRA_CD_REQUIRE_CUDA") == "1":
            pytest.fail(f"ASTRA_CD_REQUIRE_CUDA=1 but CUDA is not available: {why}")
        pytest.skip(why)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    yield
    torch.set_float32_matmul_precision(prev)


#: torch's own loggers that flood pytest's captured-log sections at DEBUG (``torch.__trace`` from ``torch.compile``:
#: 4 950 lines at f3c678c, which pushed the assertion text out of the wrapper's window). Ours
#: (``astra_vllm_plugin.cd.kernels``) is not touched.
NOISY_LOGGERS = ("torch.__trace", "torch._dynamo", "torch._inductor", "torch.fx")


@contextlib.contextmanager
def quiet_torch_logs() -> Iterator[None]:
    """Hold torch's tracing loggers at WARNING (``torch.__trace`` disabled) while the block runs; restored after."""
    saved = []
    for name in NOISY_LOGGERS:
        lg = logging.getLogger(name)
        saved.append((lg, lg.level, lg.propagate, lg.disabled))
        lg.setLevel(logging.WARNING)
        if name == "torch.__trace":
            # torch attaches handlers to this logger itself and runs it at DEBUG while tracing, so neither its level
            # nor ``propagate`` is enough (measured on CPU, 2026-10-06): ``disabled`` drops every record it handles
            lg.propagate = False
            lg.disabled = True
    try:
        yield
    finally:
        for lg, level, prop, dis in saved:
            lg.setLevel(level)
            lg.propagate = prop
            lg.disabled = dis


@pytest.fixture(autouse=True)
def quiet_torch_logs_per_row() -> Iterator[None]:
    with quiet_torch_logs():
        yield


def report(row: str, bad: Any) -> None:
    """One ``ASTRA_T0077C_DIFF`` line per failing item, printed before the row asserts (survives in the wrapper's
    evidence extraction whatever else floods the output)."""
    items = bad.items() if isinstance(bad, dict) else enumerate(bad)
    for k, v in items:
        print(f"ASTRA_T0077C_DIFF {row} {k}: {v}", flush=True)


def regime_stats(t: torch.Tensor, exp: np.ndarray, regime: str) -> tuple[float, float, float]:
    """(max, mean, share > 0.51) in the regime's unit: wide fp32 ulp of the oracle, gauged ulp(1.0)."""
    unit = c.ulp32(exp) if regime == "wide" else c.ULP1
    u = np.abs(t.double().cpu().numpy() - exp) / unit
    return float(u.max()), float(u.mean()), float(np.mean(u > 0.51))


def bar_line(
    row: str, regime: str, inner: int, seed: int, k: Any, t: Any, bar: str, ok: bool
) -> None:
    print(
        f"ASTRA_T0077C_BAR {row} {regime} inner {inner} seed {seed} kernel max/mean/share "
        f"{k[0]:.4f}/{k[1]:.5f}/{k[2]:.4f} torch max/mean/share {t[0]:.4f}/{t[1]:.5f}/{t[2]:.4f} bar {bar} "
        f"{'ok' if ok else 'FAIL'}",
        flush=True,
    )


BARS = {"wide": "max<=1.0,share<=0.05", "gauged": "mean<=0.75,max<=2.5"}


# --------------------------------------------------------------------------- helpers


def cu(t: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Move operands to CUDA keeping their view structure (strides, stride-0 batch): the base moves, the view is
    rebuilt with ``as_strided``."""
    out = []
    for x in t:
        base = x._base if x._base is not None else x
        b = base.to(DEV)
        out.append(b.as_strided(x.shape, x.stride(), x.storage_offset() - base.storage_offset()))
    return out[0], out[1]


def spec_cuda(a3: torch.Tensor, b3: torch.Tensor, comp: bool) -> torch.Tensor:
    return c.spec_log_matmul(a3, b3, compensated=comp)  # torch's CUDA exp/log on a3's device


def entry_dispatched(a3: torch.Tensor, b3: torch.Tensor, comp: bool, **kw: Any) -> torch.Tensor:
    with c.entry_calls() as seen:
        out = c.lm_entry()(a3, b3, compensated=comp, **kw)
        torch.cuda.synchronize()
    assert seen, "the entry ran no frame of cd/kernels/logmatmul.py"
    return out


@contextlib.contextmanager
def entry_variants() -> Iterator[list[Any]]:
    """The ``compensated`` argument of every call into the entry ``log_matmul`` of ``cd/kernels/logmatmul.py`` while the
    block runs (read from the entry frame's arguments at the call event; API ruling: the entry takes ``compensated``)."""
    target = str((c.KERNELS_DIR / "logmatmul.py").resolve())
    seen: list[Any] = []

    def prof(frame: Any, event: str, arg: Any) -> None:
        code = frame.f_code
        if (
            event == "call"
            and code.co_name == "log_matmul"
            and code.co_filename.endswith("logmatmul.py")
            and str(Path(code.co_filename).resolve()) == target
        ):
            seen.append(frame.f_locals.get("compensated", "<missing>"))
        # one profiler per thread: chain, so an enclosing spy (entry_calls) keeps seeing calls
        if prev is not None:
            prev(frame, event, arg)

    prev = sys.getprofile()
    sys.setprofile(prof)
    try:
        yield seen
    finally:
        sys.setprofile(prev)


def semiring_dispatched(a: torch.Tensor, b: torch.Tensor, comp: bool) -> torch.Tensor:
    """``semiring.log_matmul(compensated=comp, kernels=True)`` on CUDA: it must reach the kernel entry, and (ruling O9,
    human 2026-10-06) always with ``compensated=True``, ``comp=False`` included."""
    with c.entry_calls() as seen, entry_variants() as variants:
        out = c.live("semiring").log_matmul(a, b, compensated=comp, kernels=True)
        torch.cuda.synchronize()
    assert seen, "semiring.log_matmul(kernels=True) on CUDA did not dispatch the kernel"
    assert variants and all(v is True for v in variants), (
        f"semiring.log_matmul(compensated={comp}) reached the entry with compensated={variants} (O9: always True)"
    )
    return out


def torch_ref(a: torch.Tensor, b: torch.Tensor, comp: bool) -> torch.Tensor:
    return c.frozen("semiring").log_matmul(a, b, compensated=comp)


@functools.cache
def _payload_dir() -> Path:
    d = os.environ.get(PAYLOADS_ENV)
    assert d and (Path(d) / "manifest.json").is_file(), (
        f"${PAYLOADS_ENV} must name the host-compiled dev automata; got {d!r}"
    )
    return Path(d)


@functools.cache
def dev_world(bucket: int) -> dict[str, Any]:
    """The A/B's inputs for one dev automaton (tables from the wire payload, CL 256, V 262 144, seed 0)."""
    from astra_vllm_plugin.cd import automaton as cd_automaton
    from astra_vllm_plugin.cd import budget
    from astra_vllm_plugin.cd import requests as cd_requests

    d = _payload_dir()
    manifest = json.loads((d / "manifest.json").read_text())[str(bucket)]
    wa = cd_automaton.decode_payload((d / f"b{bucket}.b64").read_text())
    tables = cd_requests._tables_for(manifest["payload_sha256"], wa, DEV)
    sv = np.zeros(bucket, dtype=bool)
    sv[: len(wa.start_vector)] = np.asarray(wa.start_vector, dtype=bool)[:bucket]
    g = torch.Generator(device="cpu").manual_seed(0)
    x = (torch.randn(CL, V, generator=g) * 4.0).to(DEV)
    return {
        "tables": tables,
        "x": x,
        "a": torch.from_numpy(sv).to(DEV),
        "b": budget.terminal_factor(tables, 0),
        "l_eff": CL,
        "seed": 0,
        "S": int(tables.n_states),
    }


@functools.cache
def core_world(name: str) -> dict[str, Any]:
    inst = fz.build_inst(name)
    tables = kb.tables_on(c.live("emission"), inst, DEV)
    return {
        "tables": tables,
        "x": inst.logits.to(DEV),
        "a": kb._mask(inst.start, inst.bucket, DEV),
        "b": kb._mask(inst.end, inst.bucket, DEV),
        "l_eff": inst.l_eff,
        "seed": inst.draw_seed,
        "S": int(tables.n_states),
    }


def dev_operands(bucket: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Level-0 products and vec-mats of the real (live, normalised) leaves, on the live block ``[:, :S, :S]``."""
    w = dev_world(bucket)
    em, tr = c.live("emission"), c.live("tree")
    cm = em.class_mass(w["x"], w["tables"])
    with held_log_matmul():
        tree = tr.up_sweep(em.log_transfer(w["x"], w["tables"], CL, class_mass=cm))
    S = w["S"]
    lv = tree.levels[0][:, :S, :S]
    h = lv.shape[0] // 2
    return {
        f"dev_b{bucket}_level0": (lv[0 : 2 * h : 2], lv[1 : 2 * h : 2]),
        f"dev_b{bucket}_vecmat": (tree.levels[0][:, 0:1, :S], lv),
    }


@contextlib.contextmanager
def held_log_matmul() -> Iterator[None]:
    """Hold the live ``tree``/``marginals`` products on the frozen torch ``log_matmul`` (operand building only)."""
    tr, mg = c.live("tree"), c.live("marginals")
    saved = (tr.log_matmul, mg.log_matmul)
    tr.log_matmul = mg.log_matmul = c.frozen("semiring").log_matmul
    try:
        yield
    finally:
        tr.log_matmul, mg.log_matmul = saved


def groups() -> dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    g: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    gen = torch.Generator().manual_seed(4242)
    inner = {}
    for m in INNER_SWEEP:
        a = (-2.35 - torch.rand(2, 33, m, generator=gen)).float()
        b = (-2.35 - torch.rand(2, m, 40, generator=gen)).float()
        a[torch.rand(2, 33, m, generator=gen) < 0.3] = c.SENTINEL32
        b[torch.rand(2, m, 40, generator=gen) < 0.3] = c.SENTINEL32
        inner[f"inner{m}"] = (a, b)
    g["inner"] = inner
    g["core_shapes"] = {
        "live_block_s33": c.live_block_views(3, 33, 40, 0),
        "live_block_s205": c.live_block_views(4, 205, 256, 1),
        "live_block_s400": c.live_block_views(2, 400, 512, 2),
        "vec_mat": c.vec_mat(5, 91, 128, 0),
        "mat_vec": c.mat_vec(5, 91, 128, 0),
        "broadcast": c.broadcast_batch(4, 33, 91, 17, 0),
    }
    g["adversarial"] = {
        **{f"{r}{i}": c.regime_instance(r, i, 0) for r in c.REGIMES for i in c.INNERS},
        "two_band91": c.two_band(91, 0),
        "two_band205": c.two_band(205, 1),
        "all_sentinel33": c.all_sentinel(33, 0),
        "nan_inf33": c.nan_inf(33, 0),
        "near_dead": c.near_dead(),
        **{f"near_tie{m}": c.near_tie_pinned(m) for m in c.NEAR_TIE_PINS},
    }
    return g


# --------------------------------------------------------------------------- criterion 2


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_criterion2_fusion_peak(variant: str) -> None:
    comp = VARIANTS[variant]
    g = torch.Generator(device="cpu").manual_seed(2)
    a = (-3 * torch.rand(128, 400, 400, generator=g)).to(DEV)
    b = (-3 * torch.rand(128, 400, 400, generator=g)).to(DEV)
    entry_dispatched(a, b, comp)  # compile outside the measured window
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = entry_dispatched(a, b, comp)
    added = torch.cuda.max_memory_allocated() - before
    extra = added - out.numel() * out.element_size()
    print(
        f"ASTRA_T0077C_PEAK {variant} added {added} bytes, minus output {extra} bytes at [128,400,400]^2"
    )
    assert extra < c.CHUNK_BYTES, f"peak added minus output {extra / 2**20:.2f} MiB ≥ 16 MiB"


# --------------------------------------------------------------------------- criterion 3


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("group", ["inner", "core_shapes", "dev_automata", "adversarial"])
def test_criterion3a_spec_bits(group: str, variant: str) -> None:
    comp = VARIANTS[variant]
    if group == "dev_automata":
        cases = {k: v for b in BUCKETS for k, v in dev_operands(b).items()}
    else:
        cases = {k: cu(v) for k, v in groups()[group].items()}
    bad = {}
    for name, (a3, b3) in cases.items():
        want = spec_cuda(a3, b3, comp)
        d = c.bit_diff(want, entry_dispatched(a3, b3, comp))
        if d is not None:
            bad[f"{name} entry"] = d
        # O9: on CUDA semiring dispatches the compensated variant whatever ``compensated`` says
        d = c.bit_diff(spec_cuda(a3, b3, True), semiring_dispatched(a3, b3, comp))
        if d is not None:
            bad[f"{name} semiring"] = d
    report(f"3a {group}-{variant}", bad)
    assert bad == {}, json.dumps(bad, indent=1)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_criterion3b_tiles(variant: str) -> None:
    comp = VARIANTS[variant]
    g = groups()
    cases = {
        "inner91": g["inner"]["inner91"],
        "live_block_s33": g["core_shapes"]["live_block_s33"],
        "broadcast": g["core_shapes"]["broadcast"],
        "near_tie33": g["adversarial"]["near_tie33"],
        "nan_inf33": g["adversarial"]["nan_inf33"],
    }
    bad = {}
    for name, ops in cases.items():
        a3, b3 = cu(ops)
        want = spec_cuda(a3, b3, comp)
        for bn, bk in TILES:
            d = c.bit_diff(want, entry_dispatched(a3, b3, comp, block_n=bn, block_k=bk))
            if d is not None:
                bad[f"{name} {bn}x{bk}"] = d
    report(f"3b {variant}", bad)
    assert bad == {}, bad


# --------------------------------------------------------------------------- criterion 4


CELLS = [(r, i, s) for r in c.REGIMES for i in c.INNERS for s in c.SEEDS]


@pytest.mark.parametrize("regime", list(c.REGIMES))
def test_criterion4_compensated_bars(regime: str) -> None:
    bad = []
    for r, i, s in CELLS:
        if r != regime:
            continue
        a, b = cu(c.regime_instance(r, i, s))
        exp = c.regime_oracle(r, i, s)
        got = semiring_dispatched(a, b, True)
        k, t = regime_stats(got, exp, r), regime_stats(torch_ref(a, b, True), exp, r)
        # O8 (human, 2026-10-06): the max passes at the bar OR at torch's own CUDA max on the same input; the mean
        # (gauged) / the share > 0.51 ulp (wide) stays gated at the bar
        max_bar = 1.0 if r == "wide" else 2.5
        clause = "bar" if k[0] <= max_bar else ("torch" if k[0] <= t[0] else "none")
        second_ok = k[2] <= 0.05 if r == "wide" else k[1] <= 0.75
        ok = clause != "none" and second_ok
        bar_line("comp_bars", r, i, s, k, t, BARS[r], ok)
        print(
            f"ASTRA_T0077C_BAR comp_bars_O8 {r} inner {i} seed {s} kernel_max {k[0]:.4f} bar_max {max_bar} "
            f"torch_max {t[0]:.4f} max_clause {clause} {'share' if r == 'wide' else 'mean'}_ok {second_ok}",
            flush=True,
        )
        if not ok:
            bad.append(
                f"{r} inner {i} seed {s}: kernel max/mean/share {k} torch {t} (max clause {clause}, second gate {second_ok})"
            )
    report("test_criterion4_compensated_bars", bad)
    assert not bad, bad


@pytest.mark.parametrize("regime", list(c.REGIMES))
def test_criterion4_plain_no_worse_than_torch_cuda(regime: str) -> None:
    """O6 as ruled by the human (2026-10-06): per (regime, inner, seed) max ≤ torch's CUDA plain path on every regime;
    mean ≤ gated on gauged only; on wide the mean is reported (printed), not gated. Unit: wide fp32 ulp of the oracle,
    gauged ulp(1.0); share > 0.51 ulp printed."""
    bad = []
    for r, i, s in CELLS:
        if r != regime:
            continue
        a, b = cu(c.regime_instance(r, i, s))
        exp = c.regime_oracle(r, i, s)
        unit = c.ulp32(exp) if r == "wide" else c.ULP1

        def st(
            t: torch.Tensor, exp: np.ndarray = exp, unit: Any = unit
        ) -> tuple[float, float, float]:
            u = np.abs(t.double().cpu().numpy() - exp) / unit
            return float(u.max()), float(u.mean()), float(np.mean(u > 0.51))

        k, t = st(semiring_dispatched(a, b, False)), st(torch_ref(a, b, False))
        ok_cell = k[0] <= t[0] and (r != "gauged" or k[1] <= t[1])
        bar_line(
            "plain_vs_torch",
            r,
            i,
            s,
            k,
            t,
            "max<=torch" + (",mean<=torch" if r == "gauged" else ""),
            ok_cell,
        )
        if not k[0] <= t[0]:
            bad.append(f"{r} inner {i} seed {s}: kernel max {k[0]} > torch max {t[0]}")
        if r == "gauged" and not k[1] <= t[1]:  # O6: wide mean reported above, not gated
            bad.append(f"{r} inner {i} seed {s}: kernel mean {k[1]} > torch mean {t[1]}")
    report("test_criterion4_plain_no_worse_than_torch_cuda", bad)
    assert not bad, bad


def test_criterion4_two_band() -> None:
    bad = []
    for m in (32, 91, 205, 400):
        for s in c.SEEDS:
            a, b = c.two_band(m, s)
            exp = c.lse_oracle(a, b)
            live = exp > -1e29
            g = semiring_dispatched(*cu((a, b)), True).double().cpu().numpy()
            spurious = int((live & ~(np.isfinite(g) & (g > c.DEAD32))).sum())
            u = np.abs(g[live] - exp[live]) / c.ulp32(exp[live])
            print(
                f"ASTRA_T0077C_BAR two_band m {m} seed {s} spurious_dead {spurious} max {u.max():.4f} "
                f"share {np.mean(u > 0.51):.4f} bar max<=1.0,share<=0.05,spurious==0",
                flush=True,
            )
            if spurious or u.max() > 1.0 or np.mean(u > 0.51) > 0.05:
                bad.append(
                    f"m {m} seed {s}: {spurious} spurious dead, max {u.max():.3f}, share {np.mean(u > 0.51):.2%}"
                )
    report("test_criterion4_two_band", bad)
    assert not bad, bad


def test_criterion4_all_sentinel_rows_and_columns() -> None:
    sent = torch.tensor(c.SENTINEL32).view(torch.int32)
    for m in (1, 3, 33, 400):
        a, b = c.all_sentinel(m, 0)
        dead = torch.from_numpy(c.lse_oracle(a, b) <= -1e29)
        for comp in (True, False):
            out = semiring_dispatched(*cu((a, b)), comp).cpu()
            print(
                f"ASTRA_T0077C_BAR all_sentinel m {m} comp {comp} dead_not_sentinel "
                f"{int((out.view(torch.int32)[dead] != sent).sum())} live_came_out_dead {int((out[~dead] <= c.DEAD32).sum())}",
                flush=True,
            )
            assert bool((out.view(torch.int32)[dead] == sent).all()), (m, comp)
            assert bool((out[~dead] > c.DEAD32).all()), (m, comp)


def test_criterion4_dead_set_equals_the_torch_path() -> None:
    bad = []
    insts = {
        **{
            k: v
            for gname in ("inner", "core_shapes", "adversarial")
            for k, v in groups()[gname].items()
        }
    }
    for name, ops in insts.items():
        a, b = cu(ops)
        for comp in (True, False):
            k, t = semiring_dispatched(a, b, comp), torch_ref(a, b, comp)
            if not torch.equal(k == c.SENTINEL32, t == c.SENTINEL32):
                n = int(((k == c.SENTINEL32) ^ (t == c.SENTINEL32)).sum())
                bad.append(f"{name} comp={comp}: {n} entries")
    for bkt in BUCKETS:
        for name, (a, b) in dev_operands(bkt).items():
            for comp in (True, False):
                if not torch.equal(
                    semiring_dispatched(a, b, comp) == c.SENTINEL32,
                    torch_ref(a, b, comp) == c.SENTINEL32,
                ):
                    bad.append(f"{name} comp={comp}")
    report("4_dead_set", bad)
    assert not bad, bad


def test_criterion4_plain_kernel_fails_the_compensated_bar() -> None:
    """Power control (umbrella criterion 8): the kernel with ``compensated=False`` fails the compensated bar on
    T-0065's inputs (wide inner 64/128, gauged inner 300, seeds 0/1)."""
    passed = []
    for regime, inner in (("wide", 64), ("wide", 128), ("gauged", 300)):
        lo, hi = c.REGIMES[regime]
        for s in c.SEEDS:
            a, b = c.t0065_operands(lo, hi, inner, s)
            a3, b3 = a.unsqueeze(0), b.unsqueeze(0)
            exp = c.lse_oracle(a3, b3)
            # O9: semiring no longer runs the plain kernel on CUDA; the power control is the kernel entry's plain variant
            ok, msg = (c.wide_ok if regime == "wide" else c.gauged_ok)(
                entry_dispatched(*cu((a3, b3)), False), exp
            )
            print(
                f"ASTRA_T0077C_BAR plain_power {regime} inner {inner} seed {s} {msg} (must FAIL the bar)",
                flush=True,
            )
            if ok:
                passed.append(f"{regime} {inner} seed {s}: {msg}")
    assert not passed, f"plain kernel meets the compensated bar: {passed}"


# --------------------------------------------------------------------------- criterion 5


def pipeline(which: str, w: dict[str, Any]) -> dict[str, Any]:
    get = c.frozen if which == "frozen" else c.live
    em, tr, dr, md, mg = (get(n) for n in ("emission", "tree", "draw", "mapdecode", "marginals"))
    from astra_vllm_plugin.cd import budget

    x, tables, a, b, l_eff = w["x"], w["tables"], w["a"], w["b"], w["l_eff"]
    S = w["S"]
    cm = em.class_mass(x, tables)
    cw = em.class_log_weights_from(cm)
    cmax = md.class_max_of(x, tables)
    log_m = em.log_transfer(x, tables, l_eff, class_mass=cm)
    gen = torch.Generator(device=DEV).manual_seed(w["seed"])
    with c.entry_calls() as seen, entry_variants() as variants:
        tree = tr.up_sweep(log_m)
        d = dr.joint_draw(x, tables, tree, a, b, l_eff, generator=gen, class_log_weights=cw)
        marg = mg.constrained_marginals(x, tables, tree, a, b, l_eff, class_mass=cm)
        torch.cuda.synchronize()
    mp = md.constrained_map(x, tables, a, b, l_eff, class_max=cmax)
    ds = budget.delta_star(tables, a, d.tokens[:l_eff], [])
    return {
        "dispatched": len(seen),
        "variants": list(variants),
        "cmax": cmax,
        "maxtree": md.max_up_sweep(log_m[:, :S, :S]),
        "map": mp,
        "draw": d,
        "gen": gen.get_state(),
        "marg": marg,
        "delta_star": (ds.states, ds.empty),
        "terminal": budget.terminal_factor(tables, 0),
    }


def check_criterion5(w: dict[str, Any], label: str) -> None:
    f, v = pipeline("frozen", w), pipeline("live", w)
    assert v["dispatched"] > 0, f"{label}: the live path never entered cd/kernels/logmatmul.py"
    assert v["variants"] and all(x is True for x in v["variants"]), (
        f"{label}: live entry calls with compensated={sorted(set(map(str, v['variants'])))} (O9: always True on CUDA)"
    )
    eq = lambda x, y: all(torch.equal(p, q) for p, q in zip(x, y, strict=True))  # noqa: E731
    n3 = {
        "class_max": eq(f["cmax"], v["cmax"]),
        "maxplus_tree": eq(f["maxtree"], v["maxtree"]),
        "map_tokens": torch.equal(f["map"].tokens, v["map"].tokens),
        "map_score": torch.equal(f["map"].score, v["map"].score),
        "map_empty": torch.equal(f["map"].empty, v["map"].empty),
        "delta_star": eq(f["delta_star"], v["delta_star"]),
        "terminal_factor": torch.equal(f["terminal"], v["terminal"]),
        "generator_state": torch.equal(f["gen"], v["gen"]),
        "draw_tokens": torch.equal(f["draw"].tokens, v["draw"].tokens),
        "draw_states": torch.equal(f["draw"].states, v["draw"].states),
        "draw_empty": torch.equal(f["draw"].empty, v["draw"].empty),
    }
    mf, ml = f["marg"], v["marg"]
    L = w["l_eff"]
    dq = float((mf.q[:L].double() - ml.q[:L].double()).abs().max())
    dh = float((mf.entropy[:L].double() - ml.entropy[:L].double()).abs().max())
    dead = {
        k: torch.equal(getattr(mf, k) <= c.DEAD32, getattr(ml, k) <= c.DEAD32)
        for k in ("log_a", "log_b")
    }
    rep = {"dlogz": float((mf.log_z.double() - ml.log_z.double()).abs())}
    for k in ("log_a", "log_b"):
        lv = getattr(mf, k) > c.DEAD32
        rep[f"d{k}"] = (
            float((getattr(mf, k).double() - getattr(ml, k).double()).abs()[lv].max())
            if bool(lv.any())
            else 0.0
        )
    print(f"ASTRA_T0077C_N4 {label} dq {dq:.3g} dH {dh:.3g} {json.dumps(rep)}")
    bad = {
        k: ok
        for k, ok in {**n3, **{f"dead_{k}": o for k, o in dead.items()}}.items()
        if ok is not True
    }
    report(f"5 {label}", bad)
    assert not bad, f"{label}: {bad}"
    assert dq <= 1e-6 and dh <= 1e-5, f"{label}: max|dq| {dq}, max|dH| {dh}"


@pytest.mark.parametrize("name", CORE)
def test_criterion5_core_instance(name: str) -> None:
    check_criterion5(core_world(name), f"core:{name}")


@pytest.mark.parametrize("bucket", BUCKETS)
def test_criterion5_dev_automaton(bucket: int) -> None:
    check_criterion5(dev_world(bucket), f"dev:b{bucket}")


# --------------------------------------------------------------------------- criterion 7


def live_marginals(w: dict[str, Any]) -> Any:
    em, tr, mg = c.live("emission"), c.live("tree"), c.live("marginals")
    cm = em.class_mass(w["x"], w["tables"])
    tree = tr.up_sweep(em.log_transfer(w["x"], w["tables"], w["l_eff"], class_mass=cm))
    return lambda: mg.constrained_marginals(
        w["x"], w["tables"], tree, w["a"], w["b"], w["l_eff"], class_mass=cm
    )


def test_criterion7_twenty_repeats_same_bits() -> None:
    bad = {}
    for name, ops in {
        "gauged512": c.regime_instance("gauged", 512, 0),
        "live_block_s205": c.live_block_views(4, 205, 256, 1),
    }.items():
        a, b = cu(ops)
        for v, comp in VARIANTS.items():
            first = entry_dispatched(a, b, comp)
            for i in range(1, REPEATS):
                d = c.bit_diff(first, entry_dispatched(a, b, comp))
                if d:
                    bad[f"{name}-{v} call {i}"] = d
    for label, w in (("core:s205_b256", core_world("s205_b256")), ("dev:b256", dev_world(256))):
        call = live_marginals(w)
        first = call()
        for i in range(1, REPEATS):
            d = kb.field_diffs(first, call())
            if d:
                bad[f"{label} call {i}"] = d
    report("test_criterion7_twenty_repeats_same_bits", bad)
    assert bad == {}, bad


def test_criterion7_no_host_sync_under_sync_debug_error() -> None:
    a, b = cu(c.live_block_views(4, 205, 256, 1))
    call = live_marginals(core_world("s205_b256"))
    for comp in (True, False):
        entry_dispatched(a, b, comp)
        semiring_dispatched(a, b, comp)
    call()
    torch.cuda.synchronize()
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with c.entry_calls() as seen:
            for comp in (True, False):
                c.lm_entry()(a, b, compensated=comp)
                c.live("semiring").log_matmul(a, b, compensated=comp, kernels=True)
            call()
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()
    assert seen, "not dispatched inside the strict window"


def test_criterion7_digests_of_the_output_bits() -> None:
    lines = []
    for r in c.REGIMES:
        for i in c.INNERS:
            a, b = cu(c.regime_instance(r, i, 0))
            for v, comp in VARIANTS.items():
                lines.append(
                    f"{DIGEST} lm:{r}{i}-{v} {c.sha256_bits(entry_dispatched(a, b, comp))}"
                )
    for bkt in BUCKETS:
        a, b = dev_operands(bkt)[f"dev_b{bkt}_level0"]
        for v, comp in VARIANTS.items():
            lines.append(
                f"{DIGEST} lm:dev:b{bkt}-{v} {c.sha256_bits(entry_dispatched(a, b, comp))}"
            )
    for label, w in [
        *((f"core:{n}", core_world(n)) for n in CORE),
        *((f"dev:b{b}", dev_world(b)) for b in BUCKETS),
    ]:
        out = live_marginals(w)()
        for f in MARGINAL_FIELDS:
            lines.append(f"{DIGEST} cm:{label}.{f} {c.sha256_bits(getattr(out, f))}")
    print("\n" + "\n".join(lines), flush=True)
    assert [ln.split()[1] for ln in lines] == DIGEST_CASES


def test_criterion7_torch_compile_equals_eager() -> None:
    sem = c.live("semiring")
    compiled = torch.compile(sem.log_matmul)
    bad = {}
    for name, ops in {
        "live_block_s33": c.live_block_views(3, 33, 40, 0),
        "gauged91": c.regime_instance("gauged", 91, 0),
        "broadcast": c.broadcast_batch(4, 33, 91, 17, 0),
    }.items():
        a, b = cu(ops)
        for v, comp in VARIANTS.items():
            d = c.bit_diff(semiring_dispatched(a, b, comp), compiled(a, b, compensated=comp))
            if d:
                bad[f"{name}-{v}"] = d
    report("test_criterion7_torch_compile_equals_eager", bad)
    assert bad == {}, bad


# --------------------------------------------------------------------------- criterion 8


def test_criterion8_kernels_false_runs_the_torch_path() -> None:
    bad = {}
    for name, ops in groups()["adversarial"].items():
        a, b = cu(ops)
        for comp in (True, False):
            with c.entry_calls() as seen:
                got = c.live("semiring").log_matmul(a, b, compensated=comp, kernels=False)
                torch.cuda.synchronize()
            if seen:
                bad[f"{name} {comp}"] = f"entered logmatmul.py: {seen}"
            d = c.bit_diff(torch_ref(a, b, comp), got)
            if d:
                bad[f"{name} {comp}"] = d
    report("test_criterion8_kernels_false_runs_the_torch_path", bad)
    assert bad == {}, bad


_SUB = textwrap.dedent(
    """
    import json, logging, sys, time
    attempts = []
    if BLOCK:
        from tests.cd.unit._frozen_t0077b import TRITON_BLOCKER
        exec(TRITON_BLOCKER)
    import torch
    from tests.cd.unit import _frozen_t0077 as fz
    from tests.cd.unit import _frozen_t0077b as kb
    from tests.cd.unit import _frozen_t0077c as c
    records = []
    class H(logging.Handler):
        def emit(self, r):
            records.append({"name": r.name, "level": r.levelname, "msg": r.getMessage()})
    lg = logging.getLogger(c.LOGGER)
    lg.setLevel(logging.DEBUG)
    lg.addHandler(H(level=logging.DEBUG))
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    sem = c.live("semiring")
    big = c.live_block_views(3, 33, 40, 0)
    a, b = big[0].contiguous().to(dev), big[1].contiguous().to(dev)
    calls, diffs = [], []
    t0 = time.perf_counter()
    for i in range(3):
        with c.entry_calls() as seen:
            out = sem.log_matmul(a, b, compensated=bool(i % 2), kernels=True)
            torch.cuda.synchronize()
        if i == 0:
            first_call_s = time.perf_counter() - t0
        calls.append(len(seen))
        if BLOCK:
            diffs.append(c.bit_diff(c.frozen("semiring").log_matmul(a, b, compensated=bool(i % 2)), out))
    inst = fz.build_inst("s205_b256")
    with c.entry_calls() as seen:
        live = kb.prepared("live", inst, dev)()
        torch.cuda.synchronize()
    calls.append(len(seen))
    after_decision = list(records)
    for _ in range(2):
        sem.log_matmul(a.double(), b.double(), compensated=True, kernels=True)
    after_layout = list(records)
    seen_attempts = list(attempts)
    blocked = blocker_check() if BLOCK else None
    tv = None
    if not BLOCK:
        import triton
        tv = triton.__version__
    print("ASTRA_T0077C_SUB " + json.dumps({
        "after_decision": after_decision, "after_layout": after_layout, "calls": calls, "diffs": diffs,
        "attempts": seen_attempts, "blocked": blocked, "triton_version": tv, "first_call_s": first_call_s}))
    """
)


def _sub(block: bool) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-c", f"BLOCK = {block}\n" + _SUB],
        cwd=REPO,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("ASTRA_T0077C_SUB ")), None)
    assert proc.returncode == 0 and line is not None, (
        f"exit {proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    )
    out: dict[str, Any] = json.loads(line.split(" ", 1)[1])
    return out


def test_criterion8_triton_blocked_runs_the_torch_path() -> None:
    out = _sub(block=True)
    assert out["calls"] == [0, 0, 0, 0], f"dispatched without triton: {out['calls']}"
    assert out["diffs"] == [None, None, None], out["diffs"]
    assert out["blocked"] == {"find_spec_none": True, "import_raises": True}, out["blocked"]
    ours = [x for x in out["attempts"] if "/astra_vllm_plugin/cd/kernels/" in x[1]]
    assert ours, f"cd/kernels never tried triton (attempts {out['attempts']})"
    recs = out["after_decision"]
    assert len(recs) == 1 and recs[0]["name"] == c.LOGGER, recs
    msg = recs[0]["msg"]
    assert not msg.startswith("ASTRA_CD ") and "triton" in msg.lower() and "log_matmul" in msg, recs


def test_criterion8_one_decision_record_naming_triton_and_one_layout_record() -> None:
    out = _sub(block=False)
    assert all(n > 0 for n in out["calls"]), f"not dispatched: {out['calls']}"
    recs = out["after_decision"]
    assert len(recs) == 1, f"expected one decision record across both families (O7): {recs}"
    msg = recs[0]["msg"]
    assert recs[0]["name"] == c.LOGGER and not msg.startswith("ASTRA_CD "), recs
    assert "log_matmul" in msg and "triton" in msg.lower() and c.TRITON_VERSION in msg, msg
    assert len(out["after_layout"]) == 2, (
        f"a layout refusal (fp64) is logged once: {out['after_layout']}"
    )
    assert out["triton_version"] == c.TRITON_VERSION
    print(f"ASTRA_T0077C_FIRST_CALL_S {out['first_call_s']:.3f}")


def test_criterion8_regime_line() -> None:
    import triton

    regime = {
        "triton": triton.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(f"\n{REGIME} {json.dumps(regime, sort_keys=True)}", flush=True)
    assert triton.__version__ == c.TRITON_VERSION
