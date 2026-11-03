"""T-0077b on CUDA (tier 2, ``-m gpu``, ``/launch``): the fused H(q) tail is bit-identical to the torch path.

Run **inside the pinned container** through T-0067's runner (``tests/container_scripts/astra_cd_tier2_runner.py``:
5090 preflight, regime line), twice, by ``tests/contract/test_vllm_cd_kernels_gpu.py``: repo ``:ro`` at ``/repo``,
``--network none``, ``--noconftest -o addopts= -p no:cacheprovider``, ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``,
``ASTRA_CD_REQUIRE_CUDA=1``, and the four dev automata compiled on the host and mounted read-only at
``$ASTRA_CD_PAYLOADS`` (``cd_microbench.write_payloads``: ``b<bucket>.b64`` + ``manifest.json``). Without CUDA the
rows skip, and fail under ``ASTRA_CD_REQUIRE_CUDA=1``. API: ruling K1 (``tests/cd/unit/_frozen_t0077b.py``).

"Dispatched" is observed, not assumed: a call counts as dispatched only if a Python frame of
``cd/kernels/entropy.py`` ran (``entry_calls``), and ``kernels=False`` must run none.

**Ruling T-0077c O3:** after T-0077c the live ``log_matmul`` dispatches its own kernel on CUDA (N4, not bit-identity),
so every whole-call row here holds the live ``tree``/``marginals`` products on the frozen torch ``log_matmul`` (module
fixture ``held_log_matmul``; the criterion-7 subprocess does the same): these rows keep testing the H-tail kernel alone.

Rows (``KERNELS_CUDA_PASSED`` = 34):
- criterion 2 (11 + 4 + 9 = 24): on the core instances (``_frozen_t0077.INSTANCES``), the four dev automata (b64
  ``sudoku4_0``, b128 ``live_simple_133-86-0``, b256 ``130-84-0``, b512 ``165-98-0``; CL 256, V 262 144, logits
  ``randn·4``, seed 0, ``L_eff`` = CL) and the adversarial ``q_raw`` set (``tail_cases(wide=True)``: mixed sign,
  cancellation, 1e-30..1e3, an all-zero row, V not a power of two with an all-forbidden row, NaN/inf, the
  **subnormal** row (FTZ), the **contraction** row (FMA), and a V 262 144 row): ``q`` (all L × V), ``entropy``
  (all L rows), ``log_z``, ``log_a``, ``log_b`` of the dispatched call are bit-identical (NaN positions) to (a)
  ``kernels=False`` in the same process and (b) the frozen branch-base copy. A difference names the field, the count
  of differing entries and the max ulp; there is no tolerance and no fallback;
- criterion 3 (3): the row-sum entry vs ``compensated_sum`` at the interpreter rows' sizes and at V 262 144, blocks
  16 … 4096 and the production one, and on the adversarial row-sum and H-term rows;
- criterion 6 (4): 20 repeats in one process give the same bits; the dispatched call under
  ``torch.cuda.set_sync_debug_mode("error")`` (after a warm-up that absorbs the JIT compile) raises nothing; the
  digest row prints ``ASTRA_T0077B_DIGEST <case>.<field> <sha256>`` over the output bits of every criterion-2 case
  (24 × 5 = 120 lines; the host compares two processes); the tail entry adds < L·V/2 bytes of peak memory at
  [256, 262 144] (no [L, V] temporary, partials only);
- criterion 7 (3): a triton-blocked subprocess (meta-path finder) does not dispatch and equals the torch path, its
  one decision record giving the reason; three dispatched calls in a fresh process log exactly one record on
  ``astra_vllm_plugin.cd.kernels`` naming triton 3.7.1, not ``ASTRA_CD ``-prefixed; the image's triton is 3.7.1
  (printed as ``ASTRA_T0077B_REGIME <json>`` for the host's regime line).

Regime: image torch 2.13.0+cu130 / triton 3.7.1 / CUDA 13.0, RTX 5090, fp32, TF32 "high" set globally (vLLM's);
seeds per case in ``_frozen_t0077b`` and ``_frozen_t0077.INSTANCES``; 20 repeats; 2 processes (host).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.cd.unit import _frozen_t0077 as fz
from tests.cd.unit import _frozen_t0077b as kb
from tests.cd.unit import _interp_t0077b as it

pytestmark = pytest.mark.gpu

KERNELS_CUDA_PASSED = 34
REPO = Path(__file__).resolve().parents[3]
DEV = torch.device("cuda")
BUCKETS = (64, 128, 256, 512)
CL, V = 256, 262_144
REPEATS = 20
PAYLOADS_ENV = "ASTRA_CD_PAYLOADS"
DIGEST = "ASTRA_T0077B_DIGEST"
REGIME = "ASTRA_T0077B_REGIME"
TAIL_CASES = list(kb.tail_cases(wide=True))
#: every case the digest row covers, in print order: ``core:<inst>``, ``dev:b<bucket>``, ``tail:<case>``
DIGEST_CASES = (
    [f"core:{n}" for n in kb.INSTANCES]
    + [f"dev:b{b}" for b in BUCKETS]
    + [f"tail:{n}" for n in TAIL_CASES]
)


@pytest.fixture(autouse=True, scope="module")
def tf32_high() -> Iterator[None]:
    if not torch.cuda.is_available():
        why = (
            "CUDA rows run inside the pinned container (tests/contract/test_vllm_cd_kernels_gpu.py)"
        )
        if os.environ.get("ASTRA_CD_REQUIRE_CUDA") == "1":
            pytest.fail(f"ASTRA_CD_REQUIRE_CUDA=1 but CUDA is not available: {why}")
        pytest.skip(why)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    yield
    torch.set_float32_matmul_precision(prev)


@pytest.fixture(autouse=True, scope="module")
def held_log_matmul() -> Iterator[None]:
    """Ruling T-0077c O3: the criterion-2/6/7 whole-call rows test the H-tail kernel **alone**, so the live ``tree`` and
    ``marginals`` products are held on the frozen torch ``log_matmul`` (``core_frozen_t0077b``) for the whole module;
    ``log_matmul_held()`` checks the hold is in place before each live call."""
    tr, mg = kb.live("tree"), kb.live("marginals")
    saved = (tr.log_matmul, mg.log_matmul)
    tr.log_matmul = mg.log_matmul = kb.frozen("semiring").log_matmul
    try:
        yield
    finally:
        tr.log_matmul, mg.log_matmul = saved


def log_matmul_held() -> None:
    torch_lm = kb.frozen("semiring").log_matmul
    assert kb.live("tree").log_matmul is torch_lm and kb.live("marginals").log_matmul is torch_lm, (
        "log_matmul not held on the frozen torch function (ruling T-0077c O3)"
    )


# --------------------------------------------------------------------------- cases


def _payload_dir() -> Path:
    d = os.environ.get(PAYLOADS_ENV)
    assert d and (Path(d) / "manifest.json").is_file(), (
        f"${PAYLOADS_ENV} must name the host-compiled dev automata (cd_microbench.write_payloads); got {d!r}"
    )
    return Path(d)


def dev_prepared(which: str, bucket: int) -> Callable[..., Any]:
    """The A/B's inputs for one dev automaton (tables from the wire payload, CL 256, V 262 144, seed 0); the
    emission and tree are the live (byte-identical, not-owned) modules, shared by every path."""
    import numpy as np
    from astra_vllm_plugin.cd import automaton as cd_automaton
    from astra_vllm_plugin.cd import budget
    from astra_vllm_plugin.cd import requests as cd_requests

    d = _payload_dir()
    manifest = json.loads((d / "manifest.json").read_text())[str(bucket)]
    wa = cd_automaton.decode_payload((d / f"b{bucket}.b64").read_text())
    assert int(wa.n_states_bucket) == bucket, (wa.n_states_bucket, bucket)
    tables = cd_requests._tables_for(manifest["payload_sha256"], wa, DEV)
    sv = np.zeros(bucket, dtype=bool)
    sv[: len(wa.start_vector)] = np.asarray(wa.start_vector, dtype=bool)[:bucket]
    a = torch.from_numpy(sv).to(DEV)
    b = budget.terminal_factor(tables, 0)
    g = torch.Generator(device="cpu").manual_seed(0)
    x = (torch.randn(CL, V, generator=g) * 4.0).to(DEV)
    em, tr = kb.live("emission"), kb.live("tree")
    cm = em.class_mass(x, tables)
    tree = tr.up_sweep(em.log_transfer(x, tables, CL, class_mass=cm))
    mg = kb.frozen("marginals") if which == "frozen" else kb.live("marginals")

    def call(**kw: Any) -> Any:
        return mg.constrained_marginals(x, tables, tree, a, b, CL, class_mass=cm, **kw)

    return call


def case_prepared(which: str, case: str) -> Callable[..., Any]:
    log_matmul_held()
    kind, name = case.split(":", 1)
    if kind == "core":
        return kb.prepared(which, fz.build_inst(name), DEV)
    if kind == "dev":
        return dev_prepared(which, int(name.removeprefix("b")))
    return kb.prepared_tail(which, kb.tail_cases(wide=True)[name], DEV)


def dispatched(call: Callable[..., Any]) -> Any:
    with kb.entry_calls() as seen:
        out = call(kernels=True)
        torch.cuda.synchronize()
    assert seen, "kernels=True on CUDA did not reach cd/kernels/entropy.py (not dispatched)"
    return out


def torch_path(call: Callable[..., Any]) -> Any:
    with kb.entry_calls() as seen:
        out = call(kernels=False)
        torch.cuda.synchronize()
    assert seen == [], f"kernels=False reached cd/kernels/entropy.py: {seen}"
    return out


def check_criterion2(case: str) -> None:
    live = case_prepared("live", case)
    got = dispatched(live)
    ref_off = torch_path(live)
    ref_frozen = case_prepared("frozen", case)()
    bad = {
        "vs kernels=False": kb.field_diffs(ref_off, got),
        "vs frozen 2d90189": kb.field_diffs(ref_frozen, got),
        "kernels=False vs frozen": kb.field_diffs(ref_frozen, ref_off),
    }
    bad = {k: v for k, v in bad.items() if v}
    assert bad == {}, f"{case}: {json.dumps(bad, indent=1)}"


# --------------------------------------------------------------------------- criterion 2


@pytest.mark.parametrize("name", list(kb.INSTANCES))
def test_criterion2_core_instance(name: str) -> None:
    check_criterion2(f"core:{name}")


@pytest.mark.parametrize("bucket", BUCKETS)
def test_criterion2_dev_automaton(bucket: int) -> None:
    check_criterion2(f"dev:b{bucket}")


@pytest.mark.parametrize("name", TAIL_CASES)
def test_criterion2_adversarial_q_raw(name: str) -> None:
    if (
        name == "contraction"
    ):  # the row only tests fp-fusion if fusing its first product changes bits
        assert kb.fma_sensitive(kb.contraction_row()) > 0
    if name == "subnormal":
        q = kb.subnormal_row()
        assert bool((q <= 1e-38).all()) and int((q < torch.finfo(torch.float32).tiny).sum()) > 0
    check_criterion2(f"tail:{name}")


# --------------------------------------------------------------------------- criterion 3


def _row_sum(x: torch.Tensor, block: int | None) -> str | None:
    rs = kb.entry("kernels.entropy", "compensated_row_sum")
    ref = kb.frozen("semiring").compensated_sum(x)
    got = rs(x.clone(), block=block)
    torch.cuda.synchronize()
    return kb.bit_diff(ref, got)


def test_criterion3_row_sum_sizes() -> None:
    bad = {}
    for V_ in (*it.SIZES, 262_143, 262_144, 262_145):
        for L in it._l_for(V_) if V_ <= 4097 else (2,):
            x = it.sized_input(V_, L).to(DEV)
            for b in it.ROW_SUM_BLOCKS:
                d = _row_sum(x, b)
                if d is not None:
                    bad[f"V{V_}-L{L}-b{b}"] = d
    assert bad == {}, bad


def test_criterion3_row_sum_bits_do_not_depend_on_the_block() -> None:
    bad = {}
    for L, V_ in (*it.INVARIANCE_SHAPES, (4, 262_144)):
        x = it.sized_input(V_, L).to(DEV)
        for b in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, None):
            d = _row_sum(x, b)
            if d is not None:
                bad[f"{L}x{V_}-b{b}"] = d
    assert bad == {}, bad


def test_criterion3_row_sum_adversarial() -> None:
    sem = kb.frozen("semiring")
    rows = dict(kb.row_sum_cases())
    for n, q in kb.tail_cases(wide=True).items():
        rows[f"q_raw:{n}"] = q
        tot = sem.compensated_sum(q).unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
        rows[f"term:{n}"] = (
            torch.maximum(torch.log(q.clamp_min(1e-38)), torch.log(tot) + math.log(1e-30)) * q
        )
    bad = {}
    for n, x in rows.items():
        for b in it.ROW_SUM_BLOCKS:
            d = _row_sum(x.to(DEV), b)
            if d is not None:
                bad[f"{n}-b{b}"] = d
    assert bad == {}, bad


# --------------------------------------------------------------------------- criterion 6


def test_criterion6_twenty_repeats_same_bits() -> None:
    bad = {}
    for case in ("core:s205_b256", "tail:wide_v262144", "tail:nan_row"):
        call = case_prepared("live", case)
        first = dispatched(call)
        for i in range(1, REPEATS):
            d = kb.field_diffs(first, dispatched(call), ("q", "entropy"))
            if d:
                bad[f"{case} call {i}"] = d
    assert bad == {}, bad


def test_criterion6_no_host_sync_under_sync_debug_error() -> None:
    """The dispatched call (inputs prepared and the kernels compiled beforehand) under sync-debug "error"."""
    calls = {c: case_prepared("live", c) for c in ("core:s205_b256", "tail:wide_v262144")}
    for c in calls.values():
        dispatched(c)  # warm-up: JIT compile happens here, outside the strict window
    torch.cuda.synchronize()
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with kb.entry_calls() as seen:
            for c in calls.values():
                c(kernels=True)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()
    assert seen, "not dispatched inside the strict window"


def test_criterion6_digests_of_the_output_bits() -> None:
    """Prints one ``ASTRA_T0077B_DIGEST`` line per case and field; the host compares two processes."""
    lines = []
    for case in DIGEST_CASES:
        out = dispatched(case_prepared("live", case))
        for f in kb.MARGINAL_FIELDS:
            lines.append(f"{DIGEST} {case}.{f} {kb.sha256_bits(getattr(out, f))}")
    print("\n" + "\n".join(lines), flush=True)
    assert len(lines) == len(DIGEST_CASES) * len(kb.MARGINAL_FIELDS) == 120


def test_criterion6_tail_entry_holds_no_lv_temporary() -> None:
    """Implementation notes, "Memory": partials ``[L, p/BLOCK]`` only; ``q`` written in place. Peak added by the
    tail entry at [256, 262 144] fp32 is under L·V/2 bytes (32 MiB; one [L, V] fp32 temporary is 256 MiB)."""
    hq = kb.entry("kernels.entropy", "hq_tail")
    q = kb._log_uniform(CL, V, 1e-12, 30.0, 41).to(DEV)
    hq(q.clone())  # compile
    buf = q.clone()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    q_out, h = hq(buf)
    torch.cuda.synchronize()
    added = torch.cuda.max_memory_allocated() - before
    print(f"ASTRA_T0077B_TAIL_PEAK_ADDED {added} bytes at [{CL}, {V}]")
    assert q_out.data_ptr() == buf.data_ptr(), "q not written into q_raw's storage"
    assert tuple(h.shape) == (CL,)
    assert added < CL * V // 2, f"tail entry peak added {added / 2**20:.1f} MiB"


# --------------------------------------------------------------------------- criterion 7

_SUB = textwrap.dedent(
    """
    import json, logging, sys
    attempts = []
    if BLOCK:
        from tests.cd.unit._frozen_t0077b import TRITON_BLOCKER
        exec(TRITON_BLOCKER)
    import torch
    from tests.cd.unit import _frozen_t0077 as fz
    from tests.cd.unit import _frozen_t0077b as kb
    records = []
    class H(logging.Handler):
        def emit(self, r):
            records.append({"name": r.name, "msg": r.getMessage()})
    lg = logging.getLogger(kb.LOGGER)
    lg.setLevel(logging.DEBUG)
    lg.addHandler(H(level=logging.DEBUG))
    torch.set_float32_matmul_precision("high")
    # ruling T-0077c O3: the H-tail kernel alone (log_matmul held on the frozen torch function)
    _lm = kb.frozen("semiring").log_matmul
    kb.live("tree").log_matmul = kb.live("marginals").log_matmul = _lm
    dev = torch.device("cuda")
    inst = fz.build_inst("s205_b256")
    live = kb.prepared("live", inst, dev)
    entry_calls, outs = [], []
    for _ in range(3):
        with kb.entry_calls() as seen:
            outs.append(live(kernels=True))
            torch.cuda.synchronize()
        entry_calls.append(len(seen))
    ref = kb.prepared("frozen", inst, dev)()
    diffs = [kb.field_diffs(ref, o) for o in outs]
    tv = None
    if not BLOCK:
        import triton
        tv = triton.__version__
    seen_attempts = list(attempts)  # before blocker_check records its own
    blocked = blocker_check() if BLOCK else None
    print("ASTRA_T0077B_SUB " + json.dumps({
        "records": records, "entry_calls": entry_calls, "diffs": diffs, "attempts": seen_attempts,
        "blocked": blocked,
        "triton_version": tv}))
    """
)


def _sub(block: bool) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-c", f"BLOCK = {block}\n" + _SUB],
        cwd=REPO,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=1200,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("ASTRA_T0077B_SUB ")), None)
    assert proc.returncode == 0 and line is not None, (
        f"exit {proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    )
    out: dict[str, Any] = json.loads(line.split(" ", 1)[1])
    return out


def test_criterion7_triton_blocked_runs_the_torch_path() -> None:
    out = _sub(block=True)
    assert out["entry_calls"] == [0, 0, 0], f"dispatched without triton: {out['entry_calls']}"
    assert out["diffs"] == [{}, {}, {}], out["diffs"]
    assert out["attempts"], "triton availability was never checked on CUDA"
    assert out["blocked"] == {"find_spec_none": True, "import_raises": True}, out["blocked"]
    ours = [a for a in out["attempts"] if "/astra_vllm_plugin/cd/kernels/" in a[1]]
    assert ours, (
        f"cd/kernels never tried triton (attempts {out['attempts']}): the torch path was not its decision"
    )
    recs = out["records"]
    assert len(recs) == 1 and recs[0]["name"] == kb.LOGGER, recs
    assert not recs[0]["msg"].startswith("ASTRA_CD "), recs
    assert "triton" in recs[0]["msg"].lower(), f"the record does not give the reason: {recs}"


def test_criterion7_decision_logged_once_naming_triton() -> None:
    out = _sub(block=False)
    assert all(n > 0 for n in out["entry_calls"]), f"not dispatched: {out['entry_calls']}"
    assert out["diffs"] == [{}, {}, {}], out["diffs"]
    recs = out["records"]
    assert len(recs) == 1, f"expected exactly one decision record in 3 calls: {recs}"
    msg = recs[0]["msg"]
    assert recs[0]["name"] == kb.LOGGER and not msg.startswith("ASTRA_CD "), recs
    assert "triton" in msg.lower() and kb.TRITON_VERSION in msg, msg
    assert out["triton_version"] == kb.TRITON_VERSION


def test_criterion7_image_triton_is_3_7_1() -> None:
    import triton

    regime = {
        "triton": triton.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }
    print(f"\n{REGIME} {json.dumps(regime, sort_keys=True)}", flush=True)
    assert triton.__version__ == kb.TRITON_VERSION
