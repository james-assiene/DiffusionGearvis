"""T-0077b criteria 3, 4, 5 and 7 on CPU: the dispatch, the CPU path, the kernels under the interpreter, the controls.

API under test: ruling K1 in ``_frozen_t0077b`` (``entropy.compensated_row_sum``/``entropy.hq_tail``,
``backend.dispatch_decision``, ``constrained_marginals(..., kernels=)``). Reference: the frozen branch-base copy
``tests/cd/oracle/core_frozen_t0077b/`` (2d90189, byte-identical; MANIFEST.md).

Rows:
- **Frozen copy** (live): MANIFEST sha256 and the blobs at 2d90189; side-by-side import.
- **Negatives** (live; criterion 1/5): the not-owned core and per-step files byte-identical to the branch base.
- **Dispatch** (criterion 7): the decision's truth table (dispatch iff CUDA ∧ triton importable ∧ ``kernels`` not
  False); the decision logged once per process on ``astra_vllm_plugin.cd.kernels`` (subprocess, 4 calls → 1 record,
  never ``ASTRA_CD ``-prefixed, naming the CPU reason); CPU tensors never reach ``cd/kernels/entropy.py``.
- **CPU identity** (criterion 5): every field of CPU ``constrained_marginals`` bit-identical to the frozen copy's on
  the core instances (``_frozen_t0077.INSTANCES``) with ``kernels=True``, ``kernels=False`` and (live) no keyword.
- **Interpreter** (criteria 3, 4; ruling M2), one ``TRITON_INTERPRET=1`` subprocess (``_interp_t0077b``): the
  row-sum entry vs ``compensated_sum`` at sizes 1, 2, 3, 5, 2^k ± 1, L ∈ {1, 3, 8}, several blocks; block
  invariance; the adversarial row-sum and H-term rows; the tail entry's ``q`` bit for bit and in place, its entropy
  ≤ 1e-5 (N4) with NaN positions equal and the max ulp printed.
- **Power controls** (live; the adversarial set must be able to tell orders apart): on those rows (i) the
  adjacent-pair tree without the error term, (ii) ``torch.sum`` and (iii) the halves pairing each differ bitwise
  from ``compensated_sum`` on at least one row; the contraction row changes when its first product is fused; the
  subnormal row is subnormal; the ``tail_via`` route really feeds the chosen ``q_raw`` (hand-computed H).

Regime: CPU, torch 2.13.0+cpu and triton 3.7.1 per ``uv.lock``, fp32, ``OMP_NUM_THREADS=4``; seeds per case in
``_frozen_t0077b``/``_interp_t0077b``; instance seeds in ``_frozen_t0077.INSTANCES``.
"""

from __future__ import annotations

import functools
import itertools
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import torch

from tests import _git_range as gr
from tests.cd.unit import _frozen_t0077 as fz
from tests.cd.unit import _frozen_t0077b as kb
from tests.cd.unit import _interp_t0077b as it

REPO = Path(__file__).resolve().parents[3]
NAMES = list(kb.INSTANCES)


@functools.cache
def inst(name: str) -> fz.Inst:
    return fz.build_inst(name)


@functools.cache
def frozen_marg(name: str) -> Any:
    return kb.marginals_on("frozen", inst(name))


def _env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(kb.PLUGIN_SRC)])
    env.setdefault("OMP_NUM_THREADS", "4")
    env.update(extra)
    return env


# --------------------------------------------------------------------------- the frozen copy itself


def test_frozen_copies_are_byte_identical_to_2d90189() -> None:
    """MANIFEST's sha256 per file, and (where the object exists) the blob at 2d90189 itself."""
    bad = []
    for name in kb.FROZEN_MODULES:
        got = kb.file_sha256(kb.FROZEN_DIR / f"{name}.py")
        if got != kb.FROZEN_SHA256[name]:
            bad.append(f"{name}.py: sha256 {got} != MANIFEST {kb.FROZEN_SHA256[name]}")
        blob = subprocess.run(
            ["git", "show", f"{kb.FROZEN_COMMIT}:{_plugin_rel(name)}"],
            cwd=REPO,
            capture_output=True,
        )
        if blob.returncode == 0 and kb.hashlib.sha256(blob.stdout).hexdigest() != got:
            bad.append(f"{name}.py: not the blob at {kb.FROZEN_COMMIT}")
    manifest = (kb.FROZEN_DIR / "MANIFEST.md").read_text()
    bad += [f"MANIFEST lacks {h}" for h in kb.FROZEN_SHA256.values() if h not in manifest]
    assert not bad, "\n".join(bad)


def _plugin_rel(name: str) -> str:
    return f"plugin/astra_vllm_plugin/astra_vllm_plugin/cd/{name}.py"


def test_frozen_modules_load_side_by_side_with_the_live_ones() -> None:
    for name in kb.FROZEN_MODULES:
        f, v = kb.frozen(name), kb.live(name)
        assert f is not v and f.__name__ == f"{kb.FROZEN_PKG}.{name}"
        assert Path(f.__file__).resolve().parent == kb.FROZEN_DIR.resolve()
    assert kb.frozen("marginals").compensated_sum is kb.frozen("semiring").compensated_sum
    assert kb.frozen("marginals").compensated_sum is not kb.live("semiring").compensated_sum


# --------------------------------------------------------------------------- negatives: files not owned


#: not owned by T-0077b (task "Owned files"; umbrella criterion 2): byte-identical to the branch base.
#: ``semiring.py`` left this list at the T-0077b merge (ruling T-0077c O3: it is T-0077c's owned file; its
#: byte-identity negative closed with T-0077b, and T-0077c's own rows pin the rest of it against b53457a).
_NOT_OWNED_AT_T0077B = (
    *(_plugin_rel(n) for n in ("tree", "emission", "mapdecode", "draw", "steps")),
    *(
        _plugin_rel(n)
        for n in ("requests", "xargs", "sampler", "budget", "record", "record_schema")
    ),
    "local_astra/bench/thresholds.py",
    "scripts/gpu/cd_microbench.py",
    "scripts/gpu/cd_b13.py",
)


#: Every pinned path, the three T-0081 files included (T-0094b criterion 7): over T-0077b's closed range no later-task
#: exemption is needed. ``LATER_TASK_OWNED`` and its exemption row are retired with the working-tree comparison; the
#: "a one-byte change is still reported" half is the helper's own row,
#: ``tests/unit/test_guards_use_fixed_refs.py::test_changed_reports_every_modified_deleted_and_added_file_under_its_paths_only``.
NOT_OWNED = _NOT_OWNED_AT_T0077B
#: T-0077b's merge commit b53457a; BASE = ``git merge-base b53457a^1 b53457a^2`` = ``kb.FROZEN_COMMIT`` (2d90189),
#: TIP = ``b53457a^2`` (T-0094b criterion 2).
T0077B_MERGE = "b53457a"
T0077B_BASE = "2d90189"
T0077B_TIP = "ac2befb"


@pytest.mark.parametrize("rel", NOT_OWNED)
def test_not_owned_file_is_byte_identical_to_the_branch_base(rel: str) -> None:
    """Umbrella criterion 2: ``rel`` does not differ between T-0077b's BASE (the branch base) and TIP (T-0094b: a
    closed range, never the working tree)."""
    assert T0077B_BASE == kb.FROZEN_COMMIT
    gr.prove_range(T0077B_BASE, T0077B_TIP, T0077B_MERGE)
    assert gr.changed(T0077B_BASE, T0077B_TIP, rel) == [], (
        f"{rel} changed in {T0077B_BASE}..{T0077B_TIP}"
    )


def test_the_not_owned_guard_has_power_and_keeps_the_t0081_files() -> None:
    """T-0094b criteria 5 and 7 (adversarial): over ``BASE..6a8600b`` the predicate over ``NOT_OWNED`` reports exactly
    T-0081's three files (``requests``, ``xargs``, ``sampler``), reached through the tuple; a predicate reporting nothing,
    or a tuple that dropped one of them, fails here."""
    gr.require_commits(T0077B_BASE, gr.LATER)
    t0081 = sorted(
        p for p in NOT_OWNED if p.rsplit("/", 1)[-1] in ("requests.py", "xargs.py", "sampler.py")
    )
    assert len(t0081) == 3, t0081
    assert gr.changed(T0077B_BASE, gr.LATER, *NOT_OWNED) == t0081


# --------------------------------------------------------------------------- dispatch (criterion 7)


@pytest.mark.parametrize(
    ("on_cuda", "triton_available", "kernels"), list(itertools.product((True, False), repeat=3))
)
def test_dispatch_truth_table(on_cuda: bool, triton_available: bool, kernels: bool) -> None:
    """Dispatch exactly when the tensor is on CUDA, triton imports and ``kernels`` is not False."""
    decide = kb.entry("kernels.backend", "dispatch_decision")
    got = decide(on_cuda=on_cuda, triton_available=triton_available, kernels=kernels)
    assert got is (on_cuda and triton_available and kernels), (
        f"dispatch_decision(on_cuda={on_cuda}, triton_available={triton_available}, "
        f"kernels={kernels}) = {got!r}"
    )


_LOG_ONCE = textwrap.dedent(
    """
    import json, logging, sys
    from tests.cd.unit import _frozen_t0077 as fz
    from tests.cd.unit import _frozen_t0077b as kb

    records = []

    class H(logging.Handler):
        def emit(self, r):
            records.append({"name": r.name, "level": r.levelname, "msg": r.getMessage()})

    lg = logging.getLogger(kb.LOGGER)
    lg.setLevel(logging.DEBUG)
    lg.addHandler(H(level=logging.DEBUG))
    inst = fz.build_inst("s31_b64_dfa")
    for kw in ({}, {"kernels": True}, {"kernels": False}, {"kernels": True}):
        kb.marginals_on("live", inst, "cpu", **kw)
    print("ASTRA_T0077B_LOG " + json.dumps({
        "records": records,
        "triton_loaded": sorted(m for m in sys.modules if m.split(".")[0] == "triton"),
    }))
    """
)


def test_decision_logged_once_per_process_on_cpu() -> None:
    """4 CPU calls in a fresh process → exactly one record on ``astra_vllm_plugin.cd.kernels``, naming the CPU
    reason for the torch path, never starting with ``ASTRA_CD `` (the golden/record capture reads those)."""
    proc = subprocess.run(
        [sys.executable, "-c", _LOG_ONCE],
        cwd=REPO,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("ASTRA_T0077B_LOG "))
    out = json.loads(line.split(" ", 1)[1])
    recs = out["records"]
    assert len(recs) == 1, f"expected exactly one decision record in 4 calls, got {recs}"
    rec = recs[0]
    assert rec["name"] == kb.LOGGER, rec
    assert not rec["msg"].startswith("ASTRA_CD "), rec
    assert "cpu" in rec["msg"].lower(), f"the record does not name the CPU reason: {rec}"
    assert out["triton_loaded"] == [], f"CPU calls imported triton: {out['triton_loaded']}"


@pytest.mark.parametrize("kernels", [True, False])
def test_cpu_tensors_never_reach_the_kernel_entries(kernels: bool) -> None:
    """Criterion 5: on CPU the kernel is never dispatched, whatever ``kernels`` says (``kernels=True`` included)."""
    kb.live("kernels.entropy")  # the entries exist (else "never called" proves nothing)
    with kb.entry_calls() as seen:
        kb.marginals_on("live", inst("s91_b128_nfa"), "cpu", kernels=kernels)
    assert seen == [], f"CPU constrained_marginals(kernels={kernels}) called {seen} in entropy.py"


def test_entry_call_spy_sees_calls_into_entropy_py(tmp_path: Path) -> None:
    """Control for ``entry_calls``: a function compiled with ``entropy.py``'s path as its filename is seen, one
    compiled elsewhere is not (the spy matches the file, not a name)."""
    target = str((kb.KERNELS_DIR / "entropy.py").resolve())
    ns: dict[str, Any] = {}
    exec(compile("def probe():\n    return 1\n", target, "exec"), ns)
    other: dict[str, Any] = {}
    exec(compile("def probe():\n    return 2\n", str(tmp_path / "entropy.py"), "exec"), other)
    with kb.entry_calls() as seen:
        other["probe"]()
        ns["probe"]()
    assert seen == ["probe"]


# --------------------------------------------------------------------------- CPU identity (criterion 5)


@pytest.mark.parametrize("name", NAMES)
def test_cpu_marginals_without_keyword_equal_the_branch_base(name: str) -> None:
    """Criterion 5 (negative; true at the base and must stay true): every field bit-identical to the frozen copy."""
    got = kb.marginals_on("live", inst(name))
    assert kb.field_diffs(frozen_marg(name), got) == {}


@pytest.mark.parametrize("kernels", [True, False])
@pytest.mark.parametrize("name", NAMES)
def test_cpu_marginals_with_kernels_keyword_equal_the_branch_base(name: str, kernels: bool) -> None:
    """Criterion 5 / M6: the new keyword-only ``kernels`` changes nothing on CPU, either value."""
    got = kb.marginals_on("live", inst(name), "cpu", kernels=kernels)
    assert kb.field_diffs(frozen_marg(name), got) == {}


def test_kernels_keyword_is_keyword_only_with_default_true() -> None:
    import inspect

    p = inspect.signature(kb.live("marginals").constrained_marginals).parameters.get("kernels")
    assert p is not None, "constrained_marginals has no `kernels` parameter"
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is True, p


# --------------------------------------------------------------------------- interpreter (criteria 3, 4)


@pytest.fixture(scope="module")
def interp() -> dict[str, Any]:
    """One ``TRITON_INTERPRET=1`` subprocess for every interpreter row (its kernel cache never meets a compiled one)."""
    proc = subprocess.run(
        [sys.executable, "-m", "tests.cd.unit._interp_t0077b"],
        cwd=REPO,
        env=_env(TRITON_INTERPRET="1"),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(it.MARKER + " ")), None)
    assert line is not None, f"no {it.MARKER} line (exit {proc.returncode}):\n{proc.stderr[-6000:]}"
    out: dict[str, Any] = json.loads(line[len(it.MARKER) + 1 :])
    assert out["triton_interpret"] == "1"
    return out


def _no_error(interp: dict[str, Any]) -> None:
    assert interp["error"] is None, f"the interpreter run failed:\n{interp['error']}"


@pytest.mark.parametrize("key", it.row_sum_keys())
def test_interpreter_row_sum_is_compensated_sum(interp: dict[str, Any], key: str) -> None:
    """Criterion 3/4: ``compensated_row_sum`` bit-identical to ``compensated_sum`` (V{n}-L{L}-b{block})."""
    _no_error(interp)
    row = interp["row_sum"].get(key)
    assert row is not None, f"{key} not computed"
    assert row["ok"], f"{key}: {row['detail']}"


@pytest.mark.parametrize("key", it.block_keys())
def test_interpreter_row_sum_bits_do_not_depend_on_the_block(
    interp: dict[str, Any], key: str
) -> None:
    """Criterion 3: blocks 16 … 1024 and the production block all give ``compensated_sum``'s bits (aligned
    power-of-two subtrees compose)."""
    _no_error(interp)
    rows = interp["block"].get(key)
    assert rows is not None and len(rows) == len(it.INVARIANCE_BLOCKS), rows
    bad = {b: r["detail"] for b, r in rows.items() if not r["ok"]}
    assert bad == {}, f"{key}: {bad}"


@pytest.mark.parametrize("key", it.row_sum_adv_keys())
def test_interpreter_row_sum_adversarial(interp: dict[str, Any], key: str) -> None:
    """Adversarial (a different reduction order or a dropped error term): signed, cancelling, wide-range rows and
    the H-term rows of the adversarial ``q_raw`` set; the power controls below show these rows tell orders apart."""
    _no_error(interp)
    row = interp["row_sum_adv"].get(key)
    assert row is not None, f"{key} not computed"
    assert row["ok"], f"{key}: {row['detail']}"


@pytest.mark.parametrize("key", it.tail_keys())
def test_interpreter_tail_q_bit_identical_in_place(interp: dict[str, Any], key: str) -> None:
    """Criterion 4: the tail entry's ``q`` is the torch path's bit for bit (only IEEE add and divide on that route),
    written into ``q_raw``'s storage (no [L, V] temporary for q)."""
    _no_error(interp)
    row = interp["tail"].get(key)
    assert row is not None, f"{key} not computed"
    assert "error" not in row, f"{key}: {row.get('error')}"
    assert row["q"] is None, f"{key}: q {row['q']}"
    assert row["in_place"], f"{key}: q is not written into q_raw's storage"


@pytest.mark.parametrize("key", it.tail_keys())
def test_interpreter_tail_entropy_within_n4(interp: dict[str, Any], key: str) -> None:
    """Criterion 4 / ruling M2: entropy |dH| ≤ 1e-5 under the interpreter (numpy's log, not libdevice), NaN
    positions identical; the max ulp is printed (bit-identity of entropy is the CUDA module's)."""
    _no_error(interp)
    row = interp["tail"].get(key)
    assert row is not None and "error" not in row, f"{key}: {row}"
    case = key.rsplit("-b", 1)[0]
    L = kb.tail_cases()[case].shape[0]
    print(f"ASTRA_T0077B_INTERP_H {key} max|dH| {row['max_abs_dh']:.3g} max ulp {row['max_ulp_h']}")
    assert row["shape_h"] == [L], row
    assert row["nan_positions_equal"], f"{key}: NaN positions of entropy differ"
    assert row["finite_dh"] and row["max_abs_dh"] <= 1e-5, (
        f"{key}: max|dH| {row['max_abs_dh']} > 1e-5 ({row['max_ulp_h']} ulp)"
    )


# --------------------------------------------------------------------------- power controls (live)


def _adversarial_sum_rows() -> dict[str, torch.Tensor]:
    """Every adversarial input the row sum sees: the row-sum cases, the ``q_raw`` cases (first sum) and their
    H-term rows (second sum)."""
    sem = kb.frozen("semiring")
    rows = dict(kb.row_sum_cases())
    for n, q in kb.tail_cases().items():
        rows[f"q_raw:{n}"] = q
        tot = sem.compensated_sum(q).unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
        lt = torch.log(tot)
        rows[f"term:{n}"] = torch.maximum(torch.log(q.clamp_min(1e-38)), lt + math.log(1e-30)) * q
    return rows


CONTROLS = {
    "adjacent_pairs_without_error": kb.adjacent_pairs_no_error,
    "torch_sum": lambda x: x.sum(dim=-1),
    "halves_pairing": kb.halves_pairing,
}


@pytest.mark.parametrize("control", list(CONTROLS))
def test_power_control_other_orders_differ_on_the_adversarial_rows(control: str) -> None:
    """Without this, bit-identity proves nothing: each wrong order differs bitwise from ``compensated_sum`` on at
    least one finite adversarial row."""
    sem = kb.frozen("semiring")
    fn = CONTROLS[control]
    differing = []
    for name, x in _adversarial_sum_rows().items():
        for r in range(x.shape[0]):
            row = x[r : r + 1]
            if not bool(torch.isfinite(row).all()):
                continue
            if kb.bit_diff(sem.compensated_sum(row), fn(row)) is not None:
                differing.append(f"{name}[{r}]")
    print(f"ASTRA_T0077B_CONTROL {control}: differs on {len(differing)} rows: {differing[:8]}")
    assert differing, f"{control} matches compensated_sum on every adversarial row: no power"


@pytest.mark.parametrize("V", list(kb.HALVES_PINS))
def test_power_control_compensated_halves_differs_on_the_pinned_rows(V: int) -> None:
    """Reviewer A, finding 1: halves pairing that *keeps* the full TwoSum error tree differs bitwise from
    ``compensated_sum`` on every pinned ``halves_tie`` row (they were searched for this), so the row-sum rows that
    carry them (interpreter and CUDA, production block and block 16) can kill that order. Also controls that the
    reference is a sum (within 2^-40 of max|x| of the fp64 sum) and agrees with the spec on a smooth row."""
    sem = kb.frozen("semiring")
    x = kb.row_sum_cases()[f"halves_tie_v{V}"]
    ref, got = sem.compensated_sum(x), kb.compensated_halves(x)
    differ = [r for r in range(x.shape[0]) if kb.bit_diff(ref[r], got[r]) is not None]
    print(
        f"ASTRA_T0077B_CONTROL compensated_halves V {V}: differs on rows {differ} of {x.shape[0]}"
    )
    assert differ == list(range(x.shape[0])), differ
    exact = x.double().sum(-1)
    scale = x.double().abs().max(-1).values
    assert bool(((got.double() - exact).abs() <= scale * 2.0**-40).all()), (
        "the reference is not a sum"
    )
    # the reference pairs differently but is the same algorithm: on a smooth row it agrees with the spec
    smooth = kb._log_uniform(8, V, 1e-3, 1e3, 99)
    assert kb.bit_diff(sem.compensated_sum(smooth), kb.compensated_halves(smooth)) is None


def test_power_control_the_contraction_row_is_fma_sensitive() -> None:
    """The contraction row (CUDA module) is only a test of ``enable_fp_fusion=False`` if fusing its first product
    into the first TwoSum add changes bits somewhere."""
    n = kb.fma_sensitive(kb.contraction_row())
    print(f"ASTRA_T0077B_CONTROL contraction: {n} first-level pairs change under FMA")
    assert n > 0


def test_power_control_the_subnormal_row_is_subnormal() -> None:
    """Every q̃ in (0, 1e-38] with some below fp32 tiny (an FTZ kernel flushes them), total ≪ 1e-8; the torch path
    keeps them (q > 0 everywhere)."""
    q_raw = kb.subnormal_row()
    tiny = torch.finfo(torch.float32).tiny
    assert bool((q_raw > 0).all()) and bool((q_raw <= 1e-38).all())
    assert int((q_raw < tiny).sum()) > q_raw.numel() // 2
    ref = kb.tail_via("frozen", q_raw)
    assert bool((kb.frozen("semiring").compensated_sum(q_raw) < 1e-8).all())
    assert bool((ref.q > 0).all()), "the torch path flushed a subnormal q"


def test_tail_via_feeds_the_chosen_q_raw() -> None:
    """Control for ``tail_via``: q̃ = [1, 1, 2] → q = [1/4, 1/4, 1/2] and H = 1.5·ln 2 (by hand)."""
    q_raw = torch.tensor([[1.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    out = kb.tail_via("frozen", q_raw)
    assert torch.equal(out.q[0], torch.tensor([0.25, 0.25, 0.5]))
    assert abs(float(out.entropy[0]) - 1.5 * math.log(2.0)) < 1e-6
    assert torch.equal(out.q[1], torch.zeros(3))
    # the all-zero row: total clamps to fp32 tiny, H = log(tiny) exactly (no term contributes)
    assert float(out.entropy[1]) == float(torch.log(torch.tensor(torch.finfo(torch.float32).tiny)))


def test_cuda_module_row_count_matches_the_wrappers_k() -> None:
    """The CUDA module has exactly K rows, and the host wrapper pins the same K and the same case lists (a row
    added to one and not the other would make the tier-2 count check vacuous or impossible)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:randomly",
            "tests/cd/cuda/test_cd_kernels_cuda.py",
        ],
        cwd=REPO,
        env=_env(ASTRA_CD_TESTS="1"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    ids = [ln for ln in proc.stdout.splitlines() if "test_cd_kernels_cuda.py::" in ln]
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    sys.path.insert(0, str(REPO))
    from tests.cd.cuda import test_cd_kernels_cuda as cuda_mod
    from tests.contract import test_vllm_cd_kernels_gpu as wrapper

    assert len(ids) == cuda_mod.KERNELS_CUDA_PASSED == wrapper.KERNELS_CUDA_PASSED == 34, len(ids)
    assert tuple(kb.INSTANCES) == wrapper.CORE
    assert tuple(kb.tail_cases(wide=True)) == wrapper.TAIL
    assert set(cuda_mod.DIGEST_CASES) == {n.rsplit(".", 1)[0] for n in wrapper.EXPECTED_DIGESTS}


def test_sha256_bits_hashes_zero_dim_and_one_element_tensors() -> None:
    """Ruling K9: the digest row hashes ``log_z`` (0-dim). Different values, shapes or signs of zero give different
    digests; the same bits give the same digest (the two-process comparison relies on both)."""
    t = torch.tensor
    d = {
        "0d_1.5": kb.sha256_bits(t(1.5)),
        "0d_2.5": kb.sha256_bits(t(2.5)),
        "1el_1.5": kb.sha256_bits(t([1.5])),
        "1el_2.5": kb.sha256_bits(t([2.5])),
        "0d_+0": kb.sha256_bits(t(0.0)),
        "0d_-0": kb.sha256_bits(t(-0.0)),
        "1x1_1.5": kb.sha256_bits(t([[1.5]])),
    }
    assert len(set(d.values())) == len(d), d
    assert kb.sha256_bits(t(1.5)) == d["0d_1.5"]
    assert kb.sha256_bits(t([1.5, 2.5])[0]) == d["0d_1.5"]  # a 0-dim view of a larger tensor
