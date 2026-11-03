"""T-0077c criteria 2 (launch count), 3b/3c, 6, 7 (AST part) and 8 on CPU: the CPU path, the dispatch, the kernel under
the interpreter, its PTX.

API under test: ``_frozen_t0077c``'s ruling (``kernels.logmatmul.log_matmul(a3, b3, *, compensated, block_n,
block_k)``, ``kernels.backend.logmatmul_layout_ok``/``use_logmatmul``, ``semiring.log_matmul(..., kernels=True)``).
Reference: the frozen branch-base copy ``tests/cd/oracle/core_frozen_t0077c/`` (b53457a, byte-identical; MANIFEST.md)
and the spec emulation ``_frozen_t0077c.spec_log_matmul``.

Rows:
- **Frozen copy** (live): MANIFEST sha256 and the blobs at b53457a; side-by-side import, the frozen ``marginals``
  wired to the frozen ``kernels``.
- **Negatives** (live; criteria 1, 6, 12): not-owned files byte-identical to b53457a (``local_astra/**``, the locks and
  ``pyproject.toml`` files by ``git diff``); in ``semiring.py`` ``compensated_sum``, ``_finish``, ``two_sum`` and the
  segment functions byte-identical, every function other than ``log_matmul`` AST-equal, the constants unchanged.
- **CPU path** (criterion 6): ``semiring.log_matmul`` ``torch.equal`` to the frozen copy's (bits, NaN positions)
  without the keyword (live) and with ``kernels=True``/``False``; ``kernels`` keyword-only, default True; a CPU call
  never enters ``cd/kernels/logmatmul.py``; in a fresh process CPU ``log_matmul`` + ``constrained_marginals`` leave
  triton out of ``sys.modules`` and log exactly one decision record (O7) containing "cpu".
- **Layout guard and dispatch** (criterion 8): ``logmatmul_layout_ok``'s truth table on meta tensors (accepting the
  layouts the core makes: strided live-block views, a stride-0 broadcast batch, vec-mat, mat-vec, m = 1 — a guard too
  wide and a guard too narrow are both red); ``use_logmatmul`` on fake CUDA operands (a meta tensor that reports
  ``cuda``, everything else delegated): dispatch exactly when CUDA ∧ ``kernels`` ∧ triton ∧ layout.
- **Interpreter** (criteria 3b, 3c, 4; ruling O2), one ``TRITON_INTERPRET=1`` subprocess (``_interp_t0077c``).
- **PTX** (task "Tests to write"; ``_ptx_t0077c``): AOT for ``GPUTarget("cuda", 120, 32)``, no GPU: per launch both
  options ``False``; no ``.ftz`` other than libdevice ``__nv_expf``'s own ``ex2.approx.ftz.f32`` (see the deviation
  note on the row); no non-FTZ ``ex2.approx.f32`` (``tl.exp``); ``max.NaN.f32`` present; no ``mma``; the launches per
  call a constant independent of S (criterion 2). A control row shows the harness tells those apart on a toy kernel.
- **AST** (criterion 7, logmatmul.py): triton imported only inside function bodies (and imported there); no ``tl.dot``;
  T-0077b's forbidden constructs absent.

Regime: CPU, torch 2.13.0+cpu, triton 3.7.1, numpy 2.4.6 (``uv.lock``), fp32, ``OMP_NUM_THREADS=4``.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import itertools
import json
import logging
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
from tests.cd.unit import _frozen_t0077c as c
from tests.cd.unit import _interp_t0077c as it
from tests.cd.unit import _ptx_t0077c as px
from tests.cd.unit.test_cd_kernel_imports import forbidden, triton_imports

REPO = Path(__file__).resolve().parents[3]


def _env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(c.PLUGIN_SRC)])
    env.setdefault("OMP_NUM_THREADS", "4")
    env.update(extra)
    return env


def _git_blob(rev: str, rel: str) -> bytes:
    blob = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=REPO, capture_output=True)
    assert blob.returncode == 0, f"{rel} not at {rev}: {blob.stderr!r}"
    return blob.stdout


# --------------------------------------------------------------------------- the frozen copy


def test_frozen_copies_are_byte_identical_to_b53457a() -> None:
    bad = []
    manifest = (c.FROZEN_DIR / "MANIFEST.md").read_text()
    for name, want in c.FROZEN_SHA256.items():
        got = c.file_sha256(c.frozen_path(name))
        if got != want:
            bad.append(f"{name}: sha256 {got} != {want}")
        if (
            c.file_sha256(c.frozen_path(name))
            != kb.hashlib.sha256(_git_blob(c.FROZEN_COMMIT, c.plugin_rel(name))).hexdigest()
        ):
            bad.append(f"{name}: not the blob at {c.FROZEN_COMMIT}")
        if want not in manifest:
            bad.append(f"MANIFEST lacks {name} {want}")
    assert not bad, "\n".join(bad)


def test_frozen_modules_load_side_by_side_and_wire_the_frozen_kernels() -> None:
    for name in c.FROZEN_SHA256:
        f = c.frozen(name)
        assert f.__name__ == f"{c.FROZEN_PKG}.{name}"
        assert Path(f.__file__).resolve() == c.frozen_path(name).resolve()
    mg = c.frozen("marginals")
    assert mg._kernel_backend is c.frozen("kernels.backend")
    assert mg._kernel_entropy is c.frozen("kernels.entropy")
    assert c.frozen("tree").log_matmul is c.frozen("semiring").log_matmul
    assert c.frozen("semiring").log_matmul is not c.live("semiring").log_matmul


# --------------------------------------------------------------------------- negatives


#: not owned by T-0077c (task "Owned files"): byte-identical to the branch base b53457a
_NOT_OWNED_AT_T0077C = (
    *(
        c.plugin_rel(n)
        for n in (
            "tree",
            "emission",
            "marginals",
            "mapdecode",
            "draw",
            "steps",
            "requests",
            "xargs",
            "sampler",
            "budget",
            "record",
            "record_schema",
            "kernels.__init__",
            "kernels.entropy",
        )
    ),
    "plugin/astra_vllm_plugin/pyproject.toml",
    "pyproject.toml",
    "uv.lock",
    "services/versions.lock",
    "local_astra/bench/thresholds.py",
    "scripts/gpu/cd_microbench.py",
    "scripts/gpu/cd_b13.py",
    "scripts/locked/t0071a_a1.sh",
    "scripts/locked/t0071a_b13.sh",
    "scripts/locked/t0077a_ab.sh",
    "scripts/locked/t0077a_tier2.sh",
    "scripts/locked/t0077b_ab.sh",
    "scripts/locked/t0077b_tier2.sh",
)


#: Every pinned path, the three T-0081 plugin files included (T-0094b criterion 7): over T-0077c's closed range no
#: later-task exemption is needed. ``LATER_TASK_OWNED_PLUGIN`` and its exemption row are retired with the working-tree
#: comparison; the "a one-byte change is still reported" half is the helper's own row,
#: ``tests/unit/test_guards_use_fixed_refs.py::test_changed_reports_every_modified_deleted_and_added_file_under_its_paths_only``.
NOT_OWNED = _NOT_OWNED_AT_T0077C


#: T-0077c's merge commit 26ce165; BASE = ``git merge-base 26ce165^1 26ce165^2`` (= the frozen base b53457a), TIP =
#: ``26ce165^2`` (T-0094 criterion 2). Over this closed range no later-task exemption is needed.
T0077C_MERGE = "26ce165"
T0077C_BASE = "b53457a"
T0077C_TIP = "06dddef"
#: the path set of the T-0077c scope guard
T0077C_PATHS = ("local_astra",)


def test_local_astra_is_untouched() -> None:
    """``local_astra/**`` not owned: nothing under it differs over T-0077c's closed range (T-0094). The untracked-file
    half of the old guard is retired: it only meant something while the branch was open."""
    gr.prove_range(T0077C_BASE, T0077C_TIP, T0077C_MERGE)
    assert T0077C_BASE == c.FROZEN_COMMIT
    changed = gr.changed(T0077C_BASE, T0077C_TIP, *T0077C_PATHS)
    assert changed == [], changed


def test_the_t0077c_range_guard_has_power() -> None:
    """T-0094 criterion 4 (adversarial): over ``BASE..6a8600b`` the predicate reports ``local_astra/cli.py``."""
    gr.require_commits(T0077C_BASE, gr.LATER)
    assert "local_astra/cli.py" in gr.changed(T0077C_BASE, gr.LATER, *T0077C_PATHS)


@pytest.mark.parametrize("rel", NOT_OWNED)
def test_not_owned_file_is_byte_identical_to_the_branch_base(rel: str) -> None:
    """Not owned by T-0077c: ``rel`` does not differ between T-0077c's BASE (the branch base b53457a) and TIP (T-0094b:
    a closed range, never the working tree)."""
    assert T0077C_BASE == c.FROZEN_COMMIT
    gr.prove_range(T0077C_BASE, T0077C_TIP, T0077C_MERGE)
    assert gr.changed(T0077C_BASE, T0077C_TIP, rel) == [], (
        f"{rel} changed in {T0077C_BASE}..{T0077C_TIP}"
    )


def test_the_not_owned_guard_has_power_and_keeps_the_t0081_files() -> None:
    """T-0094b criteria 5 and 7 (adversarial): over ``BASE..6a8600b`` the predicate over ``NOT_OWNED`` reports T-0081's
    three plugin files (``requests``, ``xargs``, ``sampler``), reached through the tuple; a predicate reporting nothing,
    or a tuple that dropped one of them, fails here."""
    gr.require_commits(T0077C_BASE, gr.LATER)
    t0081 = {
        p for p in NOT_OWNED if p.rsplit("/", 1)[-1] in ("requests.py", "xargs.py", "sampler.py")
    }
    assert len(t0081) == 3, t0081
    assert gr.changed(T0077C_BASE, gr.LATER, *NOT_OWNED) == sorted(t0081)


def _semiring_defs(src: str) -> dict[str, ast.AST]:
    tree = ast.parse(src)
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    out[f"={t.id}"] = node
    return out


#: criterion 12: byte-identical (source text), not only AST-equal
BYTE_IDENTICAL = (
    "compensated_sum",
    "_finish",
    "two_sum",
    "segment_sumexp_split",
    "segment_logsumexp",
    "segment_layout",
    "layout_reduce_blocks",
    "layout_reduce_upper",
    "layout_sumexp_split",
    "layout_logsumexp",
    "SegmentLevel",
    "SegmentLayout",
    "safe_log",
    "sumexp_add",
)


#: The owned change of ``semiring.py`` (T-0077c ruling O3): the one top-level name allowed to differ.
SEMIRING_ALLOWED = frozenset({"log_matmul"})


def _semiring_diffs(
    base_src: str, tip_src: str, allowed: frozenset[str] = SEMIRING_ALLOWED
) -> list[str]:
    """Criterion 6/12's predicate over two sources of ``semiring.py``: top-level names added or removed, names outside
    ``allowed`` whose AST differs, and ``BYTE_IDENTICAL`` names whose source text differs."""
    base, tip = _semiring_defs(base_src), _semiring_defs(tip_src)
    bad = [f"{n}: added or removed" for n in sorted(set(base) ^ set(tip))]
    for name, node in base.items():
        if name in allowed or name not in tip:
            continue
        if ast.dump(node) != ast.dump(tip[name]):
            bad.append(f"{name}: AST differs")
        bare = name.lstrip("=")
        if bare in BYTE_IDENTICAL and ast.get_source_segment(
            base_src, node
        ) != ast.get_source_segment(tip_src, tip[name]):
            bad.append(f"{name}: source text differs")
    return bad


def test_semiring_functions_other_than_log_matmul_are_unchanged() -> None:
    """Criterion 6/12: every top-level function, class and constant of ``semiring.py`` other than ``log_matmul``
    AST-equal between T-0077c's BASE (b53457a) and TIP blobs (T-0094b: never BASE against the live file); the named
    ones byte-identical; none removed, none added except imports."""
    gr.prove_range(T0077C_BASE, T0077C_TIP, T0077C_MERGE)
    rel = c.plugin_rel("semiring")
    base_src = _git_blob(T0077C_BASE, rel).decode()
    tip_src = _git_blob(T0077C_TIP, rel).decode()
    assert base_src != tip_src, "premise: two blobs"
    assert _semiring_diffs(base_src, tip_src) == []
    assert set(BYTE_IDENTICAL) <= set(_semiring_defs(base_src))


def test_the_semiring_allowance_is_not_too_strict() -> None:
    """T-0094b criterion 6, too strict (and the differential's power): with the allowance emptied the predicate over
    T-0077c's own pair reports exactly ``log_matmul``; a pair reduced to TIP vs TIP fails here."""
    gr.prove_range(T0077C_BASE, T0077C_TIP, T0077C_MERGE)
    rel = c.plugin_rel("semiring")
    base_src = _git_blob(T0077C_BASE, rel).decode()
    tip_src = _git_blob(T0077C_TIP, rel).decode()
    assert _semiring_diffs(base_src, tip_src, frozenset()) == ["log_matmul: AST differs"]


# --------------------------------------------------------------------------- the CPU path (criterion 6)


def _cpu_cases() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for r in c.REGIMES:
        out[f"{r}91"] = c.regime_instance(r, 91, 0)
    out["two_band"] = c.two_band(91, 0)
    out["all_sentinel"] = c.all_sentinel(33, 0)
    out["nan_inf"] = c.nan_inf(33, 0)
    out["near_dead"] = c.near_dead()
    out["live_block"] = c.live_block_views(3, 33, 40, 0)
    out["vec_mat"] = c.vec_mat(3, 33, 40, 0)
    out["mat_vec"] = c.mat_vec(3, 33, 40, 0)
    a, b = c.broadcast_batch(3, 5, 7, 4, 0)
    out["host_broadcast_2d_3d"] = (a[0], b)  # [n, m] ⊗ [nb, m, k]: semiring broadcasts on the host
    g = torch.Generator().manual_seed(77)
    out["batch_4d"] = (
        -3 * torch.rand(2, 3, 6, 9, generator=g),
        -3 * torch.rand(3, 9, 5, generator=g),
    )
    out["fp64"] = (
        -3 * torch.rand(2, 4, 6, generator=g, dtype=torch.float64),
        -3 * torch.rand(2, 6, 3, generator=g, dtype=torch.float64),
    )
    out["mixed_dtype"] = (
        -3 * torch.rand(2, 4, 6, generator=g),
        -3 * torch.rand(2, 6, 3, generator=g, dtype=torch.float64),
    )
    return out


CPU_CASES = _cpu_cases()


@functools.cache
def _frozen_out(name: str, compensated: bool) -> torch.Tensor:
    a, b = CPU_CASES[name]
    return c.frozen("semiring").log_matmul(a, b, compensated=compensated)


@pytest.mark.parametrize("compensated", [True, False], ids=["comp", "plain"])
@pytest.mark.parametrize("name", list(CPU_CASES))
def test_cpu_log_matmul_without_keyword_equals_the_branch_base(
    name: str, compensated: bool
) -> None:
    """Criterion 6 (negative; true at the base and must stay true)."""
    a, b = CPU_CASES[name]
    got = c.live("semiring").log_matmul(a, b, compensated=compensated)
    assert c.bit_diff(_frozen_out(name, compensated), got) is None


@pytest.mark.parametrize("kernels", [True, False])
@pytest.mark.parametrize("compensated", [True, False], ids=["comp", "plain"])
@pytest.mark.parametrize("name", list(CPU_CASES))
def test_cpu_log_matmul_with_kernels_keyword_equals_the_branch_base(
    name: str, compensated: bool, kernels: bool
) -> None:
    """Criterion 6: the new keyword changes nothing on CPU, either value."""
    c.require_kernels_keyword()
    a, b = CPU_CASES[name]
    got = c.live("semiring").log_matmul(a, b, compensated=compensated, kernels=kernels)
    assert c.bit_diff(_frozen_out(name, compensated), got) is None


def test_cpu_inner_mismatch_still_raises_value_error() -> None:
    c.require_kernels_keyword()
    with pytest.raises(ValueError, match="inner dims differ"):
        c.live("semiring").log_matmul(torch.zeros(2, 3), torch.zeros(4, 2), kernels=True)


def test_kernels_keyword_is_keyword_only_with_default_true() -> None:
    import inspect

    p = inspect.signature(c.live("semiring").log_matmul).parameters.get("kernels")
    assert p is not None, "semiring.log_matmul has no `kernels` parameter"
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is True, p
    q = inspect.signature(c.live("semiring").log_matmul).parameters.get("compensated")
    assert q is not None and q.kind is inspect.Parameter.KEYWORD_ONLY and q.default is False


@pytest.mark.parametrize("kernels", [True, False])
def test_cpu_calls_never_enter_logmatmul_py(kernels: bool) -> None:
    """Criterion 6: on CPU the kernel is never dispatched, ``kernels=True`` included (product and whole marginals)."""
    c.lm_entry()  # the entry exists (else "never called" proves nothing)
    a, b = CPU_CASES["live_block"]
    with c.entry_calls() as seen:
        c.live("semiring").log_matmul(a, b, compensated=True, kernels=kernels)
        kb.marginals_on("live", fz.build_inst("s91_b128_nfa"), "cpu")
    assert seen == [], f"CPU work called {seen} in cd/kernels/logmatmul.py"


def test_entry_call_spy_sees_calls_into_logmatmul_py(tmp_path: Path) -> None:
    """Control for ``entry_calls``: a function compiled with ``logmatmul.py``'s path is seen, one elsewhere is not."""
    target = str((c.KERNELS_DIR / "logmatmul.py").resolve())
    ns: dict[str, Any] = {}
    exec(compile("def probe():\n    return 1\n", target, "exec"), ns)
    other: dict[str, Any] = {}
    exec(compile("def probe():\n    return 2\n", str(tmp_path / "logmatmul.py"), "exec"), other)
    with c.entry_calls() as seen:
        other["probe"]()
        ns["probe"]()
    assert seen == ["probe"]


_CPU_PROCESS = textwrap.dedent(
    """
    import json, logging, sys
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
    sem = c.live("semiring")
    a, b = c.live_block_views(3, 33, 40, 0)
    for kw in ({}, {"kernels": True}, {"kernels": False}):
        sem.log_matmul(a, b, compensated=True, **kw)
    inst = fz.build_inst("s31_b64_dfa")
    for kw in ({}, {"kernels": True}):
        kb.marginals_on("live", inst, "cpu", **kw)
    print("ASTRA_T0077C_CPU " + json.dumps({
        "records": records,
        "triton_loaded": sorted(m for m in sys.modules if m.split(".")[0] == "triton"),
    }))
    """
)


def test_cpu_process_logs_one_decision_and_never_imports_triton() -> None:
    """Criteria 6, 8 / O7: a fresh process runs CPU ``log_matmul`` (3 calls, ``kernels`` default/True/False) then CPU
    ``constrained_marginals`` (2 calls): exactly one record on ``astra_vllm_plugin.cd.kernels`` across both families,
    containing "cpu", naming a family ("log_matmul" or the H(q) family's "H(q)"), not ``ASTRA_CD ``-prefixed; triton
    never in ``sys.modules``."""
    c.require_kernels_keyword()
    proc = subprocess.run(
        [sys.executable, "-c", _CPU_PROCESS],
        cwd=REPO,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("ASTRA_T0077C_CPU "))
    out = json.loads(line.split(" ", 1)[1])
    recs = out["records"]
    assert len(recs) == 1, f"expected exactly one decision record across both families, got {recs}"
    msg = recs[0]["msg"]
    assert recs[0]["name"] == c.LOGGER and not msg.startswith("ASTRA_CD "), recs
    assert "cpu" in msg.lower(), f"the record does not name the CPU reason: {msg}"
    assert "log_matmul" in msg or "H(q)" in msg, f"the record names no family: {msg}"
    assert out["triton_loaded"] == [], f"CPU work imported triton: {out['triton_loaded']}"


# --------------------------------------------------------------------------- layout guard and dispatch (criterion 8)


def _meta(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    stride: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if stride is None:
        return torch.empty(shape, dtype=dtype, device="meta")
    return torch.empty_strided(shape, stride, dtype=dtype, device="meta")


F64 = torch.float64
#: name -> (a3, b3, the guard's answer). Accepted: what the core hands the entry (criterion 8: both fp32, 3-D
#: [nb, n, m] / [nb, m, k] after the host broadcast, every dim ≥ 1; any strides).
LAYOUTS: dict[str, tuple[torch.Tensor, torch.Tensor, bool]] = {
    "contiguous": (_meta((4, 33, 91)), _meta((4, 91, 17)), True),
    "live_block_views_row_stride_B": (
        _meta((3, 33, 33), stride=(2 * 40 * 40, 40, 1)),
        _meta((3, 33, 33), stride=(2 * 40 * 40, 40, 1)),
        True,
    ),
    "broadcast_batch_stride_0": (_meta((3, 5, 7), stride=(0, 7, 1)), _meta((3, 7, 4)), True),
    "vec_mat": (
        _meta((3, 1, 33), stride=(40, 40, 1)),
        _meta((3, 33, 33), stride=(1600, 40, 1)),
        True,
    ),
    "mat_vec": (
        _meta((3, 33, 33), stride=(1600, 40, 1)),
        _meta((3, 33, 1), stride=(40, 1, 40)),
        True,
    ),
    "transposed_operand": (_meta((2, 6, 5), stride=(30, 1, 6)), _meta((2, 5, 4)), True),
    "m1": (_meta((2, 8, 1)), _meta((2, 1, 8)), True),
    "nb1_n1_m1_k1": (_meta((1, 1, 1)), _meta((1, 1, 1)), True),
    "a_fp64": (_meta((2, 4, 6), F64), _meta((2, 6, 3)), False),
    "b_fp64": (_meta((2, 4, 6)), _meta((2, 6, 3), F64), False),
    "both_fp16": (_meta((2, 4, 6), torch.float16), _meta((2, 6, 3), torch.float16), False),
    "a_2d": (_meta((4, 6)), _meta((2, 6, 3)), False),
    "b_2d": (_meta((2, 4, 6)), _meta((6, 3)), False),
    "4d": (_meta((2, 2, 4, 6)), _meta((2, 2, 6, 3)), False),
    "batch_mismatch": (_meta((2, 4, 6)), _meta((3, 6, 3)), False),
    "inner_mismatch": (_meta((2, 4, 6)), _meta((2, 5, 3)), False),
    "nb0": (_meta((0, 4, 6)), _meta((0, 6, 3)), False),
    "n0": (_meta((2, 0, 6)), _meta((2, 6, 3)), False),
    "m0": (_meta((2, 4, 0)), _meta((2, 0, 3)), False),
    "k0": (_meta((2, 4, 6)), _meta((2, 6, 0)), False),
}


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_layout_guard_truth_table(name: str) -> None:
    """Criterion 8's guard, both directions: refuses what the kernel does not take, accepts every layout the core
    produces (a guard that demanded contiguity would send the live-block products to the torch path)."""
    a3, b3, want = LAYOUTS[name]
    ok = c.entry("kernels.backend", "logmatmul_layout_ok")
    assert ok(a3, b3) is want, f"logmatmul_layout_ok({name}) is not {want}"


class FakeCuda:
    """A meta tensor that reports CUDA: ``device``/``is_cuda`` faked, every other name delegated to the tensor."""

    def __init__(self, t: torch.Tensor) -> None:
        self._t = t

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", 0)

    @property
    def is_cuda(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._t, name)


@pytest.mark.parametrize(
    ("kernels", "triton_ok", "layout"),
    list(itertools.product((True, False), repeat=3)),
)
def test_use_logmatmul_truth_table_on_cuda_operands(
    monkeypatch: pytest.MonkeyPatch, kernels: bool, triton_ok: bool, layout: str | bool
) -> None:
    """Criterion 8: on CUDA operands, dispatch exactly when ``kernels`` is not False, triton imports and the layout
    guard passes (triton's availability faked through ``backend.triton_status``)."""
    be = c.live("kernels.backend")
    use = c.entry("kernels.backend", "use_logmatmul")
    if not triton_ok:
        monkeypatch.setattr(
            be, "triton_status", lambda: (None, "ModuleNotFoundError: blocked by the test")
        )
    a3, b3, _ = LAYOUTS["live_block_views_row_stride_B" if layout else "a_fp64"]
    got = use(FakeCuda(a3), FakeCuda(b3), kernels=kernels)
    assert got is (kernels and triton_ok and bool(layout)), (kernels, triton_ok, layout, got)


@pytest.mark.parametrize("kernels", [True, False])
def test_use_logmatmul_never_dispatches_cpu_operands(kernels: bool) -> None:
    a3, b3 = c.live_block_views(2, 5, 8, 0)
    assert c.entry("kernels.backend", "use_logmatmul")(a3, b3, kernels=kernels) is False


def test_use_kernels_stays_backward_compatible() -> None:
    """T-0077b's ``use_kernels(t, *, kernels)`` keeps its signature and its CPU answer."""
    import inspect

    uk = c.entry("kernels.backend", "use_kernels")
    params = inspect.signature(uk).parameters
    assert list(params)[:1] == ["t"] and params["kernels"].kind is inspect.Parameter.KEYWORD_ONLY
    assert uk(torch.zeros(2, 3), kernels=True) is False
    c.entry("kernels.backend", "use_logmatmul")  # and the second family exists beside it


# --------------------------------------------------------------------------- AST (criterion 7) on logmatmul.py


def _lm_src() -> str:
    p = c.KERNELS_DIR / "logmatmul.py"
    assert p.is_file(), "T-0077c not implemented: cd/kernels/logmatmul.py"
    return p.read_text()


def test_logmatmul_imports_triton_only_inside_function_bodies() -> None:
    src = _lm_src()
    assert triton_imports(src, outside_functions_only=True) == [], (
        "module-level triton import (M1/K3)"
    )
    assert triton_imports(src, outside_functions_only=False), (
        "logmatmul.py never imports triton (M1 builder)"
    )


def test_logmatmul_has_no_dot_atomics_autotune_config_writes_or_host_sync() -> None:
    """N2/N6/criterion 7: T-0077b's forbidden constructs, plus no ``tl.dot`` (TF32)."""
    src = _lm_src()
    bad = forbidden(src)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Attribute) and node.attr in ("dot", "dot_scaled"):
            bad.append(f"{node.lineno}: {node.attr}")
    assert bad == [], bad


def test_dot_walker_control() -> None:
    src = "def f(a, b):\n    return tl.dot(a, b)\n"
    assert [n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)] == ["dot"]


# --------------------------------------------------------------------------- interpreter (criteria 3b, 3c, 4)


@pytest.fixture(scope="module")
def interp() -> dict[str, Any]:
    """One ``TRITON_INTERPRET=1`` subprocess for every interpreter row."""
    proc = subprocess.run(
        [sys.executable, "-m", "tests.cd.unit._interp_t0077c"],
        cwd=REPO,
        env=_env(TRITON_INTERPRET="1"),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(it.MARKER + " ")), None)
    assert line is not None, f"no {it.MARKER} line (exit {proc.returncode}):\n{proc.stderr[-6000:]}"
    out: dict[str, Any] = json.loads(line[len(it.MARKER) + 1 :])
    assert out["triton_interpret"] == "1"
    return out


def _no_error(interp: dict[str, Any]) -> None:
    assert interp["error"] is None, f"the interpreter run failed:\n{interp['error']}"


@pytest.mark.parametrize("key", it.bits_keys())
def test_interpreter_entry_is_the_spec_bit_for_bit(interp: dict[str, Any], key: str) -> None:
    """Criterion 3c: the entry vs the spec with numpy fp64 exp/log rounded once (O2), fp32 bits and NaN positions; a
    new ``[nb, n, k]`` fp32 output; inputs untouched."""
    _no_error(interp)
    row = interp["bits"].get(key)
    assert row is not None, f"{key} not computed"
    assert row["ok"], f"{key}: {row['detail']}"


@pytest.mark.parametrize("key", it.tile_keys())
def test_interpreter_bits_do_not_depend_on_the_tiles(interp: dict[str, Any], key: str) -> None:
    """Criterion 3b: every ``(block_n, block_k)`` gives the spec's bits."""
    _no_error(interp)
    rows = interp["tiles"].get(key)
    assert rows is not None and len(rows) >= len(it.TILE_BLOCKS), rows
    bad = {b: r["detail"] for b, r in rows.items() if not r["ok"]}
    assert bad == {}, f"{key}: {bad}"


@pytest.mark.parametrize("key", it.bar_keys())
def test_interpreter_bars(interp: dict[str, Any], key: str) -> None:
    """Criterion 4 under the interpreter (outer 64, seed 0): compensated meets the wide/gauged bar; plain fails the
    compensated wide bar on T-0065's inner-64 input."""
    _no_error(interp)
    row = interp["bars"].get(key)
    assert row is not None, f"{key} not computed"
    print(f"ASTRA_T0077C_INTERP_BAR {key}: {row['detail']}")
    assert row["ok"], f"{key}: {row['detail']}"


# --------------------------------------------------------------------------- PTX (AOT, no GPU)


def _ptx(*args: str) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "tests.cd.unit._ptx_t0077c", *args],
        cwd=REPO,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(px.MARKER + " ")), None)
    assert line is not None, f"no {px.MARKER} line (exit {proc.returncode}):\n{proc.stderr[-6000:]}"
    out: dict[str, Any] = json.loads(line[len(px.MARKER) + 1 :])
    return out


LIBDEVICE_FTZ = {"ex2.approx.ftz.f32"}


def test_ptx_harness_control_on_a_toy_kernel() -> None:
    """The harness can tell the landmines apart (toy kernel, same AOT path): libdevice exp with reflect-FTZ off has
    only ``__nv_expf``'s own ``ex2.approx.ftz.f32``; reflect-FTZ on adds ``fma.*.ftz``; ``tl.exp`` lowers to a
    non-FTZ ``ex2.approx.f32``; ``tl.maximum(..., propagate_nan=ALL)`` gives ``max.NaN.f32``."""
    out = _ptx("--toy")
    assert out["error"] is None, out["error"]
    lib, tlx, ftz = (out["kernels"][k][0] for k in ("libdevice", "tl_exp", "reflect_ftz_on"))
    assert set(lib["ftz_ops"]) == LIBDEVICE_FTZ and not lib["ex2_approx_nonftz"] and lib["max_nan"]
    assert tlx["ex2_approx_nonftz"]
    assert set(ftz["ftz_ops"]) > LIBDEVICE_FTZ, ftz["ftz_ops"]
    assert lib["options"] == {"enable_fp_fusion": False, "enable_reflect_ftz": False}
    # log (reviewer A finding 1, 2026-10-06): neither libdevice __nv_logf nor tl.log emits an approximate lg2; in
    # triton 3.7.1 at sm_120 tl.log lowers to the *same* instruction stream as libdevice log (measured: identical
    # 52-instruction bodies), so a tl.log mutant is equivalent and no PTX rule can or needs to reject it; an
    # approximate log (inline lg2.approx.f32; neither tl.log nor tl.log2 emits it in 3.7.1) is what the rule rejects.
    llog, tlog, asm = (out["kernels"][k][0] for k in ("libdevice_log", "tl_log", "lg2_approx_asm"))
    assert llog["lg2_approx"] == [] and tlog["lg2_approx"] == [], (
        llog["lg2_approx"],
        tlog["lg2_approx"],
    )
    assert asm["lg2_approx"] == ["lg2.approx.f32"], asm["lg2_approx"]
    print(
        f"ASTRA_T0077C_PTX_LOG tl.log == libdevice log instruction stream: {tlog['instr_sha'] == llog['instr_sha']}"
    )


@pytest.fixture(scope="module")
def ptx() -> dict[str, Any]:
    return _ptx()


@pytest.mark.parametrize("variant", ["comp", "plain"])
def test_ptx_of_every_launch(ptx: dict[str, Any], variant: str) -> None:
    """Task "Tests to write", PTX inspection, for every launch at three shapes. **Deviation from the task's literal
    "0 .ftz" (reported):** libdevice's ``__nv_expf`` itself contains one ``ex2.approx.ftz.f32`` on its range-reduced
    argument whatever the options (control row), so the row requires no ``.ftz`` *other than* that one; reflect-FTZ
    on adds ``fma.rn.ftz``/``cvt.ftz`` (killed here)."""
    assert ptx["error"] is None, f"the PTX run failed:\n{ptx['error']}"
    bad = []
    max_nan = False
    counts = {}
    for shape in px.SHAPES:
        rows = ptx["kernels"].get(f"{variant}-{shape}")
        assert isinstance(rows, list), f"{variant}-{shape}: {rows}"
        assert rows, f"{variant}-{shape}: the entry launched no @triton.jit kernel"
        counts[shape] = len(rows)
        for r in rows:
            where = f"{variant}-{shape}:{r['name']}"
            if r["options"] != {"enable_fp_fusion": False, "enable_reflect_ftz": False}:
                bad.append(f"{where}: launch options {r['options']}")
            if not set(r["ftz_ops"]) <= LIBDEVICE_FTZ:
                bad.append(f"{where}: .ftz ops {r['ftz_ops']}")
            if r["ex2_approx_nonftz"]:
                bad.append(f"{where}: ex2.approx.f32 (tl.exp, not libdevice __nv_expf)")
            if r["lg2_approx"]:
                bad.append(
                    f"{where}: {r['lg2_approx']} (an approximate log, not libdevice __nv_logf)"
                )
            if r["mma"]:
                bad.append(f"{where}: {r['mma']}")
            max_nan |= r["max_nan"]
    assert not bad, bad
    assert max_nan, f"{variant}: no max.NaN.f32 in any launch (the anchor max must propagate NaN)"
    print(f"ASTRA_T0077C_LAUNCHES {variant} {counts}")
    assert len(set(counts.values())) == 1, (
        f"launches per call depend on the shape: {counts} (criterion 2)"
    )


# --------------------------------------------------------------------------- the CUDA module and its wrapper agree


def test_cuda_module_row_count_matches_the_wrappers_k() -> None:
    """The CUDA module has exactly K_c rows, and the host wrapper pins the same K_c and digest set (a row added to
    one and not the other would make the tier-2 count check vacuous or impossible)."""
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
            "tests/cd/cuda/test_cd_logmatmul_cuda.py",
        ],
        cwd=REPO,
        env=_env(ASTRA_CD_TESTS="1"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    ids = [ln for ln in proc.stdout.splitlines() if "test_cd_logmatmul_cuda.py::" in ln]
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    from tests.cd.cuda import test_cd_logmatmul_cuda as cuda_mod
    from tests.contract import test_vllm_cd_logmatmul_gpu as wrapper

    assert len(ids) == cuda_mod.LOGMATMUL_CUDA_PASSED == wrapper.LOGMATMUL_CUDA_PASSED == 43, len(
        ids
    )
    assert tuple(kb.INSTANCES) == wrapper.CORE
    assert set(cuda_mod.DIGEST_CASES) == set(wrapper.EXPECTED_DIGESTS)
    assert len(cuda_mod.DIGEST_CASES) == len(set(cuda_mod.DIGEST_CASES))


def test_cuda_module_quiets_torch_trace_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Tier 2 at f3c678c lost the criterion-4 assertion text to 4 950 ``torch.__trace`` DEBUG lines from the
    ``torch.compile`` row. The CUDA module's ``quiet_torch_logs`` (autouse per row) must keep them out of pytest's log
    capture while leaving our own logger alone. Control: the same compile without it does flood (CPU)."""
    from tests.cd.cuda import test_cd_logmatmul_cuda as cuda_mod

    def compiled_calls() -> int:
        torch._dynamo.reset()
        f = torch.compile(lambda x: torch.exp(x) + 1.0)
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            logging.getLogger("torch.__trace").setLevel(
                logging.DEBUG
            )  # what torch does when tracing is on
            f(torch.rand(8))
        return sum(r.name.startswith("torch.__trace") for r in caplog.records)

    flood = compiled_calls()
    with cuda_mod.quiet_torch_logs():
        quiet = compiled_calls()
        assert not logging.getLogger(c.LOGGER).disabled
    print(f"ASTRA_T0077C_TRACE_LINES without {flood} with {quiet}")
    assert flood > 0, (
        "control: torch.compile produced no torch.__trace records (cannot show the fix works)"
    )
    assert quiet == 0, quiet
    assert c.LOGGER not in cuda_mod.NOISY_LOGGERS
    trace = logging.getLogger("torch.__trace")
    assert not trace.disabled, "quiet_torch_logs did not restore torch.__trace"


@pytest.mark.parametrize("order", ["calls_outer", "variants_outer"])
def test_cuda_module_spies_nest(order: str) -> None:
    """Reviewer (a4d5f87): ``entry_variants`` replaced ``entry_calls``'s profiler instead of chaining, so the CUDA rows'
    ``with c.entry_calls() as seen, entry_variants() as variants:`` left ``seen == []`` (≈ 32 of 43 GPU rows red
    whatever the kernel does). The exact nesting, in both orders, around direct entry calls on CPU (outside the
    interpreter the entry raises at launch, after its call event): both spies see the calls, the ``compensated`` flags
    are read in order, and the previous profiler is restored."""
    from tests.cd.cuda import test_cd_logmatmul_cuda as cuda_mod

    entry = c.lm_entry()
    a, b = torch.rand(1, 4, 4), torch.rand(1, 4, 4)

    def calls() -> None:
        for comp in (True, False):
            # no GPU: the launch fails after the call event fired
            with contextlib.suppress(Exception):
                entry(a, b, compensated=comp)

    before = sys.getprofile()
    if order == "calls_outer":
        with c.entry_calls() as seen, cuda_mod.entry_variants() as variants:
            calls()
    else:
        with cuda_mod.entry_variants() as variants, c.entry_calls() as seen:
            calls()
    assert sys.getprofile() is before
    assert seen.count("log_matmul") == 2, seen
    assert variants == [True, False], variants


#: ruling P7: the T-0073 CUDA rows whose uncompensated control must be the torch path (``kernels=False``) under O9
P7_CONTROLS = {
    "tests/cd/cuda/test_cd_cuda_numerics.py": (
        "test_log_matmul_compensated_wide_on_cuda",
        "test_log_matmul_compensated_gauged_on_cuda",
    ),
    "tests/cd/cuda/test_cd_cuda_repeats.py": ("test_compensated_log_matmul_on_the_served_path",),
}


def _kw(call: ast.Call, name: str) -> object:
    for k in call.keywords:
        if k.arg == name:
            return k.value.value if isinstance(k.value, ast.Constant) else "<non-literal>"
    return "<absent>"


@pytest.mark.parametrize(("rel", "fn"), [(r, f) for r, fns in P7_CONTROLS.items() for f in fns])
def test_p7_uncompensated_controls_take_the_torch_path(rel: str, fn: str) -> None:
    """Ruling P7 (CPU-visible guard for gpu rows): under O9 ``semiring.log_matmul(compensated=False)`` on CUDA runs
    the compensated kernel, so each T-0073 uncompensated control must pass ``kernels=False`` (else it cannot be red
    and the row fails for the wrong reason, as at tier 2 a671719). Every ``compensated=False`` call in the row has
    ``kernels=False``, and there is at least one."""
    tree = ast.parse((REPO / rel).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn)
    controls = [
        n for n in ast.walk(node) if isinstance(n, ast.Call) and _kw(n, "compensated") is False
    ]
    assert controls, f"{rel}::{fn} has no compensated=False control call"
    bad = [n.lineno for n in controls if _kw(n, "kernels") is not False]
    assert bad == [], f"{rel}::{fn}: compensated=False without kernels=False at lines {bad}"


@pytest.mark.parametrize(
    "rel", ["tests/cd/cuda/test_cd_cuda_numerics.py", "tests/cd/cuda/test_cd_cuda_repeats.py"]
)
def test_t0073_cuda_modules_quiet_torch_trace_per_row(rel: str) -> None:
    """The T-0073 CUDA modules (gpu-only) carry an autouse fixture entering ``tests/cd/_quiet_logs.quiet_torch_logs``:
    CPU-visible, so a module-wide gpu skip cannot hide its absence."""
    tree = ast.parse((REPO / rel).read_text())
    found = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            auto = any(
                isinstance(d, ast.Call)
                and any(
                    k.arg == "autouse" and getattr(k.value, "value", None) is True
                    for k in d.keywords
                )
                for d in n.decorator_list
            )
            uses = any(isinstance(x, ast.Name) and x.id == "quiet_torch_logs" for x in ast.walk(n))
            if auto and uses:
                found.append(n.name)
    assert found, f"{rel}: no autouse fixture enters quiet_torch_logs"


def test_shared_quiet_torch_logs_stops_the_trace_flood(caplog: pytest.LogCaptureFixture) -> None:
    """``tests/cd/_quiet_logs.quiet_torch_logs``: torch.compile's torch.__trace records reach pytest's capture without
    it (control) and not with it; state restored."""
    from tests.cd._quiet_logs import quiet_torch_logs

    def records() -> int:
        torch._dynamo.reset()
        f = torch.compile(lambda x: torch.exp(x) + 2.0)
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            logging.getLogger("torch.__trace").setLevel(logging.DEBUG)
            f(torch.rand(8))
        return sum(r.name.startswith("torch.__trace") for r in caplog.records)

    flood = records()
    with quiet_torch_logs():
        quiet = records()
    assert flood > 0 and quiet == 0, (flood, quiet)
    assert not logging.getLogger("torch.__trace").disabled
