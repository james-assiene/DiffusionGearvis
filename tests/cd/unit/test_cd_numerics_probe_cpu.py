"""T-0073 criterion 1: the numerics probe, tier 1 (host ``cd`` group, CPU, before any GPU time).

``tests/container_scripts/astra_cd_numerics_probe.py --device cpu --small`` must run in ≤ 60 s and print one JSON
object that the schema in ``tests/numerics_probe_schema.py`` accepts, with every section P1-P7 present; a section
that raises is recorded as ``{"error": …}`` and the probe continues; it exits non-zero only when a section is
missing; it imports neither scipy nor vllm (checked from ``sys.modules`` of the probe's own process). The staged
recomputations of ``log_matmul`` / ``constrained_marginals`` inside the probe must reproduce the plugin bit for
bit on CPU, or its stage dumps and stage-swap attributions describe some other computation.

The launch this protects is expensive (a human ``/launch`` on the 5090), so the rows also pin that the probe
builds exactly the failing rows' instances (seeds and builders of ``tests/cd/cuda/test_cd_cuda_repeats.py``) and
that the schema check rejects every way a payload can be incomplete.
"""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.numerics_probe_schema import (
    MARKER,
    P6_STAGES,
    SECTION_KEYS,
    SECTIONS,
    extract,
    problems,
)

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "tests" / "container_scripts" / "astra_cd_numerics_probe.py"
BUDGET_S = 60.0
#: the probe takes its import paths from the environment, as in the container (/repo:/repo/plugin/astra_vllm_plugin)
PYTHONPATH = os.pathsep.join([str(REPO), str(REPO / "plugin" / "astra_vllm_plugin")])

#: runs the probe as ``__main__`` in a fresh interpreter, then reports which forbidden modules that process loaded
_WRAP = """
import json, runpy, sys
probe, args = sys.argv[1], sys.argv[2:]
sys.argv = [probe, *args]
code = 0
try:
    runpy.run_path(probe, run_name="__main__")
except SystemExit as exc:
    code = exc.code
mods = sorted(m for m in sys.modules if m.split(".")[0] in ("scipy", "vllm"))
print("PROBE_MODULES " + json.dumps(mods))
print("PROBE_EXIT " + json.dumps(code))
"""


def _run(*args: str) -> dict[str, Any]:
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", _WRAP, str(PROBE), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": PYTHONPATH},
        timeout=600,
        check=False,
    )
    elapsed = time.monotonic() - t0
    tail = {
        ln.split(" ", 1)[0]: ln.split(" ", 1)[1]
        for ln in proc.stdout.splitlines()
        if ln.startswith("PROBE_")
    }
    assert "PROBE_EXIT" in tail, (
        f"the probe process died\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    )
    return {
        "elapsed": elapsed,
        "exit": json.loads(tail["PROBE_EXIT"]),
        "modules": json.loads(tail["PROBE_MODULES"]),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


@pytest.fixture(scope="module")
def small_run() -> dict[str, Any]:
    run = _run("--device", "cpu", "--small")
    run["payload"] = extract(run["stdout"])
    return run


# --------------------------------------------------------------------------- criterion 1


def test_small_cpu_run_is_complete_within_60_s(small_run: dict[str, Any]) -> None:
    assert small_run["exit"] == 0, small_run["stderr"][-4000:]
    assert small_run["elapsed"] <= BUDGET_S, (
        f"--device cpu --small took {small_run['elapsed']:.1f} s > 60 s"
    )
    payload = small_run["payload"]
    bad = problems(payload)
    assert bad == [], "\n".join(bad)
    assert sorted(payload["sections"]) == sorted(SECTIONS)
    assert payload["regime"]["device"] == "cpu" and payload["regime"]["small"] is True
    for name in SECTIONS:
        sec = payload["sections"][name]
        assert "error" not in sec and "skipped" not in sec, (name, sec)
    # P6 reaches every intermediate, with the counts D1 reads
    stages = payload["sections"]["P6"]["stages"]["cpu"]
    assert sorted(stages) == sorted(P6_STAGES)
    assert tuple(payload["sections"]["P6"]["stage_order"]) == P6_STAGES
    for name in ("total", "q_raw", "entropy"):
        for tensor_stats in stages[name].values():
            if tensor_stats["dtype"].startswith("float"):
                for k in ("nan", "posinf", "neginf", "sentinel", "subnormal", "zero"):
                    assert isinstance(tensor_stats[k], int), (name, k)


def test_probe_imports_neither_scipy_nor_vllm(small_run: dict[str, Any]) -> None:
    """The image's Python 3.12 cannot load the host's cp311 scipy (T-0067 tier-2 harness), and vllm is not needed."""
    assert small_run["modules"] == [], f"the probe process imported {small_run['modules']}"
    assert small_run["payload"]["regime"]["forbidden_modules_loaded"] == []


def test_staged_recomputation_reproduces_the_plugin_bit_for_bit_on_cpu(
    small_run: dict[str, Any],
) -> None:
    sec = small_run["payload"]["sections"]
    assert sec["P4"]["staged_equals_plugin"]["cpu"] is True
    for name in ("P5", "P6"):
        eq = sec[name]["staged_equals_plugin"]
        assert eq["cpu"] is True, (name, eq)
    for temp, fields in sec["P5"]["staged_equals_plugin"]["cpu_fields"].items():
        assert set(fields) == {"log_z", "log_a", "log_b", "q", "entropy"} and all(
            fields.values()
        ), (temp, fields)
    assert all(sec["P6"]["staged_equals_plugin"]["cpu_fields"].values())
    assert sec["P6"]["staged_equals_plugin"]["cpu_fp64"] is True


def test_a_section_that_raises_is_recorded_and_the_probe_continues() -> None:
    run = _run("--device", "cpu", "--small", "--sections", "P2,P3,P6", "--force-error", "P3")
    assert run["exit"] == 0, "an errored section must not change the exit status"
    payload = extract(run["stdout"])
    sec = payload["sections"]
    assert set(sec) == set(SECTIONS), "every section is present, errored or skipped"
    assert "forced error in P3" in sec["P3"]["error"]
    assert all(k in sec["P2"] for k in SECTION_KEYS["P2"]) and "error" not in sec["P2"]
    assert all(k in sec["P6"] for k in SECTION_KEYS["P6"]) and "error" not in sec["P6"], (
        "P6 ran after P3 raised"
    )
    bad = problems(payload)
    assert any(b.startswith("P3 errored") for b in bad), bad
    assert any(b.startswith("P1 skipped") for b in bad), bad
    assert problems(payload, allow_skipped=True) == [b for b in bad if "skipped" not in b]


def test_the_probe_exits_non_zero_only_when_a_section_is_missing() -> None:
    run = _run("--device", "cpu", "--small", "--sections", "P2", "--force-missing", "P7")
    assert run["exit"] == 2
    payload = extract(run["stdout"])
    assert "P7" not in payload["sections"]
    assert "P7 missing" in problems(payload, allow_skipped=True)


# --------------------------------------------------------------------------- the probe diagnoses the failing rows' instances


def _probe_module() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("t0073_numerics_probe", PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    plugin = str(REPO / "plugin" / "astra_vllm_plugin")
    if plugin not in sys.path:  # what PYTHONPATH provides in the subprocess runs
        sys.path.insert(0, plugin)
    spec.loader.exec_module(mod)
    return mod


def test_probe_instances_are_the_failing_rows_instances() -> None:
    """Built here the way ``tests/cd/cuda/test_cd_cuda_repeats.py`` builds them (copied from the rows, not from
    the probe): (c) ``default_rng(0)``, (a) ``Random(7)`` + ``Generator(3)`` at L 256 / V 4096, (b) ``Random(5)``."""
    import torch

    from tests.cd.unit._core_support import random_logits, random_toy

    probe = _probe_module()
    # (c)
    g = np.random.default_rng(0)
    a = g.uniform(-24, -16, (64, 128))
    b = g.uniform(-24, -16, (128, 64))
    a[g.random(a.shape) < 0.3] = -1e30
    b[g.random(b.shape) < 0.3] = -1e30
    pa, pb = probe.c_operands(0, True)
    assert np.array_equal(pa, a) and np.array_equal(pb, b)
    # (a) at both temperatures, full size
    for temp in (0.8, 0.4):
        rng = random.Random(7)
        toy = random_toy(rng, V=4096, S=12, nfa=True)
        gen = torch.Generator().manual_seed(3)
        raw = torch.randn(256, 4096, generator=gen, dtype=torch.float64) * 40.0
        x64 = (30.0 * torch.tanh(raw / 30.0)) / temp
        p, px = probe.a_problem(temp, small=False)
        assert p.toy == toy and torch.equal(p.x, x64) and np.array_equal(px, x64.numpy())
        assert p.start == [0] and p.end == sorted(toy.finals) and p.L == 256
    # (b)
    rng = random.Random(5)
    toy = random_toy(rng, V=6, S=4, nfa=True)
    x = torch.tensor(random_logits(rng, 6, 6, span=8.0), dtype=torch.float32)
    keep = x.topk(3, dim=-1).indices
    filt = torch.full_like(x, float("-inf")).scatter(1, keep, x.gather(1, keep))
    filt[0, keep[0, 0]] = -1e20
    p = probe.b_problem(5)
    assert p.toy == toy and torch.equal(p.x, filt) and p.L == 6 and p.start == [0]


def test_p4_row_as_written_control_matches_the_rows_own_computation(
    small_run: dict[str, Any],
) -> None:
    """The probe's "row_as_written" numbers for (c) on CPU are what the CUDA row's own code computes on CPU
    (oracle on the unrounded float64 inputs, ulp from ``np.spacing``, live = oracle > -1e20)."""
    import torch

    from tests.cd.unit._core_support import core

    semiring = core("semiring")
    g = np.random.default_rng(0)
    a = g.uniform(-24, -16, (64, 128))
    b = g.uniform(-24, -16, (128, 64))
    a[g.random(a.shape) < 0.3] = semiring.SENTINEL
    b[g.random(b.shape) < 0.3] = semiring.SENTINEL
    s = a[:, :, None] + b[None, :, :]
    m = s.max(axis=-2)
    want = m + np.log(np.exp(s - m[:, None, :]).sum(axis=-2))
    ulp = np.spacing(np.abs(want).astype(np.float32)).astype(np.float64)
    live = want > -1e20
    got = semiring.log_matmul(
        torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32), compensated=True
    )
    e = (np.abs(got.double().numpy() - want) / ulp)[live]
    ctl = small_run["payload"]["sections"]["P4"]["oracle_controls"]["seed0"]["cpu"]
    row = ctl["row_as_written (fp64-input oracle)"]["compensated"]
    assert row["max"] == pytest.approx(float(np.max(e)), rel=1e-12)
    assert row["frac_gt_0.51"] == pytest.approx(float(np.mean(e > 0.51)), rel=1e-12)


# --------------------------------------------------------------------------- the schema check itself (adversarial)


def _good(small_run: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(small_run["payload"])


def test_schema_rejects_every_incomplete_payload(small_run: dict[str, Any]) -> None:
    assert problems(_good(small_run)) == []

    p = _good(small_run)
    del p["sections"]["P5"]
    assert "P5 missing" in problems(p)

    p = _good(small_run)
    p["sections"]["P4"] = {"error": "Traceback …\nRuntimeError: boom"}
    assert any(b.startswith("P4 errored") for b in problems(p))

    p = _good(small_run)
    p["sections"]["P7"] = {"skipped": "not selected"}
    assert any(b.startswith("P7 skipped") for b in problems(p))

    p = _good(small_run)
    del p["sections"]["P3"]["chain"]
    assert "P3.chain missing" in problems(p)

    p = _good(small_run)
    del p["sections"]["P6"]["stages"]["cpu"]["tree2"]
    assert any("P6.stages[cpu] lacks ['tree2']" in b for b in problems(p))

    p = _good(small_run)
    p["sections"]["P6"]["staged_equals_plugin"]["cpu"] = False
    assert any(b.startswith("P6: staged recomputation") for b in problems(p))

    p = _good(small_run)
    p["regime"]["forbidden_modules_loaded"] = ["scipy"]
    assert any("forbidden modules" in b for b in problems(p))

    p = _good(small_run)
    p["sections"]["P4"]["arms"]["seed0"]["cpu"]["fp64_exp"] = {
        "error": "Traceback …\nRuntimeError: arm"
    }
    assert "P4 nested error at /arms/seed0/cpu/fp64_exp" in problems(p)

    p = _good(small_run)
    p["sections"]["P1"]["ops"][2]["cpu/eager"] = {"error": "boom"}
    assert "P1 nested error at /ops[2]/cpu/eager" in problems(p)

    p = _good(small_run)
    del p["verdicts"]
    assert "verdicts block missing (a, b, c)" in problems(p)

    p = _good(small_run)
    del p["regime"]["device_name"]
    assert "regime.device_name missing" in problems(p)


def test_extract_demands_exactly_one_marker_line(small_run: dict[str, Any]) -> None:
    line = f"{MARKER} {json.dumps(small_run['payload'])}"
    assert extract(f"noise\n{line}\nmore noise") == small_run["payload"]
    with pytest.raises(AssertionError, match="found 0"):
        extract("no marker here")
    with pytest.raises(AssertionError, match="found 2"):
        extract(f"{line}\n{line}")


# --------------------------------------------------------------------------- probe v2 (T-0073 D2 / D3)


def _decode(a: dict[str, Any]) -> np.ndarray:
    import base64

    dt = {"float32": "<f4", "float64": "<f8"}[a["dtype"]]
    return np.frombuffer(base64.b64decode(a["b64"]), dtype=dt).reshape(a["shape"])


def test_p5_saved_arrays_recompute_the_d2_score_offline(small_run: dict[str, Any]) -> None:
    """D3: the saved H(q), stock H(p) and fp64 references let the D2 score be recomputed without the probe."""
    p5 = small_run["payload"]["sections"]["P5"]
    for temp in ("T0.4", "T0.8"):
        inst = p5["instances"][temp]
        row = inst["cpu"]
        h = _decode(row["arrays"]["entropy_q"]).astype(np.float64)
        hp32 = _decode(row["arrays"]["stock_Hp32"])
        oracle = _decode(inst["arrays"]["oracle_H_rounded"])
        hp64 = _decode(inst["arrays"]["stock_Hp64_rounded"])
        assert h.shape == oracle.shape == hp32.shape == hp64.shape == (inst["L"],)
        d2 = float(np.max(np.abs(h - oracle) / (10.0 * np.abs(hp32 - hp64) + 1e-6)))
        assert row["worst_err_over_bar"] == pytest.approx(d2, rel=1e-12)
        assert row["scores"]["oracle_rounded/bar_rounded (D2)"][
            "worst_err_over_bar"
        ] == pytest.approx(d2, rel=1e-12)
        assert set(row["scores"]) == {
            "oracle_rounded/bar_rounded (D2)",
            "oracle_rounded/bar_unrounded",
            "oracle_unrounded/bar_unrounded (pre-D2 row)",
            "oracle_unrounded/bar_rounded",
        }
        for arm in ("repeats", "deterministic_segments"):
            rep = p5[arm][temp]["cpu"]
            assert "error" not in rep, rep
            runs = _decode(rep["entropy_runs"])
            assert runs.shape == (rep["n"], inst["L"])
            assert rep["all_bitwise_repeatable"] is True, (
                arm,
                temp,
                rep["first_differing_stage_per_repeat"],
            )


def test_deterministic_segment_sum_matches_an_fsum_oracle_and_is_restored() -> None:
    """The D3 arm's fixed-order segment sum against ``math.fsum`` per segment (independent), and the probe's
    keyword swap leaves the plugin's own functions in place afterwards."""
    import math

    import torch

    probe = _probe_module()
    g = torch.Generator().manual_seed(11)
    rows, width, n_seg = 5, 300, 17
    vals = torch.rand(rows, width, generator=g) * torch.exp(
        -10 * torch.rand(rows, width, generator=g)
    )
    seg = torch.randint(0, n_seg, (width,), generator=g)
    seg[:3] = 16  # segment 16 non-empty; some segments may stay empty (must give 0)
    got = probe.det_segment_sum(vals, seg, n_seg).double().numpy()
    v64 = vals.double().numpy()
    for r in range(rows):
        for k in range(n_seg):
            want = math.fsum(v64[r, j] for j in range(width) if int(seg[j]) == k)
            if want == 0.0:
                assert got[r, k] == 0.0, (r, k, got[r, k])  # an empty segment sums to exactly 0
            else:
                assert abs(got[r, k] - want) <= 1e-6 * want, (r, k, got[r, k], want)
    names = ("segment_sumexp_split",)
    before = {
        m: getattr(mod, n)
        for m, mod in (("sr", probe.sr), ("em", probe.em), ("mg", probe.mg))
        for n in names
    }
    tm = probe.mg._token_marginals
    with probe.segment_variant("deterministic"):
        assert probe.sr.segment_sumexp_split is probe.det_segment_sumexp_split
        assert probe.mg._token_marginals is probe.det_token_marginals
    after = {
        m: getattr(mod, n)
        for m, mod in (("sr", probe.sr), ("em", probe.em), ("mg", probe.mg))
        for n in names
    }
    assert after == before and probe.mg._token_marginals is tm
