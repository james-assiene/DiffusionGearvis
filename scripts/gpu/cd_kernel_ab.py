"""T-0077c criterion 9 (umbrella criterion 3): same-process A/B of the per-step CD core, frozen vs live with the fused
``log_matmul`` kernel dispatched, in the pinned container.

    uv run --group cd python scripts/gpu/cd_kernel_ab.py             # host side (the /launch command)
    python3 /repo/scripts/gpu/cd_kernel_ab.py --inner /cdpayloads   # inside the container (the host runs it)

GPU-spending by definition (scripts/gpu/, T-0016). No engine boots. The host side is T-0071a's microbench host
side, reused by import (``cd_microbench``: the pinned image and its digest, the four real T-0063 dev-cut automata
compiled on the host and handed in read-only at ``/cdpayloads``, one ``docker run --rm --gpus all --network none``
with the repository read-only at ``/repo``); it refuses a dirty tree (naming the paths), an unreadable image digest
or an evidence directory that already exists (exit 2), and writes ``logs/t0077c-ab-<sha>/report.txt``,
``lines.jsonl`` and ``container.log`` before its exit code (0: every bucket measured and every gate below held;
1: otherwise; 2: refused).

Container side, per bucket 64/128/256/512, in one process: path **A** = the frozen b53457a copies
(``tests/cd/oracle/core_frozen_t0077c/``, MANIFEST.md: the T-0077b merge — torch ``log_matmul`` and the frozen
``kernels``' Triton H(q) tail), path **B** = the live ``astra_vllm_plugin.cd`` modules, whose ``semiring.log_matmul``
dispatches every CUDA product to ``cd/kernels/logmatmul.py`` (``kernels=True``, the default), both on the same tables
(``requests._tables_for``) and the same inputs (the microbench's: CUDA, fp32, TF32 "high" set globally, CL 256,
V 262 144, seed 0, random logits ×4, one L_eff candidate = CL, stock knobs, bf16 embedding of the model's shape).
Both paths run T-0077b's H(q) kernel, so A vs B isolates ``log_matmul``.

Checks before timing (CUDA):
- N3 ``torch.equal`` A vs B (gated; T-0077a's 9 names): ``log_transfer``'s output, the max-plus tree's live block,
  the MAP's tokens/score/``empty``, the draw's tokens/states/``empty`` and the CUDA generator's state after
  ``joint_draw`` (the draw per ruling O5: a flip fails the gate and is adjudicated, never edited).
- N4 (gated as T-0077a's L6; replaces T-0077b's criterion-2 equality set, task criterion 9): max |dq| ≤ 1e-6 and
  max |dH| ≤ 1e-5 of ``constrained_marginals`` on rows < L_eff, A vs B (facts key ``n4``; |d log Z| and the max |Δ|
  of live ``log_a``/``log_b`` entries are reported, not gated).
- Dead sets (gated): ``log_a <= DEAD`` and ``log_b <= DEAD``, A vs B, ``torch.equal`` (facts key ``dead_set``).
- Repeat (gated, T-0077a ruling L6): live ``constrained_marginals`` (q, entropy, log_a, log_b) and ``joint_draw``
  (tokens, states, ``empty``, generator state after; a fresh generator with the same seed per call) called
  ``REPEATS`` times in the same process, each ``torch.equal`` to the first.
- V2 (an error of the bucket, which fails the gate): the tables are padded (S < B); the live pass-1 tree ran on S;
  the dispatch decisions on this bucket's real operands are the kernels (``kernels.backend.use_kernels`` on
  ``q_raw``, ``kernels.backend.use_logmatmul`` on the level-0 product); live ``constrained_marginals`` entered
  ``hq_tail`` at its own call site; and the live path really entered ``cd/kernels/logmatmul.py``: every call of
  ``kernels.logmatmul.log_matmul`` made through ``semiring``'s own call site is recorded with its ``compensated``
  flag (T-0077b reviewer B's lesson: observe the call site; T-0077c reviewer B finding 1: per variant), and every
  call of ``log_matmul`` through the live ``tree``/``marginals`` module globals is counted. Ruling O9 (on CUDA
  ``semiring`` runs every product, plain calls included, in the compensated kernel): in the live pass-1 up-sweep,
  the live N4 call and every live repeat call, each live ``semiring.log_matmul`` call must reach the kernel (as
  many entries as calls, ≥ 1) and every entry must be compensated; one spied live call each of ``lm_plain`` (a
  plain semiring call) and ``lm_comp`` (outside the timed region) must be exactly one compensated entry. The
  counts (total, plain = 0, compensated) are facts key ``dispatch``.

Timing (gated: no row of B more than 10 % slower than A): per row, ``ROUNDS`` rounds alternating the order (AB, BA,
...), each a block of the T-0072 probe's ``_timed`` (CUDA events; median, p95, peak added memory of one call);
per-path median = median of its block medians; ratio = median over rounds of B/A of the paired block medians.
Ruling M4 (= T-0077a L8): if any row of a bucket is over 1.10×, that bucket's whole A/B is re-run once in the same
process and the gate reads the re-run; both runs are reported. Components: ``emission`` (log_transfer + up_sweep),
``pass1_tree`` (up_sweep), ``marginals_pass1`` (the ungauged down-sweep), ``pass2`` (gauged leaves + compensated
up-sweep + down-sweep), ``marginals_full``, ``map``, ``map_sweep`` (max_up_sweep on each path's leaves), ``draw``,
``hq_tail`` (the tail alone, ``q_raw`` -> ``(q, entropy)``: A the frozen ``kernels.entropy.hq_tail``, B the live
one; ``q_raw`` is this bucket's real one, captured from the frozen ``_token_marginals``, and every call gets a fresh
copy made before its timed region, the same setup on both paths), ``lm_plain`` (the level-0 pass-1 product
``[L/2, S, S] ⊗ [L/2, S, S]`` of the real normalised leaves, as ``up_sweep`` hands it over: strided live-block
views, a plain call; A the frozen ``semiring.log_matmul`` (torch plain), B the live one, which on CUDA runs the
compensated kernel, ruling O9), ``lm_comp`` (the same on pass 2's normalised gauged leaves, ``compensated=True``);
and ``whole_call`` per arm (``steps.prepare`` + δ* with every function ``steps``
binds from the A/B'd modules routed to path A or B; ``end_projection`` = class_max_of + constrained_map + δ*). Peak
added memory: B's ``whole_call`` peak must not exceed A's, no tolerance (T-0077a ruling L9). The tree, pass-2 and
MAP components are reported against (|S|/B)³ (not gated).

Reported, not gated (criterion 10): both kernels' tiles/BLOCK, launches per call and launch options, and, in
seconds, the first live pass-1 up-sweep and the first live ``constrained_marginals`` of each bucket. On the first
bucket of the process the up-sweep includes the JIT compiles of the matrix-product specialisations and
``constrained_marginals`` those of the vec-mat/mat-vec ones (all compensated under ruling O9; later buckets may
compile new shape specialisations too).

Gated per launch: N3 9 × 4 = 36, N4 (dq, dH) 2 × 4 = 8, dead sets 2 × 4 = 8, live repeat 8 × 4 = 32, timed rows
(11 components + 5 whole calls) 16 × 4 = 64, whole-call peak 5 × 4 = 20, one container line per bucket 4: 172. Plus,
per bucket, the V1 completeness checks (the N3, N4, dead-set and repeat sets; the 16 timed rows of the gated run
and, under M4, of the superseded first run) and V2 (the ``dispatch`` observation included).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import cd_microbench as mb  # scripts/gpu/ is sys.path[0] on the host and in the container

MARKER = "ASTRA_T0077C_AB_JSON"
FROZEN_PKG = "tests.cd.oracle.core_frozen_t0077c"
FROZEN_COMMIT = "b53457a"
OWNED = ("emission", "tree", "draw", "mapdecode", "marginals", "semiring")
#: (name in ``steps``, module, attribute): every binding ``steps`` holds of a function of the A/B'd modules
STEPS_BINDINGS = (
    ("joint_draw", "draw", "joint_draw"),
    ("class_log_weights_from", "emission", "class_log_weights_from"),
    ("class_mass", "emission", "class_mass"),
    ("log_transfer", "emission", "log_transfer"),
    ("class_max_of", "mapdecode", "class_max_of"),
    ("constrained_map", "mapdecode", "constrained_map"),
    ("constrained_marginals", "marginals", "constrained_marginals"),
    ("up_sweep", "tree", "up_sweep"),
)
ARMS = ("end_projection", "perstep_mf", "perstep_mar", "perstep_mar_sc", "perstep_mar_qargmax")
COMPONENTS = (
    "emission",
    "pass1_tree",
    "marginals_pass1",
    "pass2",
    "marginals_full",
    "map",
    "map_sweep",
    "draw",
    "hq_tail",
    "lm_plain",
    "lm_comp",
)
#: components whose work is a dense state-cubed product: reported against (|S|/B)^3
CUBIC = ("pass1_tree", "pass2", "map_sweep")
ROUNDS = 4
N = 200
WARMUP = 20
ROW_BUDGET_S = 60.0
MAX_SLOWDOWN = 1.10
#: the N3 gates ``measure_bucket`` measures; a facts dict with any other set fails (V1)
N3_NAMES = (
    "log_transfer",
    "maxplus_tree_live_block",
    "map_tokens",
    "map_score",
    "map_empty",
    "draw_tokens",
    "draw_states",
    "draw_empty",
    "generator_state",
)
#: N4 on CUDA (T-0077a's L6): gated bars of ``constrained_marginals`` A vs B on rows < L_eff (facts key ``n4``)
N4_BARS = {"dq": 1e-6, "dh": 1e-5}
#: dead sets (``<= DEAD``) of these ``constrained_marginals`` fields, A vs B, ``torch.equal`` (facts key ``dead_set``)
DEAD_SET_NAMES = ("log_a", "log_b")
#: the timed rows of one A/B run: every component and one whole call per arm (V1)
EXPECTED_ROWS = tuple(("components", c) for c in COMPONENTS) + tuple(
    (a, "whole_call") for a in ARMS
)
#: same-process calls of the live repeat row (ruling L6: at least 2)
REPEATS = 3
REPEAT_FIELDS = (
    "q",
    "entropy",
    "log_a",
    "log_b",
    "draw_tokens",
    "draw_states",
    "draw_empty",
    "generator_state",
)


# --------------------------------------------------------------------------- the report (pure)


def verdict_problems(facts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every gate failure in the container's per-bucket facts (empty list = PASS)."""
    out: list[str] = []
    seen = set()
    for f in facts:
        b = f.get("bucket")
        seen.add(b)
        if f.get("error"):
            out.append(f"bucket {b}: {str(f['error']).strip().splitlines()[-1]}")
            continue
        n3 = f.get("n3") or {}
        if set(n3) != set(N3_NAMES):  # V1: every N3 gate measured, no more, no fewer
            out.append(f"bucket {b}: N3 gates measured {sorted(n3)} != expected {sorted(N3_NAMES)}")
        for name, ok in n3.items():
            if ok is not True:
                out.append(f"bucket {b}: N3 {name} differs frozen vs live")
        # N4 (L6): every bar measured and met; a missing, non-numeric or NaN value fails
        n4 = f.get("n4")
        if not isinstance(n4, Mapping):
            out.append(f"bucket {b}: N4 not measured (no n4 facts)")
        else:
            for name, bar in N4_BARS.items():
                v = n4.get(name)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                    out.append(f"bucket {b}: N4 {name} not measured ({v!r})")
                elif v > bar:
                    out.append(f"bucket {b}: N4 {name} {v:.3g} > {bar:g} (live vs frozen)")
        # dead sets (V1: the whole set, no more, no fewer)
        ds = f.get("dead_set") or {}
        if set(ds) != set(DEAD_SET_NAMES):
            out.append(
                f"bucket {b}: dead-set gates measured {sorted(ds)} != expected {sorted(DEAD_SET_NAMES)}"
            )
        for name, ok in ds.items():
            if ok is not True:
                out.append(f"bucket {b}: dead set of {name} differs live vs frozen ({ok!r})")
        # V2: the log_matmul kernel observed dispatched on the live path
        dmap = f.get("dispatch") or {}
        disp = dmap.get("log_matmul")
        if isinstance(disp, bool) or not isinstance(disp, int) or disp < 1:
            out.append(
                f"bucket {b}: V2: the live path was not observed entering the log_matmul kernel "
                f"(dispatch count {disp!r})"
            )
        # per variant, required (V1; measure_bucket always emits both): under ruling O9 every kernel entry from
        # semiring is compensated, so compensated ≥ 1 and plain exactly 0; a missing key is a problem
        for key, ok_fn, want in (
            ("log_matmul_comp", lambda v: v >= 1, "≥ 1"),
            ("log_matmul_plain", lambda v: v == 0, "0 (ruling O9)"),
        ):
            v = dmap.get(key)
            if isinstance(v, bool) or not isinstance(v, int) or not ok_fn(v):
                out.append(f"bucket {b}: V2: log_matmul kernel {key} count {v!r}, want {want}")
        # V1: every timed row present, in the gated run and (M4/L8) in the superseded first run
        for label, key in (("gated run", "rows"), ("first run (L8)", "rows_first")):
            if key == "rows_first" and key not in f:
                continue
            got = [(r.get("arm"), r.get("component")) for r in f.get(key) or []]
            if sorted(got, key=str) != sorted(EXPECTED_ROWS, key=str):
                out.append(
                    f"bucket {b}: {label} has {len(got)} timed rows, expected the "
                    f"{len(EXPECTED_ROWS)} of COMPONENTS + ARMS"
                )
        rep = f.get("repeat") or {}
        missing = [k for k in REPEAT_FIELDS if k not in rep]
        if missing:
            out.append(f"bucket {b}: live repeat row missing {missing}")
        for name, ok in rep.items():
            if ok is not True:
                out.append(
                    f"bucket {b}: live repeat {name} not bit-identical across {REPEATS} calls"
                )
        for r in f.get("rows") or []:
            ratio = r.get("ratio")
            if not isinstance(ratio, (int, float)):
                out.append(f"bucket {b}: {r.get('arm')}/{r.get('component')} not timed")
            elif ratio > MAX_SLOWDOWN:
                out.append(
                    f"bucket {b}: {r.get('arm')}/{r.get('component')} live/frozen {ratio:.3f} > "
                    f"{MAX_SLOWDOWN}"
                )
            if r.get("component") == "whole_call":
                pa, pb = r.get("peak_frozen"), r.get("peak_live")
                if isinstance(pa, int) and isinstance(pb, int) and pb > pa:
                    out.append(
                        f"bucket {b}: {r.get('arm')} peak added {pb / 2**20:.1f} MiB > frozen "
                        f"{pa / 2**20:.1f} MiB"
                    )
    out += [f"bucket {b}: no line from the container" for b in mb.BUCKETS if b not in seen]
    return out


def needs_rerun(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Ruling M4 (= T-0077a L8): a bucket whose A/B has any row over ``MAX_SLOWDOWN`` is re-run once."""
    return any(isinstance(r.get("ratio"), (int, float)) and r["ratio"] > MAX_SLOWDOWN for r in rows)


def format_report(facts: Sequence[Mapping[str, Any]], regime: str) -> str:
    lines = [
        f"T-0077c same-process A/B (criterion 9): frozen {FROZEN_COMMIT} (A, torch log_matmul) vs live with the "
        "log_matmul kernel dispatched (B)",
        regime.strip(),
        "",
        f"{'bucket':>6} {'|S|':>4}  {'arm':<20} {'component':<16} {'A ms':>10} {'B ms':>10} "
        f"{'B/A':>6} {'(S/B)^3':>8} {'A MiB':>9} {'B MiB':>9}  note",
    ]

    def num(x: Any, fmt: str) -> str:
        return format(float(x), fmt) if isinstance(x, (int, float)) else "-"

    for f in facts:
        b, s = f.get("bucket"), (f.get("automaton") or {}).get("n_states")
        if f.get("error"):
            lines.append(f"{b!s:>6} ERROR {str(f['error']).strip().splitlines()[-1][:160]}")
            continue
        timed = [("", r) for r in f.get("rows") or []]
        if f.get("rows_first"):
            timed = [("[first run; superseded by the L8 re-run] ", r) for r in f["rows_first"]]
            timed += [("[L8 re-run; gated] ", r) for r in f.get("rows") or []]
        for tag, r in timed:
            cube = (
                num((s / b) ** 3, ".3f")
                if r.get("component") in CUBIC and isinstance(s, int) and isinstance(b, int)
                else "-"
            )
            pa, pb = r.get("peak_frozen"), r.get("peak_live")
            lines.append(
                f"{b!s:>6} {s!s:>4}  {r.get('arm')!s:<20} {r.get('component')!s:<16} "
                f"{num(r.get('median_frozen_ms'), '.3f'):>10} {num(r.get('median_live_ms'), '.3f'):>10} "
                f"{num(r.get('ratio'), '.3f'):>6} {cube:>8} "
                f"{num(pa / 2**20 if isinstance(pa, int) else None, '.1f'):>9} "
                f"{num(pb / 2**20 if isinstance(pb, int) else None, '.1f'):>9}  {tag}{r.get('note') or ''}"
            )
        n3 = f.get("n3") or {}
        lines.append(
            f"{b!s:>6} N3 (torch.equal, one device): "
            + ", ".join(f"{k} {'equal' if v is True else 'DIFFERS'}" for k, v in n3.items())
        )
        n4 = f.get("n4") or {}
        lines.append(
            f"{b!s:>6} N4 (gated, rows < L_eff): max|dq| {num(n4.get('dq'), '.3g')} (bar 1e-6), "
            f"max|dH| {num(n4.get('dh'), '.3g')} (bar 1e-5); reported: |d log Z| "
            f"{num(n4.get('dlogz'), '.3g')}, max|d log_a| {num(n4.get('dlog_a'), '.3g')}, max|d log_b| "
            f"{num(n4.get('dlog_b'), '.3g')} (live entries)"
        )
        ds = f.get("dead_set") or {}
        lines.append(
            f"{b!s:>6} dead sets (torch.equal, live vs frozen): "
            + (
                ", ".join(f"{k} {'equal' if v is True else 'DIFFERS'}" for k, v in ds.items())
                or "MISSING"
            )
        )
        lines.append(
            f"{b!s:>6} V2 log_matmul dispatch (entries into cd/kernels/logmatmul.py from the live path): "
            + (", ".join(f"{k} {v}" for k, v in (f.get("dispatch") or {}).items()) or "MISSING")
        )
        rp = f.get("repeat") or {}
        lines.append(
            f"{b!s:>6} live repeat ({REPEATS} calls, torch.equal): "
            + (
                ", ".join(f"{k} {'equal' if v is True else 'DIFFERS'}" for k, v in rp.items())
                or "MISSING"
            )
        )
        kr = f.get("kernel") or {}
        if kr:
            lines.append(
                f"{b!s:>6} H(q) kernel (reported): triton {kr.get('triton')}, BLOCK {kr.get('block')}, "
                f"{kr.get('launches')} launches per call, options {kr.get('options')}"
            )
            lines.append(
                f"{b!s:>6} log_matmul kernel (reported): tiles {kr.get('lm_tiles')} (lm_plain/lm_comp shape "
                f"{kr.get('lm_shape')}), {kr.get('lm_launches')} launch per call, num_warps "
                f"{kr.get('lm_num_warps')}; first live pass-1 up_sweep here "
                f"{num(kr.get('first_call_s'), '.3f')} s, first live constrained_marginals here "
                f"{num(kr.get('first_marginals_s'), '.3f')} s"
                + (
                    " (first in the process: the up-sweep includes the JIT compiles of the matrix "
                    "products; constrained_marginals those of the vec-mat/mat-vec (all compensated, O9); "
                    "later buckets may still compile new shape specialisations)"
                    if kr.get("first_in_process")
                    else " (compiles of shape specialisations new to this bucket included)"
                )
            )
        if f.get("rows_first"):
            lines.append(
                f"{b!s:>6} L8: a row was over {MAX_SLOWDOWN}x; the bucket's A/B was re-run once"
            )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- host side


def host(args: argparse.Namespace) -> int:
    dirty = mb._git("status", "--porcelain")
    if dirty:
        mb.die(2, f"refusing: the tree is dirty:\n{dirty}")
    commit = mb._git("rev-parse", "HEAD")
    out = mb.REPO / "logs" / f"t0077c-ab-{commit[:7]}"
    if out.exists():
        mb.die(2, f"refusing: {out} exists; move it aside first")
    image = mb.pinned_image()
    digest = mb.image_digest(image)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    with tempfile.TemporaryDirectory(prefix="t0077c-cdpayloads-") as tmp:
        manifest = mb.write_payloads(Path(tmp))
        argv = ["docker", "run", "--rm", "--gpus", "all", "--network", "none"]
        argv += ["-v", f"{mb.REPO}:/repo:ro", "-v", f"{tmp}:{mb.PAYLOADS}:ro", "-w", "/repo"]
        for k, v in {
            "PYTHONPATH": "/repo:/repo/plugin/astra_vllm_plugin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "ASTRA_CONTAINER_DIGEST": digest,
            "ASTRA_COMMIT": commit,
        }.items():
            argv += ["-e", f"{k}={v}"]
        argv += ["--entrypoint", "python3", image, "/repo/scripts/gpu/cd_kernel_ab.py", "--inner"]
        argv += [mb.PAYLOADS, "--n", str(args.n), "--warmup", str(args.warmup)]
        argv += ["--rounds", str(args.rounds), "--row-budget-s", str(args.row_budget_s)]
        print("running: " + shlex.join(argv), flush=True)
        t0 = time.monotonic()
        rc, stdout, stderr = mb.run_container(
            argv, name=f"t0077c-ab-{commit[:7]}", timeout_s=args.timeout_s
        )
        elapsed = time.monotonic() - t0

    out.mkdir(parents=True)
    facts = [json.loads(ln[len(MARKER) :]) for ln in stdout.splitlines() if ln.startswith(MARKER)]
    (out / "lines.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts))
    (out / "container.log").write_text(stdout + "\n--- stderr ---\n" + stderr)
    problems = verdict_problems(facts)
    inner = next((f.get("regime") for f in facts if f.get("regime")), {}) or {}
    kern = next((f.get("kernel") for f in facts if f.get("kernel")), {}) or {}
    for f in facts:
        for k, want in (("container_digest", digest), ("commit", commit)):
            got = (f.get("regime") or {}).get(k)
            if got not in (None, want):
                problems.append(f"bucket {f.get('bucket')}: inner {k} {got} != host's {want}")
    if rc != 0:
        problems.append(f"container exited {rc}: {stderr.strip()[-400:]}")
    automata = "; ".join(
        f"b{b} {m['task']} {m['id']} (|S| {m['n_states']})"
        for b, m in sorted(manifest.items(), key=lambda kv: int(kv[0]))
    )
    regime = (
        f"regime: container {image} ({digest}; no engine), torch {inner.get('torch')}, triton "
        f"{kern.get('triton')}, CUDA {inner.get('cuda')}, vllm {inner.get('vllm')}, {inner.get('device')}, "
        f"driver {driver}; CL {mb.CL}, V {mb.V}, fp32, TF32 {inner.get('matmul_precision')}, kernel launch "
        f"options {kern.get('options')}, {args.rounds} alternating rounds of n {max(1, args.n // args.rounds)} "
        f"after {max(1, args.warmup // args.rounds)} warm-up per path unless a row says otherwise (row budget "
        f"{args.row_budget_s:.0f} s), CUDA events, seed {mb.SEED}, automata (dev cuts, JSON whitespace): "
        f"{automata}; random logits x4, one L_eff candidate (= CL); host clean at {commit}; wall {elapsed:.0f} s"
    )
    report = format_report(facts, regime)
    report += "".join(f"PROBLEM: {p}\n" for p in problems)
    report += f"verdict: {'PASS' if not problems else 'FAIL'}\n"
    (out / "report.txt").write_text(report)
    print(report)
    print(f"evidence: {out}")
    return 0 if not problems else 1


# --------------------------------------------------------------------------- container side


class WithSetup(NamedTuple):
    """A timed row whose every call needs untimed preparation (``hq_tail``: a fresh copy of ``q_raw``)."""

    setup: Callable[[], Any]
    fn: Callable[[], Any]


def _timed_setup(row: WithSetup, n: int, warmup: int) -> dict[str, float]:
    """The T-0072 probe's ``_timed`` (CUDA events around the call, median, p95, peak added memory of one call),
    with ``row.setup()`` run before each call, outside the timed region (and before the peak's baseline)."""
    import torch

    for _ in range(warmup):
        row.setup()
        row.fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        row.setup()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        row.fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    row.setup()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    row.fn()
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - base
    return {
        "peak_mem_delta_bytes": int(peak_delta),
        "median_ms": times[len(times) // 2],
        "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
    }


def _paths() -> dict[str, dict[str, Any]]:
    return {
        "frozen": {m: importlib.import_module(f"{FROZEN_PKG}.{m}") for m in OWNED},
        "live": {m: importlib.import_module(f"astra_vllm_plugin.cd.{m}") for m in OWNED},
    }


@contextlib.contextmanager
def _steps_on(steps: Any, mods: Mapping[str, Any]) -> Iterator[None]:
    """Route every binding ``steps`` holds of a function of the A/B'd modules to ``mods`` (frozen or live)."""
    saved = [(name, getattr(steps, name)) for name, _, _ in STEPS_BINDINGS]
    try:
        for name, mod, attr in STEPS_BINDINGS:
            setattr(steps, name, getattr(mods[mod], attr))
        yield
    finally:
        for name, val in saved:
            setattr(steps, name, val)


@contextlib.contextmanager
def _capturing_q_raw(mg: Any, box: list[Any]) -> Iterator[None]:
    """While the block runs, every ``q_raw`` that ``mg.constrained_marginals`` forms is cloned into ``box``."""
    saved = mg._token_marginals

    def spy(*a: Any) -> Any:
        q = saved(*a)
        box.append(q.clone())
        return q

    mg._token_marginals = spy
    try:
        yield
    finally:
        mg._token_marginals = saved


@contextlib.contextmanager
def _counting_hq_tail(mg: Any, calls: list[int]) -> Iterator[None]:
    """While the block runs, every entry into ``hq_tail`` through ``mg``'s own call site (``mg._kernel_entropy``)
    appends 1 to ``calls``: proof that ``mg.constrained_marginals`` dispatched the kernel (V2, reviewer B 1)."""
    ent = mg._kernel_entropy
    saved = ent.hq_tail

    def spy(*a: Any, **kw: Any) -> Any:
        calls.append(1)
        return saved(*a, **kw)

    ent.hq_tail = spy
    try:
        yield
    finally:
        ent.hq_tail = saved


@contextlib.contextmanager
def _counting_logmatmul(klm: Any, calls: list[bool]) -> Iterator[None]:
    """While the block runs, every call of ``klm.log_matmul`` (the kernel entry, looked up on the module by
    ``semiring.log_matmul``'s own dispatch at each call) appends its ``compensated`` flag to ``calls``: proof that
    the live path entered ``cd/kernels/logmatmul.py``, per variant (V2; T-0077b reviewer B's lesson: observe the
    call site, not the decision; T-0077c reviewer B finding 1: a count that ignores the variant passes when the
    compensated products never dispatch)."""
    saved = klm.log_matmul

    def spy(*a: Any, **kw: Any) -> Any:
        calls.append(bool(kw.get("compensated", False)))
        return saved(*a, **kw)

    klm.log_matmul = spy
    try:
        yield
    finally:
        klm.log_matmul = saved


@contextlib.contextmanager
def _counting_semiring(mods: Sequence[Any], calls: list[int]) -> Iterator[None]:
    """While the block runs, every call of ``log_matmul`` through the module global each of ``mods`` holds (the
    live ``tree`` and ``marginals``: the only ``cd`` modules that call it) appends 1 to ``calls``."""
    saved = [(m, m.log_matmul) for m in mods]

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        def spy(*a: Any, **kw: Any) -> Any:
            calls.append(1)
            return fn(*a, **kw)

        return spy

    try:
        for m, fn in saved:
            m.log_matmul = wrap(fn)
        yield
    finally:
        for m, fn in saved:
            m.log_matmul = fn


def _require_compensated(kernel: Sequence[bool], semiring_calls: int, where: str) -> None:
    """V2 under ruling O9: ``semiring_calls`` ≥ 1 live ``semiring.log_matmul`` calls happened in ``where``, every one
    of them reached the kernel (as many kernel entries as calls), and every kernel entry is compensated (on CUDA
    ``semiring`` sends plain calls to the compensated kernel; a plain kernel entry from ``semiring`` is an error).
    Otherwise an ``AssertionError`` naming ``where`` (an error of the bucket, which fails the gate)."""
    n_comp = sum(kernel)
    n_plain = len(kernel) - n_comp
    if semiring_calls < 1 or len(kernel) != semiring_calls or n_plain:
        raise AssertionError(
            f"{where}: {semiring_calls} live semiring.log_matmul calls, {len(kernel)} entries into "
            f"cd/kernels/logmatmul.py ({n_comp} compensated, {n_plain} plain); want every call in the kernel and "
            "every entry compensated (ruling O9)"
        )


def _map_leaves(mods: Mapping[str, Any], path: str, tables: Any, cmax: Any, L: int) -> Any:
    """The MAP leaves each path's ``constrained_map`` builds: ``[L, S, S]`` on both paths (the frozen b53457a
    ``mapdecode`` is T-0077a's live-states code, ``pair_flat_live``)."""
    import torch

    S = tables.n_states
    flat = tables.pair_flat_live
    sent = mods["mapdecode"].SENTINEL
    wmax, _ = cmax
    ew = wmax.index_select(1, tables.edge_class)
    pw = torch.full((L, tables.n_pairs), sent, dtype=ew.dtype, device=ew.device)
    pw.scatter_reduce_(
        1, tables.edge_pair.unsqueeze(0).expand(L, -1), ew, reduce="amax", include_self=True
    )
    leaves = torch.full((L, S * S), sent, dtype=ew.dtype, device=ew.device)
    leaves.index_copy_(1, flat, pw)
    return leaves.view(L, S, S)


def _ab(
    probe: Any,
    fns: Mapping[str, Callable[[], Any] | WithSetup],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Alternating-round A/B of ``fns["frozen"]`` vs ``fns["live"]``: block medians, paired ratio, peaks."""
    import torch

    def sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def timed(fn: Callable[[], Any] | WithSetup, n: int, w: int) -> dict[str, Any]:
        if isinstance(fn, WithSetup):
            return _timed_setup(fn, n, w)
        return dict(probe._timed(fn, n, w))

    per_call: dict[str, float] = {}
    for path, fn in fns.items():  # one cold call each: sizes the blocks to the row budget
        call = fn.fn if isinstance(fn, WithSetup) else fn
        if isinstance(fn, WithSetup):
            fn.setup()
        sync()
        t0 = time.perf_counter()
        call()
        sync()
        per_call[path] = max(time.perf_counter() - t0, 1e-6)
    blocks = 2 * args.rounds
    n_blk = max(1, args.n // args.rounds)
    w_blk = max(1, args.warmup // args.rounds)
    worst = max(per_call.values())
    note = ""
    if worst * (n_blk + w_blk) * blocks > args.row_budget_s:
        n_new = max(3, int(0.8 * args.row_budget_s / (worst * blocks)) - 1)
        w_new = 1
        if (n_new, w_new) != (n_blk, w_blk):
            note = f"blocks of n {n_new} after {w_new} warm-up (cold call {worst * 1e3:.0f} ms)"
            n_blk, w_blk = n_new, w_new
    med: dict[str, list[float]] = {"frozen": [], "live": []}
    p95: dict[str, list[float]] = {"frozen": [], "live": []}
    peak: dict[str, int] = {"frozen": 0, "live": 0}
    for r in range(args.rounds):
        order = ("frozen", "live") if r % 2 == 0 else ("live", "frozen")
        for path in order:
            t = timed(fns[path], n_blk, w_blk)
            med[path].append(float(t["median_ms"]))
            p95[path].append(float(t["p95_ms"]))
            peak[path] = max(peak[path], int(t["peak_mem_delta_bytes"]))
    ratios = [b / a for a, b in zip(med["frozen"], med["live"], strict=True) if a > 0]
    return {
        "median_frozen_ms": statistics.median(med["frozen"]),
        "median_live_ms": statistics.median(med["live"]),
        "p95_frozen_ms": statistics.median(p95["frozen"]),
        "p95_live_ms": statistics.median(p95["live"]),
        "block_medians_frozen_ms": med["frozen"],
        "block_medians_live_ms": med["live"],
        "ratio": statistics.median(ratios) if ratios else None,
        "peak_frozen": peak["frozen"],
        "peak_live": peak["live"],
        "n_per_block": n_blk,
        "warmup_per_block": w_blk,
        "rounds": args.rounds,
        "note": note,
    }


#: whether a live pass-1 up-sweep has run in this process yet (the first one pays the log_matmul JIT compiles)
_FIRST_CALL_DONE: list[bool] = []


def measure_bucket(
    probe: Any, bucket: int, payload_dir: Path, args: argparse.Namespace, device: str = "cuda"
) -> dict[str, Any]:
    import math
    import types

    import numpy as np
    import torch
    from astra_vllm_plugin.cd import automaton as cd_automaton
    from astra_vllm_plugin.cd import budget, steps
    from astra_vllm_plugin.cd import requests as cd_requests
    from astra_vllm_plugin.cd.kernels import backend as kbackend
    from astra_vllm_plugin.cd.kernels import entropy as kentropy
    from astra_vllm_plugin.cd.kernels import logmatmul as klogmatmul

    fentropy = importlib.import_module(f"{FROZEN_PKG}.kernels.entropy")
    paths = _paths()
    manifest = json.loads((payload_dir / "manifest.json").read_text())[str(bucket)]
    payload = (payload_dir / f"b{bucket}.b64").read_text()
    wa = cd_automaton.decode_payload(payload)
    if int(wa.n_states_bucket) != bucket:
        raise AssertionError(f"payload bucket {wa.n_states_bucket} != {bucket}")
    dev = torch.device(device)
    tables = cd_requests._tables_for(manifest["payload_sha256"], wa, dev)
    S = int(tables.n_states)
    # V2 (first part), before any work or check: the tables are padded (S < B), so the live path is exercised;
    # otherwise this bucket is an error, which fails the gate
    if int(tables.bucket) != bucket or not int(tables.bucket) > S:
        raise AssertionError(
            f"bucket {bucket}: tables have n_states {S}, bucket {tables.bucket}; need S < bucket = "
            f"{bucket} (the live path would not be exercised)"
        )
    sv = np.zeros(bucket, dtype=bool)
    sv[: len(wa.start_vector)] = np.asarray(wa.start_vector, dtype=bool)[:bucket]
    a_cur = torch.from_numpy(sv).to(dev)
    budget.prepare_stop_tensor(mb.STOP_IDS, tables.distance_to_final.device)

    g = torch.Generator(device="cpu").manual_seed(mb.SEED)
    x = (torch.randn(mb.CL, mb.V, generator=g) * 4.0).to(dev)
    emb = (torch.randn(mb.V, mb.HIDDEN, generator=g) * 0.02).to(torch.bfloat16).to(dev)
    inner = types.SimpleNamespace(
        t_min=mb.T_MIN,
        t_max=mb.T_MAX,
        embed_weight=emb,
        sc_vocab_start=0,
        sc_vocab_end=mb.V,
        tp_size=1,
        normalizer=math.sqrt(mb.HIDDEN),
        entropy_bound=mb.STOCK_BOUND,
        confidence_threshold=mb.STOCK_THRESHOLD,
    )
    st = types.SimpleNamespace(
        canvas_length=mb.CL,
        max_denoising_steps=mb.MAX_STEPS,
        accepted_canvas_history_len=torch.zeros(4, dtype=torch.int64, device=dev),
    )
    step0 = torch.zeros((), dtype=torch.int64, device=dev)
    r_cap = torch.full((), mb.CL, dtype=torch.int64, device=dev)
    l_dev = torch.full((), mb.CL, dtype=torch.int64, device=dev)
    l_eff = mb.CL
    scaled = x / steps._temperature(inner, step0, float(mb.MAX_STEPS))
    b = budget.terminal_factor(tables, r_cap - l_eff)
    live = paths["live"]
    cm = live["emission"].class_mass(scaled, tables)
    cw = live["emission"].class_log_weights_from(cm)
    cmax = live["mapdecode"].class_max_of(scaled, tables)

    # per path: its own log_m, pass-1 tree, MAP leaves and the pass-2 inputs (shared: the frozen pass 1's)
    per: dict[str, dict[str, Any]] = {}
    # V2: the compensated flag of every entry into cd/kernels/logmatmul.py through the live path's own call site
    dispatched: list[bool] = []
    # the live modules whose module-global log_matmul the core calls (V2 counts semiring-level calls there)
    live_lm_mods = (paths["live"]["tree"], paths["live"]["marginals"])
    first_up_s = None
    for path, mods in paths.items():
        log_m = mods["emission"].log_transfer(scaled, tables, l_eff, class_mass=cm)
        up_calls: list[bool] = []
        up_sem: list[int] = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with (
            (
                _counting_logmatmul(klogmatmul, up_calls)
                if path == "live"
                else contextlib.nullcontext()
            ),
            (
                _counting_semiring(live_lm_mods, up_sem)
                if path == "live"
                else contextlib.nullcontext()
            ),
        ):
            tr = mods["tree"].up_sweep(log_m)
        torch.cuda.synchronize()
        if path == "live":
            first_up_s = time.perf_counter() - t0
            # pass 1 asks for plain products; on CUDA every one must reach the compensated kernel (O9)
            _require_compensated(
                up_calls, len(up_sem), f"bucket {bucket}: the live pass-1 up_sweep"
            )
            dispatched += up_calls
        root = tr.levels[-1][0]
        per[path] = {
            "log_m": log_m,
            "tree": tr,
            "la0": mods["marginals"]._mask_to_log(a_cur, root),
            "lb_end": mods["marginals"]._mask_to_log(b, root),
            "leaves": _map_leaves(mods, path, tables, cmax, mb.CL),
        }
    # V2 (second part): the live pass-1 tree was built on S (a dropped live-states tag would silently fall back
    # to the bucket path); otherwise this bucket is an error, which fails the gate
    live_n = per["live"]["tree"].n_states
    if live_n != S:
        raise AssertionError(
            f"bucket {bucket}: live pass-1 tree ran on n_states {live_n}, not {S} (live-states tag lost?)"
        )
    fm = paths["frozen"]["marginals"]
    box: list[Any] = []
    with _capturing_q_raw(fm, box):
        marg_f = fm.constrained_marginals(
            scaled, tables, per["frozen"]["tree"], a_cur, b, l_eff, class_mass=cm
        )
    q_raw = box[0]
    del box
    # V2 (third part): path B really is the kernel on this bucket's q_raw (device, triton, layout)
    if not kbackend.use_kernels(q_raw, kernels=True):
        raise AssertionError(
            f"bucket {bucket}: the H(q) kernel is not dispatched for this q_raw (dtype {q_raw.dtype}, shape "
            f"{tuple(q_raw.shape)}, triton {kbackend.triton_status()}); see the astra_vllm_plugin.cd.kernels log"
        )
    log_b = marg_f.log_b
    viable = log_b > fm.DEAD
    beta = torch.where(viable, torch.round(log_b), torch.zeros_like(log_b))
    w_edge = fm._gauged_edge_weights(cm, tables, beta, viable, l_eff)
    la0f, lbf = per["frozen"]["la0"], per["frozen"]["lb_end"]
    sent = torch.full_like(la0f, fm.SENTINEL)
    ga0 = torch.where(viable[0] & (la0f > fm.DEAD), la0f + beta[0], sent)
    gbl = torch.where(viable[-1] & (lbf > fm.DEAD), lbf - beta[-1], sent)
    del marg_f

    # the lm_plain / lm_comp operands: the level-0 products up_sweep hands log_matmul (strided live-block views of
    # the frozen pass-1 tree's normalised leaves; pass 2's normalised gauged leaves, [L, S, S])
    lv0 = per["frozen"]["tree"].levels[0][:, :S, :S]
    half = lv0.shape[0] // 2
    lm_plain_ops = (lv0[0 : 2 * half : 2], lv0[1 : 2 * half : 2])
    g0, _ = paths["frozen"]["semiring"].normalise_last2(
        fm._gauged_leaves(w_edge, tables, beta, viable, l_eff)
    )
    lm_comp_ops = (g0[0 : 2 * half : 2], g0[1 : 2 * half : 2])
    # V2 (third part): the log_matmul kernel is the dispatch for this bucket's real level-0 operands
    if not kbackend.use_logmatmul(*lm_plain_ops, kernels=True):
        raise AssertionError(
            f"bucket {bucket}: the log_matmul kernel is not dispatched for the level-0 product "
            f"{tuple(lm_plain_ops[0].shape)} (triton {kbackend.triton_status()}); see the "
            "astra_vllm_plugin.cd.kernels log"
        )
    bn, bk = klogmatmul.tiles_for(S, S)
    kernel = {
        "triton": kbackend.triton_status()[0],
        "block": kentropy.block_for(mb.V),
        "launches": kentropy.launches(mb.V),
        "options": kbackend.launch_options(),
        "lm_shape": [half, S, S],
        "lm_tiles": [bn, bk],
        "lm_launches": klogmatmul.launches(S, S),
        "lm_num_warps": klogmatmul.num_warps_for(bn, bk),
        # criterion 10 (reported): the first live pass-1 up_sweep of the process pays the JIT compiles
        "first_call_s": first_up_s,
        "first_in_process": not _FIRST_CALL_DONE,
    }
    _FIRST_CALL_DONE.append(True)

    # ---- N3, N4 and dead sets (gated), CUDA
    def gen() -> Any:
        gg = torch.Generator(device=dev)
        gg.manual_seed(mb.SEED)
        return gg

    out: dict[str, dict[str, Any]] = {}
    for path, mods in paths.items():
        gg = gen()
        d = mods["draw"].joint_draw(
            scaled, tables, per[path]["tree"], a_cur, b, l_eff, generator=gg, class_log_weights=cw
        )
        tail_calls: list[int] = []
        lm_calls: list[bool] = []
        sem_calls: list[int] = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with (
            (
                _counting_hq_tail(mods["marginals"], tail_calls)
                if path == "live"
                else contextlib.nullcontext()
            ),
            (
                _counting_logmatmul(klogmatmul, lm_calls)
                if path == "live"
                else contextlib.nullcontext()
            ),
            (
                _counting_semiring(live_lm_mods, sem_calls)
                if path == "live"
                else contextlib.nullcontext()
            ),
        ):
            marg = mods["marginals"].constrained_marginals(
                scaled, tables, per[path]["tree"], a_cur, b, l_eff, class_mass=cm
            )
        torch.cuda.synchronize()
        if path == "live":
            # criterion 10 (reported): the first live constrained_marginals of the process compiles the
            # specialisations the pass-1 up-sweep did not use (vec-mat / mat-vec, and every compensated one)
            kernel["first_marginals_s"] = time.perf_counter() - t0
        # V2 (fourth part): path B's constrained_marginals itself entered both kernels at their call sites
        if path == "live" and not tail_calls:
            raise AssertionError(
                f"bucket {bucket}: live constrained_marginals did not dispatch the H(q) kernel "
                "(no entry into marginals._kernel_entropy.hq_tail)"
            )
        if path == "live":
            # every product (pass 1's down-sweep asks plain, pass 2 compensated) in the compensated kernel (O9)
            _require_compensated(
                lm_calls,
                len(sem_calls),
                f"bucket {bucket}: live constrained_marginals (the N4 call)",
            )
            dispatched += lm_calls
        out[path] = {
            "draw": d,
            "gen": gg.get_state(),
            "map": mods["mapdecode"].constrained_map(
                scaled, tables, a_cur, b, l_eff, class_max=cmax
            ),
            "marg": marg,
            "maxtree": mods["mapdecode"].max_up_sweep(per[path]["leaves"]),
        }
    fo, lo = out["frozen"], out["live"]
    n3 = {
        "log_transfer": torch.equal(per["frozen"]["log_m"], per["live"]["log_m"]),
        "maxplus_tree_live_block": all(
            torch.equal(f[:, :S, :S], v[:, :S, :S])
            for f, v in zip(fo["maxtree"], lo["maxtree"], strict=True)
        ),
        "map_tokens": torch.equal(fo["map"].tokens, lo["map"].tokens),
        "map_score": torch.equal(fo["map"].score, lo["map"].score),
        "map_empty": torch.equal(fo["map"].empty, lo["map"].empty),
        "draw_tokens": torch.equal(fo["draw"].tokens, lo["draw"].tokens),
        "draw_states": torch.equal(fo["draw"].states, lo["draw"].states),
        "draw_empty": torch.equal(fo["draw"].empty, lo["draw"].empty),
        "generator_state": torch.equal(fo["gen"], lo["gen"]),
    }
    mf, ml = fo["marg"], lo["marg"]
    dead_set = {
        k: bool(torch.equal(getattr(mf, k) <= fm.DEAD, getattr(ml, k) <= fm.DEAD))
        for k in DEAD_SET_NAMES
    }

    def live_dmax(f: Any, v: Any) -> float:
        # max |Δ| over the entries live on both paths (reported)
        both = (f > fm.DEAD) & (v > fm.DEAD)
        d = (f.double() - v.double()).abs()
        return float(torch.where(both, d, torch.zeros_like(d)).max())

    n4 = {
        "dq": float((mf.q[:l_eff].double() - ml.q[:l_eff].double()).abs().max()),
        "dh": float((mf.entropy[:l_eff].double() - ml.entropy[:l_eff].double()).abs().max()),
        "dlogz": float((mf.log_z.double() - ml.log_z.double()).abs()),
        "dlog_a": live_dmax(mf.log_a, ml.log_a),
        "dlog_b": live_dmax(mf.log_b, ml.log_b),
    }
    del out, fo, lo, mf, ml

    # ---- live repeat row (ruling L6): same process, inputs and seed; REPEATS calls, each torch.equal to the first
    lmg, ldr = paths["live"]["marginals"], paths["live"]["draw"]
    first: dict[str, Any] | None = None
    repeat = dict.fromkeys(REPEAT_FIELDS, True)
    for i in range(REPEATS):
        rep_calls: list[int] = []
        rep_lm: list[bool] = []
        rep_sem: list[int] = []
        with (
            _counting_hq_tail(lmg, rep_calls),
            _counting_logmatmul(klogmatmul, rep_lm),
            _counting_semiring(live_lm_mods, rep_sem),
        ):
            mg_r = lmg.constrained_marginals(
                scaled, tables, per["live"]["tree"], a_cur, b, l_eff, class_mass=cm
            )
        if not rep_calls:  # V2: every repeat call is the kernel path, not the torch tail
            raise AssertionError(
                f"bucket {bucket}: live repeat call {i} did not dispatch the H(q) kernel"
            )
        # V2: every product of every repeat call in the compensated kernel, none on the torch path (O9)
        _require_compensated(rep_lm, len(rep_sem), f"bucket {bucket}: live repeat call {i}")
        dispatched += rep_lm
        gg = gen()
        d_r = ldr.joint_draw(
            scaled, tables, per["live"]["tree"], a_cur, b, l_eff, generator=gg, class_log_weights=cw
        )
        cur = {
            "q": mg_r.q,
            "entropy": mg_r.entropy,
            "log_a": mg_r.log_a,
            "log_b": mg_r.log_b,
            "draw_tokens": d_r.tokens,
            "draw_states": d_r.states,
            "draw_empty": d_r.empty,
            "generator_state": gg.get_state(),
        }
        del mg_r, d_r
        if first is None:
            first = cur
            continue
        for k in REPEAT_FIELDS:
            repeat[k] = repeat[k] and bool(torch.equal(first[k], cur[k]))
        del cur
    del first

    # V2: the timed lm rows' live calls (one spied call each, outside the timed region) enter the kernel once, and
    # compensated: lm_plain is a plain semiring call, which on CUDA runs the compensated kernel (O9)
    lsem = paths["live"]["semiring"]
    for row, ops, comp in (("lm_plain", lm_plain_ops, False), ("lm_comp", lm_comp_ops, True)):
        row_lm: list[bool] = []
        with _counting_logmatmul(klogmatmul, row_lm):
            lsem.log_matmul(*ops, compensated=comp)
        _require_compensated(
            row_lm, 1, f"bucket {bucket}: live {row} (semiring compensated={comp})"
        )
        dispatched += row_lm

    # ---- timing
    def tail_row(path: str) -> WithSetup:
        buf: list[Any] = []

        def setup() -> None:
            buf.clear()  # the previous call's buffer is freed before the next copy is made
            buf.append(q_raw.clone())

        ent = fentropy if path == "frozen" else kentropy
        return WithSetup(setup, lambda: ent.hq_tail(buf[0]))

    def comps(path: str) -> dict[str, Callable[[], Any] | WithSetup]:
        mods, p = paths[path], per[path]
        mg, trm = mods["marginals"], mods["tree"]

        def pass2() -> Any:
            leaves = mg._gauged_leaves(w_edge, tables, beta, viable, l_eff)
            tree2 = trm.up_sweep(leaves, compensated=True)
            return mg._down_sweep(tree2, ga0, gbl, compensated=True)

        gg = gen()
        return {
            "emission": lambda: trm.up_sweep(
                mods["emission"].log_transfer(scaled, tables, l_eff, class_mass=cm)
            ),
            "pass1_tree": lambda: trm.up_sweep(p["log_m"]),
            "marginals_pass1": lambda: mg._down_sweep(p["tree"], p["la0"], p["lb_end"]),
            "pass2": pass2,
            "marginals_full": lambda: mg.constrained_marginals(
                scaled, tables, p["tree"], a_cur, b, l_eff, class_mass=cm
            ),
            "map": lambda: mods["mapdecode"].constrained_map(
                scaled, tables, a_cur, b, l_eff, class_max=cmax
            ),
            "map_sweep": lambda: mods["mapdecode"].max_up_sweep(p["leaves"]),
            "draw": lambda: mods["draw"].joint_draw(
                scaled, tables, p["tree"], a_cur, b, l_eff, generator=gg, class_log_weights=cw
            ),
            "hq_tail": tail_row(path),
            "lm_plain": lambda: mods["semiring"].log_matmul(*lm_plain_ops),
            "lm_comp": lambda: mods["semiring"].log_matmul(*lm_comp_ops, compensated=True),
        }

    def whole(path: str, arm: str) -> Callable[[], Any]:
        mods = paths[path]
        gg = gen()
        if arm == "end_projection":

            def ep() -> Any:
                cmx = mods["mapdecode"].class_max_of(x, tables)
                mm = mods["mapdecode"].constrained_map(x, tables, a_cur, b, l_eff, class_max=cmx)
                return budget.delta_star(tables, a_cur, mm.tokens, mb.STOP_IDS)

            return ep
        ps = steps.PerStep(
            marginal=arm != "perstep_mf",
            constrained_sc=arm == "perstep_mar_sc",
            bound=mb.STOCK_BOUND,
            threshold=mb.STOCK_THRESHOLD,
            acc=torch.zeros((), dtype=torch.int64, device=dev),
            q_argmax=arm == "perstep_mar_qargmax",
        )

        def call() -> Any:
            with _steps_on(steps, mods):
                pr = steps.prepare(
                    ps, inner, st, 0, x, tables, a_cur, r_cap, l_dev, [l_eff], step0, gg
                )
            return budget.delta_star(tables, a_cur, pr.tokens, mb.STOP_IDS)

        return call

    def timing() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cf, cl = comps("frozen"), comps("live")
        for name in COMPONENTS:
            rows.append(
                {
                    "arm": "components",
                    "component": name,
                    **_ab(probe, {"frozen": cf[name], "live": cl[name]}, args),
                }
            )
        for arm in ARMS:
            fns: dict[str, Callable[[], Any] | WithSetup] = {
                "frozen": whole("frozen", arm),
                "live": whole("live", arm),
            }
            rows.append({"arm": arm, "component": "whole_call", **_ab(probe, fns, args)})
        return rows

    facts: dict[str, Any] = {
        "bucket": bucket,
        "automaton": manifest,
        "n3": n3,
        "n4": n4,
        "dead_set": dead_set,
        "dispatch": {
            "log_matmul": len(dispatched),
            "log_matmul_plain": len(dispatched) - sum(dispatched),
            "log_matmul_comp": sum(dispatched),
        },
        "repeat": repeat,
        "kernel": kernel,
    }
    rows = timing()
    if needs_rerun(rows):  # ruling M4 (= L8): one re-run of this bucket's A/B, same process
        facts["rows_first"] = rows
        rows = timing()
    facts["rows"] = rows
    return facts


def inner_main(args: argparse.Namespace) -> int:
    import torch

    probe = mb._load_probe()
    torch.set_float32_matmul_precision("high")
    payload_dir = Path(args.inner)
    status = 0
    for bucket in mb.BUCKETS:
        try:
            facts = measure_bucket(probe, bucket, payload_dir, args)
        except Exception:
            facts = {"bucket": bucket, "error": traceback.format_exc(), "rows": []}
            status = 1
        facts["regime"] = probe.regime()
        torch.cuda.empty_cache()
        print(MARKER + json.dumps(facts), flush=True)
    return status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--inner", default=None, help="container side: the payload directory")
    ap.add_argument(
        "--n", type=int, default=N, help="calls per path per component, over all rounds"
    )
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--row-budget-s", type=float, default=ROW_BUDGET_S)
    ap.add_argument("--timeout-s", type=float, default=3 * 3600.0)
    args = ap.parse_args(argv)
    if args.inner is not None:
        return inner_main(args)
    return host(args)


if __name__ == "__main__":
    sys.exit(main())
