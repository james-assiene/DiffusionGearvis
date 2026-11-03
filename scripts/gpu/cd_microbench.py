"""T-0071a a1 (criterion 2): the per-step CD microbench, in the pinned container, no engine. `/launch` ONLY.

    uv run --group cd python scripts/gpu/cd_microbench.py            # host side (the /launch command)
    python3 /repo/scripts/gpu/cd_microbench.py --inner /cdpayloads  # inside the container (the host runs it)

GPU-spending by definition (scripts/gpu/, T-0016). The host side never boots an engine; it runs one
``docker run --rm --gpus all --network none`` of the image ``services/vllm.sh`` pins, the way T-0072's perf
probe was launched (``tests/contract/test_vllm_cd_perf_gpu.py``): the repository read-only at ``/repo``,
``PYTHONPATH=/repo:/repo/plugin/astra_vllm_plugin``, nothing from the host venv (ruling S7), the image digest
and the host commit handed in as ``ASTRA_CONTAINER_DIGEST`` / ``ASTRA_COMMIT``.

Host side, in order: refuse a dirty tree (naming the paths), an unreadable image digest or an evidence
directory that already exists (exit 2); compile the four real T-0063 dev-cut automata of ``AUTOMATA`` (one per
bucket, JSON whitespace; the bucket is checked after compiling) and write their wire payloads
(``local_astra.decoding.artifact.encode_automaton``) to a temporary directory mounted read-only at
``/cdpayloads``; run the container; parse its ``ASTRA_MICROBENCH_JSON`` lines; write
``logs/t0071a-microbench-<sha>/report.txt`` (``format_report``) and ``lines.jsonl`` before the exit code
(0: every bucket measured; 1: a bucket errored or the container failed; 2: refused).

Container side (``--inner``), per bucket 64/128/256/512, the plugin's own decode path
(``automaton.decode_payload`` → ``requests._tables_for``) on CUDA, fp32, TF32 "high" set globally, CL 256,
V 262 144, seed 0, random logits (×4, the probe's), one L_eff candidate (= CL), stock knobs 0.1/0.005, an
embedding of the model's shape (bf16 ``[262144, 2816]``, random) for the SC matmul. Timing is the T-0072
probe's ``_timed`` (CUDA events, median and p95 over n calls after the warm-up, then ``peak_mem_delta_bytes``
of one call: ``max_memory_allocated`` after a reset minus ``memory_allocated`` before it). n = 200 after 20
warm-up unless one row would exceed ``--row-budget-s``; then n (and the warm-up) shrink and the row says so.

Rows (``arm``, ``bucket``, ``component``):
- ``whole_call`` per arm: ``end_projection`` = ``class_max_of`` + ``constrained_map`` + ``delta_star``; the
  per-step arms = ``steps.prepare`` (everything CD adds before the stock call: H(p), class mass, emission and
  tree, draw, MAP, marginals for Mar, the SC matmul) + ``delta_star``. Its ``peak_added_bytes`` is the arm's
  peak added memory (criterion 2; SC at 512 is flagged against T-0069's free-after-boot − 1 GiB).
- components (arm ``components``): ``class_mass``, ``class_max``, ``emission`` (``log_transfer`` +
  ``up_sweep``), ``draw`` (``joint_draw``), ``map`` (``constrained_map``), ``delta_star``, ``marginals_full``
  (both passes + q + H(q), decision 10), ``marginals_pass1`` (the ungauged down-sweep, ``_down_sweep``),
  ``hq`` (the H(q) ops of ``constrained_marginals`` replayed on its q), ``marginals_pass2_derived``
  (full − pass 1 − H(q); derived, not timed), ``sc_matmul`` (``steps._soft`` on q).
- the T-0072 probe's own breakdown (arm ``t0072_probe``; synthetic NFA per bucket, its ``measure``): class_max,
  MAP sweep, δ* scan and sequential, stop-id build, constrained_map.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
MARKER = "ASTRA_MICROBENCH_JSON"
BUCKETS = (64, 128, 256, 512)
#: one real T-0063 dev-cut automaton per bucket (JSON whitespace): the first dev record of that bucket in cut
#: order (CPU check 2026-10-03: Sudoku dev is all bucket 64; BFCL dev has 35/32/32/1 at 64/128/256/512).
#: The bucket is re-checked after compiling; a mismatch refuses the run.
AUTOMATA: Mapping[int, tuple[str, str]] = {
    64: ("sudoku", "dev:sudoku4_0"),
    128: ("bfcl", "live_simple_133-86-0"),
    256: ("bfcl", "live_simple_130-84-0"),
    512: ("bfcl", "live_simple_165-98-0"),
}
ARMS = ("end_projection", "perstep_mf", "perstep_mar", "perstep_mar_sc")
CL = 256
V = 262_144
HIDDEN = 2816  # text_config.hidden_size of nvidia/diffusiongemma-26B-A4B-it-NVFP4 rev 2ea8372
N = 200
WARMUP = 20
SEED = 0
STOCK_BOUND, STOCK_THRESHOLD = 0.1, 0.005
T_MIN, T_MAX, MAX_STEPS = 0.4, 0.8, 48
STOP_IDS = (1, 50, 106)
ROW_BUDGET_S = 60.0
#: T-0071 implementation notes: T-0069 had 4.7 GiB free after boot; flag SC's peak at 512 above that − 1 GiB
FREE_AFTER_BOOT_GIB = 4.7
PROBE = "/repo/tests/container_scripts/astra_cd_perf_probe.py"
PAYLOADS = "/cdpayloads"

#: R5: what a ``whole_call`` row measures: CD's own pre-work only, never the stock sampler call
WHOLE_CALL_LABEL = {
    "end_projection": "class_max_of + constrained_map + delta_star only (no stock call)",
    "perstep_mf": "steps.prepare + delta_star (δ*) only (no stock call)",
    "perstep_mar": "steps.prepare + delta_star (δ*) only (no stock call)",
    "perstep_mar_sc": "steps.prepare + delta_star (δ*) only (no stock call)",
}
_DIGEST = re.compile(r"(?:[a-z0-9][a-z0-9._/:-]*@)?sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


# --------------------------------------------------------------------------- the report (pure)


def format_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    container_digest: str | None,
    commit: str | None,
    regime: str | None,
) -> str:
    """The microbench report. Refuses (``ValueError``) without a content digest of the image
    (``[repo@]sha256:<64 lowercase hex>``), a full 40-hex commit, or a one-line regime (ruling T5)."""
    if not isinstance(container_digest, str) or not _DIGEST.fullmatch(container_digest):
        raise ValueError(
            f"container digest {container_digest!r} is not [repo@]sha256:<64 lowercase hex>: the "
            "report would not identify the image"
        )
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError(f"commit {commit!r} is not a full 40-hex commit id")
    if not isinstance(regime, str) or not regime.strip():
        raise ValueError("no regime line: no number without its regime")
    if "\n" in regime.strip():
        raise ValueError("the regime must be one line")
    lines = [
        "T-0071a a1 microbench (criterion 2): reported, not gated",
        f"container digest: {container_digest}",
        f"commit: {commit}",
        regime.strip(),
        "",
        f"{'arm':<16} {'bucket':>6}  {'component':<26} {'median ms':>10} {'p95 ms':>10} "
        f"{'n':>4} {'warm':>4} {'peak added MiB':>15}  note",
    ]

    def num(x: Any, fmt: str) -> str:
        return format(float(x), fmt) if isinstance(x, (int, float)) else "-"

    for r in rows:
        peak = r.get("peak_added_bytes")
        peak_s = num(peak / 2**20, ".1f") if isinstance(peak, (int, float)) else "-"
        if r.get("error"):
            lines.append(
                f"{r.get('arm', '-')!s:<16} {r.get('bucket', '-')!s:>6}  {r.get('component', '-')!s:<26} "
                f"ERROR {str(r['error']).strip().splitlines()[-1][:120]}"
            )
            continue
        labels = [str(r.get("note") or "")]
        if (
            r.get("component") == "whole_call"
        ):  # R5: not the whole sampler call; the stock call is outside
            labels.insert(0, WHOLE_CALL_LABEL.get(str(r.get("arm")), "CD pre-work only"))
        if r.get("component") == "marginals_pass2_derived":
            med = r.get("median_ms")
            if isinstance(med, (int, float)) and med < 0:
                labels.insert(0, "NEGATIVE: derived value is subtraction noise, not a cost")
        lines.append(
            f"{r['arm']!s:<16} {r['bucket']!s:>6}  {r['component']!s:<26} "
            f"{num(r.get('median_ms'), '.3f'):>10} {num(r.get('p95_ms'), '.3f'):>10} "
            f"{r.get('n', '-')!s:>4} {r.get('warmup', '-')!s:>4} {peak_s:>15}  "
            + "; ".join(x for x in labels if x)
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- host side


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def die(code: int, msg: str) -> None:
    print(f"cd_microbench: {msg}", file=sys.stderr)
    raise SystemExit(code)


def pinned_image() -> str:
    """The tag ``services/vllm.sh`` defaults to (as ``tests/vllm_container.pinned_image`` reads it)."""
    text = (REPO / "services" / "vllm.sh").read_text()
    m = re.search(r"^CONTAINER=\$\{ASTRA_VLLM_CONTAINER:-([^}]+)\}", text, re.MULTILINE)
    if m is None:
        die(2, "services/vllm.sh no longer defines CONTAINER=${ASTRA_VLLM_CONTAINER:-<image>}")
        raise AssertionError  # unreachable
    return m.group(1).strip()


def _docker_direct() -> bool:
    """R4: probe ``docker info`` alone; True when the daemon answers without ``sg docker``."""
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _wrap(argv: list[str], direct: bool) -> list[str]:
    return argv if direct else ["sg", "docker", "-c", shlex.join(argv)]


def _text(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode(errors="replace")
    return x or ""


def docker(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """A short docker command (``image inspect``), run once, directly or through ``sg docker -c``."""
    return subprocess.run(
        _wrap(argv, _docker_direct()), capture_output=True, text=True, timeout=timeout
    )


def run_container(argv: list[str], *, name: str, timeout_s: float) -> tuple[int, str, str]:
    """R4: the ``docker run ...`` job, exactly once, as ``--name <name>``. On timeout the container is killed
    by name, the partial stdout kept and ``rc == 1`` returned; a failed job is returned as is, never re-run."""
    direct = _docker_direct()
    job = list(argv)
    if job[:2] != ["docker", "run"]:
        raise ValueError(f"not a docker run command: {job[:3]}")
    job[2:2] = ["--name", name]
    try:
        proc = subprocess.run(_wrap(job, direct), capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        kill = subprocess.run(
            _wrap(["docker", "kill", name], direct), capture_output=True, text=True, timeout=60
        )
        return (
            1,
            _text(exc.output),
            f"timed out after {timeout_s:.0f} s; docker kill {name} -> rc {kill.returncode} "
            f"{_text(kill.stderr).strip()[:200]}",
        )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def image_digest(image: str) -> str:
    fmt = "{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}"
    proc = docker(["docker", "image", "inspect", "--format", fmt, image], timeout=60)
    out = proc.stdout.strip()
    if proc.returncode != 0 or not _DIGEST.fullmatch(out):
        die(2, f"refusing: no content digest for {image}: {out or proc.stderr.strip()[:200]}")
    return out


def write_payloads(dest: Path) -> dict[str, Any]:
    """Compile ``AUTOMATA`` on the host (the image has no compiler) and write their wire payloads."""
    import hashlib

    from local_astra.decoding.artifact import encode_automaton
    from local_astra.eval.cd import cuts, replay

    manifest: dict[str, Any] = {}
    for bucket, (task, rid) in AUTOMATA.items():
        case = next((c for c in cuts.load(task, "dev") if str(c.id) == rid), None)
        if case is None:
            die(2, f"refusing: {task} dev record {rid} not found")
        automaton = replay.compile_automaton(task, case, whitespace="json")
        if int(automaton.n_states_bucket) != bucket:
            die(2, f"refusing: {rid} compiled to bucket {automaton.n_states_bucket}, not {bucket}")
        payload = encode_automaton(automaton)
        (dest / f"b{bucket}.b64").write_text(payload)
        manifest[str(bucket)] = {
            "task": task,
            "id": rid,
            "cut": "dev",
            "whitespace": "json",
            "n_states": int(automaton.n_states),
            "bucket": bucket,
            "payload_sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
        }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def host(args: argparse.Namespace) -> int:
    dirty = _git("status", "--porcelain")
    if dirty:
        die(2, f"refusing: the tree is dirty:\n{dirty}")
    commit = _git("rev-parse", "HEAD")
    out = REPO / "logs" / f"t0071a-microbench-{commit[:7]}"
    if out.exists():
        die(2, f"refusing: {out} exists; move it aside first")
    image = pinned_image()
    digest = image_digest(image)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    with tempfile.TemporaryDirectory(prefix="t0071a-cdpayloads-") as tmp:
        manifest = write_payloads(Path(tmp))
        argv = ["docker", "run", "--rm", "--gpus", "all", "--network", "none"]
        argv += ["-v", f"{REPO}:/repo:ro", "-v", f"{tmp}:{PAYLOADS}:ro", "-w", "/repo"]
        for k, v in {
            "PYTHONPATH": "/repo:/repo/plugin/astra_vllm_plugin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "ASTRA_CONTAINER_DIGEST": digest,
            "ASTRA_COMMIT": commit,
        }.items():
            argv += ["-e", f"{k}={v}"]
        argv += ["--entrypoint", "python3", image, "/repo/scripts/gpu/cd_microbench.py", "--inner"]
        argv += [PAYLOADS, "--n", str(args.n), "--warmup", str(args.warmup)]
        argv += ["--row-budget-s", str(args.row_budget_s)]
        print("running: " + shlex.join(argv), flush=True)
        t0 = time.monotonic()
        rc, stdout, stderr = run_container(
            argv, name=f"t0071a-microbench-{commit[:7]}", timeout_s=args.timeout_s
        )
        elapsed = time.monotonic() - t0

    out.mkdir(parents=True)
    facts = [json.loads(ln[len(MARKER) :]) for ln in stdout.splitlines() if ln.startswith(MARKER)]
    (out / "lines.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts))
    (out / "container.log").write_text(stdout + "\n--- stderr ---\n" + stderr)
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    inner_regime: dict[str, Any] = {}
    seen: set[int] = set()
    for f in facts:
        bucket = int(f["bucket"])
        seen.add(bucket)
        inner_regime = f.get("regime") or inner_regime
        if f.get("error"):
            problems.append(f"bucket {bucket}: {str(f['error']).strip().splitlines()[-1]}")
            rows.append({"arm": "-", "bucket": bucket, "component": "-", "error": f["error"]})
        rows.extend(f.get("rows") or [])
        if f.get("probe_error"):
            problems.append(f"bucket {bucket} T-0072 probe: {f['probe_error'].strip()[-200:]}")
        for k in ("container_digest", "commit"):
            if (f.get("regime") or {}).get(k) not in (
                None,
                digest if k == "container_digest" else commit,
            ):
                problems.append(f"bucket {bucket}: inner {k} {f['regime'][k]} != host's")
    problems += [f"bucket {b}: no line from the container" for b in BUCKETS if b not in seen]
    if rc != 0:
        problems.append(f"container exited {rc}: {stderr.strip()[-400:]}")

    automata = "; ".join(
        f"b{b} {m['task']} {m['id']} (|S| {m['n_states']})"
        for b, m in sorted(manifest.items(), key=lambda kv: int(kv[0]))
    )
    regime = (
        f"regime: container {image} (no engine), torch {inner_regime.get('torch')}, CUDA "
        f"{inner_regime.get('cuda')}, vllm {inner_regime.get('vllm')}, {inner_regime.get('device')}, "
        f"driver {driver}; CL {CL}, V {V}, fp32, TF32 {inner_regime.get('matmul_precision')}, n {args.n} "
        f"after {args.warmup} warm-up unless a row says otherwise (row budget {args.row_budget_s:.0f} s), "
        f"CUDA events, seed {SEED}, buckets {'/'.join(map(str, BUCKETS))}, automata (dev cuts, JSON "
        f"whitespace): {automata}; random logits x4, one L_eff candidate (= CL), stock knobs "
        f"{STOCK_BOUND}/{STOCK_THRESHOLD}, embedding bf16 [{V}, {HIDDEN}] random; host clean at "
        f"{commit}; wall {elapsed:.0f} s"
    )
    sc512 = next(
        (
            r.get("peak_added_bytes")
            for r in rows
            if r.get("arm") == "perstep_mar_sc"
            and r.get("bucket") == 512
            and r.get("component") == "whole_call"
        ),
        None,
    )
    limit = (FREE_AFTER_BOOT_GIB - 1.0) * 2**30
    if isinstance(sc512, (int, float)):
        mem_line = (
            f"memory: perstep_mar_sc peak added at bucket 512 = {sc512 / 2**30:.2f} GiB vs "
            f"free-after-boot {FREE_AFTER_BOOT_GIB} GiB − 1 GiB = {limit / 2**30:.2f} GiB (T-0069): "
            + ("FLAG for H1 (exceeds)" if sc512 > limit else "within")
        )
    else:
        mem_line = "memory: perstep_mar_sc peak at bucket 512 not measured"
        problems.append(mem_line)
    report = format_report(rows, container_digest=digest, commit=commit, regime=regime)
    report += "\n" + mem_line + "\n"
    report += "".join(f"PROBLEM: {p}\n" for p in problems)
    report += f"verdict: {'COMPLETE' if not problems else 'INCOMPLETE'} (reported, not gated)\n"
    (out / "report.txt").write_text(report)
    print(report)
    print(f"evidence: {out}")
    return 0 if not problems else 1


# --------------------------------------------------------------------------- container side


def _load_probe() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("_t0072_perf_probe", PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _timed_capped(
    probe: Any, fn: Any, n: int, warmup: int, budget_s: float
) -> tuple[dict[str, Any], str]:
    """The probe's ``_timed`` at (n, warmup), shrunk so that one row stays within ``budget_s``."""
    import torch

    def sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    per_call = max(time.perf_counter() - t0, 1e-6)  # the first call: cold, an over-estimate
    note = ""
    if per_call * (n + warmup) > budget_s:
        n_eff = max(20, int(0.8 * budget_s / per_call))
        warm_eff = max(2, min(warmup, n_eff // 10))
        if (n_eff, warm_eff) != (n, warmup):
            note = f"n reduced to {n_eff} after {warm_eff} warm-up (first call {per_call * 1e3:.0f} ms)"
            n, warmup = n_eff, warm_eff
    out = dict(probe._timed(fn, n, warmup))
    out.update(n=n, warmup=warmup)
    return out, note


def measure_bucket(
    probe: Any, bucket: int, payload_dir: Path, args: argparse.Namespace, device: str = "cuda"
) -> dict[str, Any]:
    import types

    import numpy as np
    import torch
    from astra_vllm_plugin.cd import automaton as cd_automaton
    from astra_vllm_plugin.cd import budget, draw, emission, mapdecode, marginals, steps, tree
    from astra_vllm_plugin.cd import requests as cd_requests
    from astra_vllm_plugin.cd.semiring import compensated_sum

    manifest = json.loads((payload_dir / "manifest.json").read_text())[str(bucket)]
    payload = (payload_dir / f"b{bucket}.b64").read_text()
    wa = cd_automaton.decode_payload(payload)
    if int(wa.n_states_bucket) != bucket:
        raise AssertionError(f"payload bucket {wa.n_states_bucket} != {bucket}")
    dev = torch.device(device)
    tables = cd_requests._tables_for(manifest["payload_sha256"], wa, dev)
    sv = np.zeros(bucket, dtype=bool)
    sv[: len(wa.start_vector)] = np.asarray(wa.start_vector, dtype=bool)[:bucket]
    a_cur = torch.from_numpy(sv).to(dev)
    budget.prepare_stop_tensor(STOP_IDS, tables.distance_to_final.device)

    g = torch.Generator(device="cpu").manual_seed(SEED)
    x = (torch.randn(CL, V, generator=g) * 4.0).to(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(SEED)
    emb = (torch.randn(V, HIDDEN, generator=g) * 0.02).to(torch.bfloat16).to(dev)
    inner = types.SimpleNamespace(
        t_min=T_MIN,
        t_max=T_MAX,
        embed_weight=emb,
        sc_vocab_start=0,
        sc_vocab_end=V,
        tp_size=1,
        normalizer=math.sqrt(HIDDEN),
        entropy_bound=STOCK_BOUND,
        confidence_threshold=STOCK_THRESHOLD,
    )
    st = types.SimpleNamespace(
        canvas_length=CL,
        max_denoising_steps=MAX_STEPS,
        accepted_canvas_history_len=torch.zeros(4, dtype=torch.int64, device=dev),
    )
    step0 = torch.zeros((), dtype=torch.int64, device=dev)
    r_cap = torch.full((), CL, dtype=torch.int64, device=dev)
    l_dev = torch.full((), CL, dtype=torch.int64, device=dev)
    l_eff = CL
    scaled = x / steps._temperature(inner, step0, float(MAX_STEPS))
    b = budget.terminal_factor(tables, r_cap - l_eff)
    cm = emission.class_mass(scaled, tables)
    cw = emission.class_log_weights_from(cm)
    cmax = mapdecode.class_max_of(scaled, tables)
    tr = tree.up_sweep(emission.log_transfer(scaled, tables, l_eff, class_mass=cm))
    m = mapdecode.constrained_map(scaled, tables, a_cur, b, l_eff, class_max=cmax)
    marg = marginals.constrained_marginals(scaled, tables, tr, a_cur, b, l_eff, class_mass=cm)
    root = tr.levels[-1][0]
    la0 = marginals._mask_to_log(a_cur, root)
    lb_end = marginals._mask_to_log(b, root)
    q = marg.q

    def hq() -> Any:  # the H(q) ops of constrained_marginals, on its q
        total = compensated_sum(q).unsqueeze(-1).clamp_min(torch.finfo(q.dtype).tiny)
        log_total = torch.log(total)
        term = q.clamp_min(1e-38)
        term.log_()
        torch.maximum(term, log_total + math.log(1e-30), out=term)
        term.mul_(q)
        return log_total.squeeze(-1) - compensated_sum(term) / total.squeeze(-1)

    def perstep(arm: str) -> Any:
        ps = steps.PerStep(
            marginal=arm != "perstep_mf",
            constrained_sc=arm == "perstep_mar_sc",
            bound=STOCK_BOUND,
            threshold=STOCK_THRESHOLD,
            acc=torch.zeros((), dtype=torch.int64, device=dev),
        )

        def call() -> Any:
            pr = steps.prepare(
                ps, inner, st, 0, x, tables, a_cur, r_cap, l_dev, [l_eff], step0, gen
            )
            return budget.delta_star(tables, a_cur, pr.tokens, STOP_IDS)

        return call

    def end_projection() -> Any:
        cmx = mapdecode.class_max_of(x, tables)
        mm = mapdecode.constrained_map(x, tables, a_cur, b, l_eff, class_max=cmx)
        return budget.delta_star(tables, a_cur, mm.tokens, STOP_IDS)

    whole = {"end_projection": end_projection, **{a: perstep(a) for a in ARMS[1:]}}
    components = {
        "class_mass": lambda: emission.class_mass(scaled, tables),
        "class_max": lambda: mapdecode.class_max_of(scaled, tables),
        "emission": lambda: tree.up_sweep(
            emission.log_transfer(scaled, tables, l_eff, class_mass=cm)
        ),
        "draw": lambda: draw.joint_draw(
            scaled, tables, tr, a_cur, b, l_eff, generator=gen, class_log_weights=cw
        ),
        "map": lambda: mapdecode.constrained_map(scaled, tables, a_cur, b, l_eff, class_max=cmax),
        "delta_star": lambda: budget.delta_star(tables, a_cur, m.tokens, STOP_IDS),
        "marginals_full": lambda: marginals.constrained_marginals(
            scaled, tables, tr, a_cur, b, l_eff, class_mass=cm
        ),
        "marginals_pass1": lambda: marginals._down_sweep(tr, la0, lb_end),
        "hq": hq,
        "sc_matmul": lambda: steps._soft(q, inner),
    }
    rows: list[dict[str, Any]] = []
    src = f"{manifest['task']} {manifest['id']} |S| {manifest['n_states']}"
    for arm, fn in whole.items():
        t, note = _timed_capped(probe, fn, args.n, args.warmup, args.row_budget_s)
        rows.append(
            {
                "arm": arm,
                "bucket": bucket,
                "component": "whole_call",
                "median_ms": t["median_ms"],
                "p95_ms": t["p95_ms"],
                "n": t["n"],
                "warmup": t["warmup"],
                "peak_added_bytes": t["peak_mem_delta_bytes"],
                "note": "; ".join(x for x in (src, note) if x),
            }
        )
    timed: dict[str, dict[str, Any]] = {}
    for name, fn in components.items():
        t, note = _timed_capped(probe, fn, args.n, args.warmup, args.row_budget_s)
        timed[name] = t
        rows.append(
            {
                "arm": "components",
                "bucket": bucket,
                "component": name,
                "median_ms": t["median_ms"],
                "p95_ms": t["p95_ms"],
                "n": t["n"],
                "warmup": t["warmup"],
                "peak_added_bytes": t["peak_mem_delta_bytes"],
                "note": note,
            }
        )
    full, p1, h = timed["marginals_full"], timed["marginals_pass1"], timed["hq"]
    rows.append(
        {
            "arm": "components",
            "bucket": bucket,
            "component": "marginals_pass2_derived",
            "median_ms": full["median_ms"] - p1["median_ms"] - h["median_ms"],
            "p95_ms": None,
            "note": "derived: marginals_full - marginals_pass1 - hq medians (pass 2 + q), not timed",
        }
    )
    return {"bucket": bucket, "automaton": manifest, "rows": rows}


def inner_main(args: argparse.Namespace) -> int:
    import torch

    probe = _load_probe()
    torch.set_float32_matmul_precision("high")
    payload_dir = Path(args.inner)
    status = 0
    for bucket in BUCKETS:
        try:
            facts = measure_bucket(probe, bucket, payload_dir, args)
        except Exception:
            facts = {"bucket": bucket, "error": traceback.format_exc(), "rows": []}
            status = 1
        try:
            p = probe.measure(bucket, CL=CL, n=args.n, warmup=args.warmup, seed=SEED)
            for name, comp in p["components"].items():
                facts["rows"].append(
                    {
                        "arm": "t0072_probe",
                        "bucket": bucket,
                        "component": name,
                        "median_ms": comp["median_ms"],
                        "p95_ms": comp["p95_ms"],
                        "n": args.n,
                        "warmup": args.warmup,
                        "peak_added_bytes": comp.get("peak_mem_delta_bytes"),
                        "note": "T-0072 probe, synthetic NFA",
                    }
                )
        except Exception:
            facts["probe_error"] = traceback.format_exc()
            status = 1
        facts["regime"] = probe.regime()
        torch.cuda.empty_cache()
        print(MARKER + json.dumps(facts), flush=True)
    return status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--inner", default=None, help="container side: the payload directory")
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--row-budget-s", type=float, default=ROW_BUDGET_S)
    ap.add_argument("--timeout-s", type=float, default=3 * 3600.0)
    args = ap.parse_args(argv)
    if args.inner is not None:
        return inner_main(args)
    return host(args)


if __name__ == "__main__":
    raise SystemExit(main())
