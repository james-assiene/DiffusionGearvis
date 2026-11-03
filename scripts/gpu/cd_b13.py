"""T-0071a a2 -- B13 on the engine (criteria 3-4) and the runner's first engine run (criterion 5). `/launch` ONLY.

    uv run --group cd python scripts/gpu/cd_b13.py [--boot-log logs/vllm.log] [--profile lean]

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never
boots one; ``scripts/locked/t0071a_b13.sh`` does (engine only, ``services/vllm.sh``, the plugin wheel built
from this commit).

Refusals (exit 2, before any measured request): a dirty tree (paths named); ``services/vllm.sh`` changed since
806da12; no engine log or no boot in it; the evidence directory exists; no engine on 127.0.0.1:8001; an engine
not on the diffusion path (a ``temperature`` probe must be a 400); a gated prompt whose automaton is not of its
bucket or not sendable; the tap subscription not live after the warm-up (T-0069 O7).

Order, concurrency 1 throughout:

1. Subscription warm-up (T-0069 O7): one small unconstrained ``warmup:sub`` request, up to 3 attempts within
   30 s, until one of its tap frames is received; none → exit 2. Its frames are in no row.
2. Gated strata (criterion 3), bucket 64 (Sudoku dev ``dev:sudoku4_0``) then bucket 128 (BFCL dev
   ``live_simple_133-86-0``): 2 warm-up rounds (discarded), then 30 rounds. Round ``r`` runs
   ``b13_steps.rotation`` (ABA′, BA′A, A′AB, ...) with D inserted at slot ``r % 4`` (ruling T2). A and A′ send
   ``vllm_xargs={"astra_tap": true}``; B sends the same plus the ``perstep_mar`` ``astra_cd_*`` keys (stock
   knobs 0.1/0.005 sent explicitly); D sends no xargs (T-0011 R17).
3. Report-only strata (criterion 4), buckets 256 and 512 (BFCL dev ``live_simple_130-84-0``,
   ``live_simple_165-98-0``): 1 warm-up round, then 5 rounds of A, B (``perstep_mar``), A′ and E
   (``end_projection``) at slot ``r % 4``.
4. The B13 report (``report.txt``, ``requests.jsonl``, ``frames.jsonl``, ``engine.log`` = the last boot of the
   engine log, R16) is written **before** the runner runs.
5. Criterion 5: ``local_astra.eval.cd.runner.run`` (the entry point ``scripts/gpu/cd_eval.py run`` calls; the
   CLI cannot pass ``mar_knobs``) with ``limit=10`` on the dev cut, every arm x both tasks, Mar arms with the
   stock knobs through ``mar_knobs`` (the runner labels such a run ``calibration``; here it is the stock-knob
   dry run). Artifacts under the evidence directory. Pass = exit 0 and zero ``join_error`` rows each.

Step time = (wall − F̂)/Σsteps (``local_astra.bench.b13_steps``). Σsteps of a request = Σ of its tap commit
frames' ``step`` (joined by response id, ``smoke.belongs``); every request must have exactly
``ceil(completion_tokens / 256)`` commit frames and a tap release line with ``commit_overrun == 0``. For B (and
E) the commit frames are cross-checked against the CD records joined on ``record_tag``: per canvas, the
committed ids (``smoke.commit_frame_problems``) and ruling T3 -- the frame's ``step`` equals the record's
``steps`` exactly (both are the slot's step at entry of the commit call: the tap's ``pre_step`` clone and CD's
``_Job.step0`` clone, taken before stock resets it). Every decision is ``local_astra.bench.b13_run``'s (ruling
R3): ``check_rows`` on the saved evidence, then ``evaluate_b13``: per gated stratum F̂ and SE(F̂) from its own A
and A′ (R2), E10 at F̂, at 0 and at F̂ + 2·SE, PASS only if all three pass, FAIL if F̂ and 0 both fail (R1); the
pooled F̂ is a sensitivity line only. The xargs exposure (D vs A per step, against the A/A′ control) is printed
before the gate. A round whose A, B or A′ request has any problem is dropped and named; a D with a problem drops
only that round's D. A crash after some strata still evaluates the strata that completed (R4).

Exit code (report first): ``b13_run.exit_code`` -- 0 = no run problem (criterion 5 passed too), whatever B13's
verdict (D-H1: a FAIL is recorded, not hidden); 1 = a run defect, listed under PROBLEMS; 2 = refused. The
verdicts are in ``verdict.json`` and the report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import httpx
import msgpack

from local_astra.bench import b13_run
from local_astra.bench.b16_tap import last_boot
from local_astra.bench.thresholds import THRESHOLDS
from local_astra.contracts.cd import MAX_AUTOMATON_CHARS
from local_astra.eval.cd import arms, cuts, prompts, records, replay, runner, smoke

REPO = Path(__file__).resolve().parents[2]
API = "http://127.0.0.1:8001"
V1 = API + "/v1"
TAP = "tcp://127.0.0.1:8002"
TOPIC = b"astra.canvas.step"
REFERENCE_COMMIT = "806da12"
REFERENCE = REPO / "logs" / "t0011-tier3-806da12" / "boot-reference.txt"
CANVAS = 256
#: stratum bucket -> (task, dev record id); the first dev record of that bucket in cut order (CPU check
#: 2026-10-03, JSON whitespace). Re-checked after compiling.
GATED: dict[int, tuple[str, str]] = {
    64: ("sudoku", "dev:sudoku4_0"),
    128: ("bfcl", "live_simple_133-86-0"),
}
REPORT_ONLY: dict[int, tuple[str, str]] = {
    256: ("bfcl", "live_simple_130-84-0"),
    512: ("bfcl", "live_simple_165-98-0"),
}
METRIC = {64: "cd_step_overhead_pct_b64", 128: "cd_step_overhead_pct_b128"}
ROUNDS = 30
REPORT_ROUNDS = 5
WARMUP_ROUNDS = 2
REPORT_WARMUP_ROUNDS = 1
#: stock DiffusionGemma knobs (generation_config.json; docs/02 §5): entropy bound 0.1, stop 0.005
STOCK_KNOBS = {"entropy_bound": 0.1, "confidence_threshold": 0.005}
TAP_XARGS = {"astra_tap": True}
B_ARM = "perstep_mar"
E_ARM = "end_projection"
FRAME_WAIT_S = 5.0
GAP_S = 0.5
RECORD_WAIT_S = 10.0
FLUSH_WAIT_S = 5.0
MAX_FLUSHES = 2
FLUSH_PROMPT = "Reply with OK."
WARMUP_ATTEMPTS = 3
WARMUP_BUDGET_S = 30.0
WARMUP_MAX_TOKENS = 8
RUNNER_TASKS = ("sudoku", "bfcl")
RUNNER_LIMIT = 10
#: R4: explicit per-run wall budget and per-request timeout for criterion 5's runner.run calls
RUNNER_BUDGET_S = 1800.0
RUNNER_REQUEST_TIMEOUT_S = 600.0


def die(code: int, msg: str) -> None:
    print(f"cd_b13: {msg}", file=sys.stderr)
    raise SystemExit(code)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def gpu_memory() -> str:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,memory.free,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() or f"nvidia-smi failed: {out.stderr.strip()}"


def thresholds() -> dict[int, float]:
    by = {t.metric: t.limit for t in THRESHOLDS if t.bench == "B13"}
    return {b: float(by[m]) for b, m in METRIC.items()}


# --------------------------------------------------------------------------- tap subscriber


class Subscriber:
    """Every ``astra.canvas.step`` frame of the run, decoded, in arrival order."""

    def __init__(self) -> None:
        import zmq

        self.frames: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sock = zmq.Context.instance().socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, TOPIC)
        self._sock.connect(TAP)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._sock.poll(timeout=50):
                _topic, payload = self._sock.recv_multipart()
                frame = msgpack.unpackb(payload, raw=False)
                with self._lock:
                    self.frames.append(frame)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.frames)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2.0)
        self._sock.close(linger=0)


class SubscriberNotReady(Exception):
    """T-0069 O7: no frame of the warm-up request reached the subscriber; the run is refused (exit 2)."""


# --------------------------------------------------------------------------- prompts and automata


class Stratum:
    def __init__(self, bucket: int, task: str, rid: str, gated: bool) -> None:
        from local_astra.decoding.artifact import encode_automaton

        self.bucket, self.task, self.rid, self.gated = bucket, task, rid, gated
        case = next((c for c in cuts.load(task, "dev") if str(c.id) == rid), None)
        if case is None:
            raise ValueError(f"{task} dev record {rid} not found")
        self.case = case
        self.prompt = prompts.build_prompt(task, case)
        self.automaton = replay.compile_automaton(task, case, whitespace="json")
        got = int(self.automaton.n_states_bucket)
        if got != bucket:
            raise ValueError(f"{rid} compiled to bucket {got}, not {bucket}")
        self.n_states = int(self.automaton.n_states)
        self.payload = encode_automaton(self.automaton)
        if len(self.payload) > MAX_AUTOMATON_CHARS:
            raise ValueError(f"{rid}: payload {len(self.payload)} chars > {MAX_AUTOMATON_CHARS}")
        self.extra = None if gated else "E"
        self.rounds = ROUNDS if gated else REPORT_ROUNDS
        self.warmup_rounds = WARMUP_ROUNDS if gated else REPORT_WARMUP_ROUNDS

    @property
    def label(self) -> str:
        return f"b{self.bucket} {self.task} {self.rid} (|S| {self.n_states})"

    def spec(self) -> b13_run.StratumSpec:
        return b13_run.StratumSpec(
            bucket=self.bucket,
            gated=self.gated,
            rounds=self.rounds,
            label=self.label,
            payload_chars=len(self.payload),
        )

    def xargs(self, arm: str, tag: str | None) -> dict[str, Any] | None:
        """A and A′: the tap key only; B/E: the same plus the ``astra_cd_*`` keys; D: no xargs."""
        if arm in ("A", "A'"):
            return dict(TAP_XARGS)
        if arm == "D":
            return None
        assert tag is not None
        if arm == "B":
            return {**TAP_XARGS, **arms.cd_xargs(B_ARM, self.payload, tag, STOCK_KNOBS)}
        if arm == "E":
            return {**TAP_XARGS, **arms.cd_xargs(E_ARM, self.payload, tag)}
        raise ValueError(arm)


# --------------------------------------------------------------------------- the run


def response_id(row: dict[str, Any]) -> str | None:
    body = row.get("response")
    if row.get("status") == 200 and isinstance(body, dict) and body.get("id"):
        return str(body["id"])
    return None


def completion_tokens(row: dict[str, Any]) -> int | None:
    try:
        return int(row["response"]["usage"]["completion_tokens"])
    except (KeyError, TypeError, ValueError):
        return None


class Run:
    def __init__(self, out: Path, boot_log: Path, model: str) -> None:
        self.out = out
        self.boot_log = boot_log
        self.model = model
        self.run_id = uuid.uuid4().hex[:8]
        self.log = records.EngineLog(boot_log)  # tails from here: waits only
        self.sub: Subscriber | None = None
        self.rows: list[dict[str, Any]] = []
        self.flushes = 0
        self.warmup_ids: list[str] = []
        self._fh = (out / "requests.jsonl").open("a")

    def write(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        slim = dict(row)
        body = slim.get("response")
        if isinstance(body, dict):  # keep the evidence small: no token id list
            slim["response"] = {k: v for k, v in body.items() if k != "choices"} | {
                "content": _content(body)
            }
        self._fh.write(json.dumps(slim) + "\n")
        self._fh.flush()

    def post(self, client: httpx.Client, body: dict[str, Any], **meta: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        status, response = 0, None
        try:
            r = client.post(f"{V1}/chat/completions", json=body)
            status = r.status_code
            try:
                response = r.json()
            except ValueError:
                response = {"raw": r.text[:2000]}
        except httpx.HTTPError as exc:
            response = {"transport_error": repr(exc)}
        wall_ms = 1000.0 * (time.perf_counter() - t0)
        xa = body.get("vllm_xargs")
        row = {
            **meta,
            "wall_ms": wall_ms,
            "status": status,
            "response": response,
            "xargs_keys": sorted(xa) if isinstance(xa, dict) else None,
            "t_done": time.monotonic(),
        }
        self.write(row)
        return row

    def flush(self, client: httpx.Client) -> None:
        """T-0070 E9: a tiny unconstrained request, so records emitted at the next sampler call land."""
        self.flushes += 1
        body = prompts.request_body(self.model, FLUSH_PROMPT)
        body["max_tokens"] = 1
        body.pop("return_token_ids", None)
        self.post(client, body, name=f"flush:{self.flushes}", arm="flush", warmup=True)

    def settle(self, client: httpx.Client, row: dict[str, Any]) -> None:
        """Wait (bounded) for the request's commit frames and, for a constrained one, its records."""
        rid, ct = response_id(row), completion_tokens(row)
        if rid is not None and ct is not None and self.sub is not None:
            want = math.ceil(ct / CANVAS)
            deadline = time.monotonic() + FRAME_WAIT_S
            while time.monotonic() < deadline:
                got = sum(
                    1
                    for f in self.sub.snapshot()
                    if f.get("phase") == "commit"
                    and smoke.belongs(str(f.get("request_id", "")), rid)
                )
                if got >= want:
                    break
                time.sleep(0.02)
        tag = row.get("record_tag")
        if tag:
            joined = self.log.wait(tag, max_tokens=prompts.MAX_TOKENS, timeout_s=RECORD_WAIT_S)
            sent = 0
            while sent < MAX_FLUSHES and (
                joined is None or not records.is_complete(joined.records, prompts.MAX_TOKENS)
            ):
                self.flush(client)
                sent += 1
                joined = self.log.wait(tag, max_tokens=prompts.MAX_TOKENS, timeout_s=FLUSH_WAIT_S)
            row["flush_requests"] = sent
        time.sleep(GAP_S)

    def close(self) -> None:
        self._fh.close()


def _content(body: dict[str, Any]) -> str | None:
    try:
        c = body["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return None if c is None else str(c)[:400]


def warm_subscription(run: Run, client: httpx.Client, info: list[str]) -> None:
    """T-0069 O7, as ``scripts/gpu/cd_smoke.py`` does it: attempt 1 is the request the slow joiner loses."""
    assert run.sub is not None
    deadline = time.monotonic() + WARMUP_BUDGET_S
    per_attempt = WARMUP_BUDGET_S / WARMUP_ATTEMPTS
    start = len(run.sub.snapshot())
    for attempt in range(1, WARMUP_ATTEMPTS + 1):
        if time.monotonic() >= deadline:
            break
        body = prompts.request_body(run.model, FLUSH_PROMPT)
        body["max_tokens"] = WARMUP_MAX_TOKENS
        body.pop("return_token_ids", None)
        row = run.post(client, body, name="warmup:sub", arm="warmup", warmup=True, attempt=attempt)
        rid = response_id(row)
        if rid is not None:
            run.warmup_ids.append(rid)
        attempt_end = min(deadline, time.monotonic() + per_attempt)
        while run.warmup_ids:
            frames = run.sub.snapshot()[start:]
            if any(
                smoke.belongs(str(f.get("request_id", "")), w)
                for f in frames
                for w in run.warmup_ids
            ):
                info.append(f"subscriber warmed by warmup:sub attempt {attempt} (T-0069 O7)")
                return
            if time.monotonic() >= attempt_end:
                break
            time.sleep(0.1)
    raise SubscriberNotReady(
        f"no frame of {len(run.warmup_ids)} warmup:sub request(s) reached the subscriber within "
        f"{WARMUP_BUDGET_S:.0f} s: the tap subscription is not live; refusing before any measured request"
    )


def drive_stratum(run: Run, client: httpx.Client, s: Stratum) -> None:
    extra = s.extra or "D"
    plan = [(-(k + 1), o, True) for k, o in enumerate(b13_run.schedule(s.warmup_rounds, extra))]
    plan += [(r, o, False) for r, o in enumerate(b13_run.schedule(s.rounds, extra))]
    for round_idx, order, warm in plan:
        for slot, arm in enumerate(order):
            tag = None
            if arm in ("B", "E"):
                rr = f"w{-round_idx}" if warm else str(round_idx)
                tag = f"t71a.{run.run_id}.b{s.bucket}.{rr}.{arm}"
            body = prompts.request_body(run.model, s.prompt, s.xargs(arm, tag))
            row = run.post(
                client,
                body,
                name=f"b{s.bucket}:{'w' if warm else 'r'}{abs(round_idx)}:{arm}",
                stratum=s.bucket,
                round=round_idx,
                slot=slot,
                arm=arm,
                warmup=warm,
                record_tag=tag,
                task=s.task,
                dataset_id=s.rid,
            )
            run.settle(client, row)


# --------------------------------------------------------------------------- criterion 5


def run_runner(out: Path, boot_log: Path, head: str) -> tuple[list[str], list[str]]:
    lines = [
        f"Runner on the engine (criterion 5): runner.run, dev cut, limit {RUNNER_LIMIT}, budget_s "
        f"{RUNNER_BUDGET_S:.0f}, request_timeout_s {RUNNER_REQUEST_TIMEOUT_S:.0f}, every arm x both tasks; Mar "
        f"arms with the stock knobs {STOCK_KNOBS} through mar_knobs (the artifact's label reads "
        "'calibration': the runner's name for mar_knobs on dev; here it is the stock-knob dry run)"
    ]
    problems: list[str] = []
    for task in RUNNER_TASKS:
        for arm in arms.ARMS:
            name = f"runner {task}/{arm}"
            try:
                res = runner.run(
                    runner.RunConfig(
                        task=task,
                        cut="dev",
                        arm=arm,
                        base_url=V1,
                        engine_log=boot_log,
                        out_dir=out / "runner",
                        repo=REPO,
                        expected_commit=head,
                        limit=RUNNER_LIMIT,
                        budget_s=RUNNER_BUDGET_S,
                        request_timeout_s=RUNNER_REQUEST_TIMEOUT_S,
                        mar_knobs=STOCK_KNOBS if arm in arms.MAR_ARMS else None,
                    )
                )
            except Exception:
                tb = traceback.format_exc()
                problems.append(f"{name}: raised {tb.strip().splitlines()[-1]}")
                lines.append(f"  {name}: RAISED\n" + "\n".join("    " + x for x in tb.splitlines()))
                continue
            rows = runner.read_rows(res.artifact)
            errors: dict[str, int] = {}
            for r in rows:
                if r.get("error"):
                    errors[str(r["error"])] = errors.get(str(r["error"]), 0) + 1
            joins = errors.get("join_error", 0)
            ok = res.exit_code == 0 and joins == 0
            lines.append(
                f"  {name}: exit {res.exit_code}, stopped {res.stopped}, rows {len(rows)} of {res.n}, "
                f"join_error {joins}, errors {errors or 'none'}, summary {dict(res.summary)} -> "
                f"{'PASS' if ok else 'FAIL'}; {res.artifact.relative_to(REPO)}"
            )
            if not ok:
                problems.append(
                    f"{name}: exit {res.exit_code}, stopped {res.stopped}, join_error {joins}"
                )
    return lines, problems


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--boot-log", type=Path, default=REPO / "logs" / "vllm.log")
    ap.add_argument("--profile", default="lean")
    args = ap.parse_args()

    # ---- refusals (no inference request before these pass)
    dirty = git("status", "--porcelain")
    if dirty:
        die(2, f"refusing: the tree is dirty:\n{dirty}")
    drift = smoke.vllm_sh_problems(
        git("diff", f"{REFERENCE_COMMIT}..HEAD", "--", "services/vllm.sh")
    )
    if drift:
        die(2, "refusing: " + "; ".join(drift))
    if not args.boot_log.is_file():
        die(2, f"refusing: no engine log at {args.boot_log}")
    try:
        last_boot(args.boot_log.read_text(errors="replace"))
    except ValueError as exc:
        die(2, f"refusing: {exc}")
    head = git("rev-parse", "HEAD")
    sha = head[:7]
    out = REPO / "logs" / f"t0071a-b13-{sha}"
    if out.exists():
        die(2, f"refusing: {out} exists; move it aside first")
    try:
        strata = [Stratum(b, t, r, True) for b, (t, r) in GATED.items()]
    except Exception as exc:
        die(2, f"refusing: a gated stratum cannot be built: {exc}")
    skipped: list[str] = []
    #: R4: a report-only stratum that fails to build is a run problem (its spec carries the error)
    specs = [s.spec() for s in strata]
    for b, (t, r) in REPORT_ONLY.items():
        try:
            strata.append(Stratum(b, t, r, False))
            specs.append(strata[-1].spec())
        except Exception as exc:
            skipped.append(f"report-only stratum b{b} ({t} {r}) not run: {exc}")
            specs.append(
                b13_run.StratumSpec(
                    bucket=b,
                    gated=False,
                    rounds=REPORT_ROUNDS,
                    build_error=f"{type(exc).__name__}: {exc}",
                    label=f"b{b} {t} {r}",
                )
            )

    client = httpx.Client(timeout=1800.0)
    try:
        model = client.get(f"{V1}/models").json()["data"][0]["id"]
    except Exception as exc:
        die(2, f"no engine on {API} ({exc}); the locked script boots it")
    probe = client.post(
        f"{V1}/chat/completions",
        json={
            "model": model,
            "max_tokens": 8,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    if probe.status_code != 400:
        die(2, f"temperature returned {probe.status_code}, not 400: not the diffusion path")

    out.mkdir(parents=True)
    mem_boot = gpu_memory()
    print(f"device memory after boot: {mem_boot}", flush=True)
    load = os.getloadavg()
    thr = thresholds()
    regime = (
        f"T-0071a a2: profile {args.profile} (engine only), commit {sha} clean, {model} rev 2ea8372, vllm "
        f"v0.29.0, CL {CANVAS}, 48 steps, t 0.8->0.4, TRITON_ATTN, bf16 KV 4 GiB, max_model_len 16384, "
        f"max_num_seqs 3, plugin Tap∘CD, tap {TAP} at 30 Hz, enable_thinking=false, max_tokens "
        f"{prompts.MAX_TOKENS}, JSON whitespace, stock knobs {STOCK_KNOBS['entropy_bound']}/"
        f"{STOCK_KNOBS['confidence_threshold']} (sent explicitly on B), no seed, concurrency 1; gated "
        f"strata {', '.join(s.label for s in strata if s.gated)}: {WARMUP_ROUNDS} warm-up + {ROUNDS} rounds "
        f"of A/B/A′ rotated (ABA′, BA′A, A′AB) + D at slot r % 4; A, A′ xargs {TAP_XARGS}, B the same + "
        f"perstep_mar astra_cd_*, D no xargs; thresholds b64 {thr[64]:g} %, b128 {thr[128]:g} %; report-only "
        f"strata {', '.join(s.label for s in strata if not s.gated) or 'none'}: {REPORT_WARMUP_ROUNDS} warm-up "
        f"+ {REPORT_ROUNDS} rounds of A/B/A′ + E (end_projection) at slot r % 4; dev cuts only; host load "
        f"{load[0]:.2f}/{load[1]:.2f}/{load[2]:.2f}, python {platform.python_version()}"
    )
    print(regime, flush=True)

    info: list[str] = []
    run = Run(out, args.boot_log, model)
    crashed: str | None = None
    refused: str | None = None
    try:
        run.sub = Subscriber()
        time.sleep(0.5)  # SUB slow-joiner
        warm_subscription(run, client, info)
        for s in strata:
            t0 = time.monotonic()
            drive_stratum(run, client, s)
            info.append(f"{s.label}: {time.monotonic() - t0:.0f} s")
    except SubscriberNotReady as exc:
        refused = str(exc)
    except Exception:
        crashed = traceback.format_exc()
        print(crashed, file=sys.stderr)
    finally:
        time.sleep(1.0)
        if run.sub is not None:
            run.sub.close()
        run.close()

    received = run.sub.snapshot() if run.sub is not None else []
    frames = [
        f
        for f in received
        if not any(smoke.belongs(str(f.get("request_id", "")), w) for w in run.warmup_ids)
    ]
    with (out / "frames.jsonl").open("w") as fh:
        for f in frames:
            fh.write(json.dumps(f) + "\n")
    engine_log = last_boot(args.boot_log.read_text(errors="replace"))
    (out / "engine.log").write_text(engine_log)
    head_lines = [
        regime,
        f"device memory after boot: {mem_boot}",
        f"requests: {len(run.rows)} ({run.flushes} flush requests); frames: {len(frames)} "
        f"({len(received) - len(frames)} warm-up frames excluded, T-0069 O7)",
        *skipped,
        *info,
        "note for H1 (R5): CD's admission (sampler.add_request) makes one host sync per constrained request by "
        "design; it lands in B's per-request cost, which F̂ (fitted on A/A′) does not subtract, so it is spread "
        "over B's Σsteps (ruling T1) and counts against B",
    ]
    if REFERENCE.is_file():
        bp = smoke.boot_problems(engine_log, REFERENCE.read_text())
        head_lines.append(
            f"boot config vs {REFERENCE.relative_to(REPO)} (informational): "
            + ("identical" if not bp else "; ".join(bp[:10]))
        )
    if refused is not None:
        report = "\n".join(
            [*head_lines, f"REFUSED: {refused}", "verdict: REFUSED (nothing measured)"]
        )
        (out / "report.txt").write_text(report + "\n")
        print(report)
        client.close()
        return 2

    problems_all: list[str] = []
    b13_lines: list[str] = []
    verdicts: dict[str, Any] = {}
    # R4: a crash during the requests still evaluates whatever completed; the crash is a named run problem
    crash_line = crashed.strip().splitlines()[-1] if crashed else None
    eval_crash: str | None = None
    steps_of: dict[int, int] = {}
    try:
        chk = b13_run.check_rows(run.rows, frames, engine_log)
        steps_of = chk.steps
        problems_all += chk.run_problems
        rows = [{**r, "steps": chk.steps.get(i)} for i, r in enumerate(run.rows)]
        ev = b13_run.evaluate_b13(rows, chk.problems, specs, thresholds(), crash=crash_line)
        problems_all += ev.run_problems
        row_lines = [
            f"  {run.rows[i]['name']}: {p}"
            for i, ps in sorted(chk.problems.items())
            for p in ps[:5]
        ]
        b13_lines = [
            f"ruling T3 (commit frame step == CD record steps, per canvas, B and E): "
            f"{chk.t3_canvases} canvases compared, {len(chk.t3_mismatches)} mismatches",
            *(f"  {m}" for m in chk.t3_mismatches[:20]),
            *ev.lines,
            "",
            f"row problems (rows dropped or excluded as described above): {len(row_lines)}",
            *row_lines[:200],
        ]
        verdicts = {
            "gates": {
                METRIC[b]: {
                    "verdict": g.verdict,
                    "fixed_ms": g.fixed,
                    "se_ms": g.se,
                    "at_fit": vars(g.at_fit) if g.at_fit else None,
                    "at_zero": vars(g.at_zero) if g.at_zero else None,
                    "at_upper": vars(g.at_upper) if g.at_upper else None,
                    "why": g.why,
                }
                for b, g in ev.gates.items()
            },
            "pooled_fixed_ms_sensitivity": ev.pooled_fixed,
            "pooled_sensitivity_verdicts": {METRIC[b]: v for b, v in ev.pooled_sensitivity.items()},
            "xargs_exposure_d_vs_a": {f"b{b}": vars(x) for b, x in ev.exposure.items()},
            "dropped_rounds": {f"b{b}": d for b, d in ev.dropped.items()},
            "t3": {"canvases": chk.t3_canvases, "mismatches": len(chk.t3_mismatches)},
            "crash": crash_line,
        }
    except Exception:
        eval_crash = traceback.format_exc()
        print(eval_crash, file=sys.stderr)
        problems_all.append(f"evaluation raised: {eval_crash.strip().splitlines()[-1]}")
        if crash_line:
            problems_all.append(f"crash: {crash_line}")
    mem_end = gpu_memory()
    (out / "rows.json").write_text(
        json.dumps(
            [
                {**{k: v for k, v in r.items() if k != "response"}, "steps": steps_of.get(i)}
                for i, r in enumerate(run.rows)
            ],
            indent=1,
            default=str,
        )
    )

    def write_report(extra: list[str], status: str) -> str:
        body = [*head_lines, f"device memory after the B13 run: {mem_end}", "", *b13_lines, *extra]
        if crashed is not None:
            body += [
                "CRASHED (the B13 run did not complete):",
                *("  " + x for x in crashed.splitlines()),
            ]
        body.append("")
        body.append(f"PROBLEMS: {len(problems_all)}")
        body += [f"  {p}" for p in problems_all[:200]]
        body.append(f"run: {status}")
        text = "\n".join(body) + "\n"
        (out / "report.txt").write_text(text)
        (out / "verdict.json").write_text(json.dumps(verdicts, indent=1, default=str))
        return text

    write_report([], "B13 part written; runner (criterion 5) not yet run")
    runner_lines: list[str] = []
    if crashed is None:
        runner_lines, rp = run_runner(out, args.boot_log, head)
        problems_all += rp
    else:
        runner_lines = ["Runner (criterion 5): not run, the B13 requests crashed (see PROBLEMS)"]
        problems_all.append("criterion 5 not run: the B13 requests crashed")
    client.close()
    code = b13_run.exit_code(problems_all)
    status = "COMPLETE" if code == 0 else "DEFECTIVE (see PROBLEMS)"
    report = write_report(["", *runner_lines, f"device memory at the end: {gpu_memory()}"], status)
    print(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
