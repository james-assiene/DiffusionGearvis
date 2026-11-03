"""T-0069 tier 3 -- the Tap ∘ CD ∘ stock smoke on a booted engine (criteria 4-9). `/launch` ONLY.

    uv run --group cd python scripts/gpu/cd_smoke.py [--boot-log logs/vllm.log] [--profile lean]

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It
never boots one; `scripts/locked/t0069_cd_smoke.sh` does (engine only, `services/vllm.sh`, the plugin wheel
built from this commit). **It makes no latency or accuracy claim**: wall times are logged and labelled
"not a measurement".

Refusals (exit 2, before any inference request): a dirty tree; `git diff 806da12..HEAD --
services/vllm.sh` non-empty; the boot reference `logs/t0011-tier3-806da12/boot-reference.txt` missing; the
evidence directory already present; no engine on 127.0.0.1:8001; an engine NOT on the diffusion path (a
`temperature` probe must be a 400).

Order (concurrency 1 except criterion 5):

- warm-up (ruling O7): one small unconstrained `warmup:sub` request, retried up to 3 times within 30 s,
  until at least one of its tap frames reaches the subscriber (the tap's PUB socket is created on the first
  publish, so the first request after boot loses frames). None received: exit 2 with `report.txt` saying
  so, before any criterion. Its frames are dropped from `frames.jsonl` and from every check; its row stays
  in `requests.jsonl` under a name no check reads.
- c8: one unconstrained request with no xargs, then a flush request; the engine-log bytes between are the
  window (no `ASTRA_CD ` / `astra_cd ` line); its frames validate and its commit frames are
  `ceil(completion_tokens / 256)`. Run first, before any constrained request.
- c9: 20 unconstrained dev records (10 Sudoku, 10 BFCL), `return_token_ids`; first 8 ids, header rate
  (reported, not gated).
- c4: one Sudoku-dev and one BFCL-dev request per arm (`end_projection`, `perstep_mf`, `perstep_mar`,
  `perstep_mar_sc`), stock bound/threshold (no calibration), JSON whitespace, built with
  `CdRequest.to_xargs()`.
- c5: G (constrained, valid) starts; at G's first tap `denoise` frame X is sent (payload altered, sha
  recomputed; the host `parse_cd_xargs` predicts its code); then an unconstrained request and `/health`.
- c6: a synthetic 270-token literal (bucket 512), `max_tokens` 300, under `perstep_mar` and
  `end_projection`.
- c7: hand-built payloads at `cap - 512` (correct sha, not an archive) and `cap + 1` chars.

Evidence (`logs/t0069-smoke-<sha>/`, layout fixed by `tests/integration/test_cd_engine_smoke.py`):
`requests.jsonl`, `frames.jsonl`, `engine.log` (the last boot of the engine log, R16), `c8-window.log`,
`report.txt`. Joins: tap frames by response-id prefix (`smoke.belongs`), CD records by `record_tag`. Free
device memory is logged after boot and at the end. The report is written before the exit code; the exit
code is the verdict (0 pass, 1 fail, 2 refused).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import msgpack

from local_astra.bench.b16_tap import last_boot, parse_tap_release_lines
from local_astra.contracts.cd import MAX_AUTOMATON_CHARS, CdRequest
from local_astra.eval.cd import arms, cuts, prompts, records, replay, smoke

REPO = Path(__file__).resolve().parents[2]
API = "http://127.0.0.1:8001"
V1 = API + "/v1"
TAP = "tcp://127.0.0.1:8002"
TOPIC = b"astra.canvas.step"
REFERENCE_COMMIT = "806da12"
REFERENCE = REPO / "logs" / "t0011-tier3-806da12" / "boot-reference.txt"
PLUGIN_SRC = REPO / "plugin" / "astra_vllm_plugin"
TASKS = ("sudoku", "bfcl")
C4_ARMS = ("end_projection", "perstep_mf", "perstep_mar", "perstep_mar_sc")
C6_ARMS = ("perstep_mar", "end_projection")
C6_MAX_TOKENS = 300
LITERAL_LEN = 270
LITERAL_TEXT = (
    "The lighthouse keeper climbed the spiral stair at dusk, trimmed the wick, wound the clockwork "
    "that turned the great lens, and wrote the weather, the sea and the passing ships in the log. "
)
#: c4/c5 grammars stay on the live-path gate (`.claude/rules/gpu-services.md`: <= 128 states); only c6
#: is bucket 512, by design
MAX_STATES = 128
HEADER_PER_TASK = 10
RECORD_WAIT_S = 10.0
FLUSH_WAIT_S = 5.0
MAX_FLUSHES = 2
G_DENOISE_TIMEOUT_S = 120.0
FLUSH_PROMPT = "Reply with OK."
#: Ruling O7: the subscription warm-up before c8
WARMUP_ATTEMPTS = 3
WARMUP_BUDGET_S = 30.0
WARMUP_MAX_TOKENS = 8


def die(code: int, msg: str) -> None:
    print(f"cd_smoke: {msg}", file=sys.stderr)
    raise SystemExit(code)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


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


def host_parse() -> Any:
    """The plugin's own ``parse_cd_xargs`` (stdlib + numpy) for the host-predicted codes."""
    if str(PLUGIN_SRC) not in sys.path:
        sys.path.insert(0, str(PLUGIN_SRC))
    from astra_vllm_plugin.cd import xargs as plugin_xargs

    return plugin_xargs


def predicted_code(xargs: dict[str, Any]) -> str | None:
    mod = host_parse()
    try:
        mod.parse_cd_xargs(dict(xargs))
    except mod.CdRequestError as exc:
        return str(exc.reason)
    return None


def sha(payload: str) -> str:
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


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


# --------------------------------------------------------------------------- evidence + requests


class Run:
    def __init__(self, out: Path, boot_log: Path, model: str) -> None:
        self.out = out
        self.boot_log = boot_log
        self.model = model
        self.run_id = uuid.uuid4().hex[:8]
        self.log = records.EngineLog(boot_log)  # tails from here: waits only
        self.sub: Subscriber | None = None
        self.rows: dict[str, dict[str, Any]] = {}
        self._seq = 0
        self._lock = threading.Lock()
        self._fh = (out / "requests.jsonl").open("a")
        self.flushes = 0
        #: response ids of the ``warmup:sub`` requests (ruling O7): their frames are in no criterion
        self.warmup_ids: list[str] = []

    def tag(self, *parts: str) -> str:
        return ".".join(["t69", self.run_id, *parts])

    def write(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.rows[row["name"]] = row
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()

    def post(
        self,
        name: str,
        body: dict[str, Any],
        *,
        task: str | None,
        dataset_id: str | None,
        arm: str,
        record_tag: str | None,
        client: httpx.Client,
        **extra: Any,
    ) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            seq = self._seq
        t0 = time.monotonic()
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
        row = {
            "name": name,
            "seq": seq,
            "task": task,
            "dataset_id": dataset_id,
            "arm": arm,
            "record_tag": record_tag,
            "max_tokens": body.get("max_tokens"),
            "xargs": body.get("vllm_xargs"),
            "status": status,
            "response": response,
            "wall_ms_not_a_measurement": round(1000.0 * (time.monotonic() - t0), 1),
            **extra,
        }
        self.write(row)
        return row

    def flush(self, client: httpx.Client, name: str | None = None) -> dict[str, Any]:
        """Ruling E9: one tiny unconstrained request, so records emitted at the next sampler call land."""
        self.flushes += 1
        body = prompts.request_body(self.model, FLUSH_PROMPT)
        body["max_tokens"] = 1
        body.pop("return_token_ids", None)
        return self.post(
            name or f"flush:{self.flushes}",
            body,
            task=None,
            dataset_id=None,
            arm="unconstrained",
            record_tag=None,
            client=client,
        )

    def await_records(self, tag: str, max_tokens: int, client: httpx.Client) -> None:
        joined = self.log.wait(tag, max_tokens=max_tokens, timeout_s=RECORD_WAIT_S)
        sent = 0
        while sent < MAX_FLUSHES and (
            joined is None or not records.is_complete(joined.records, max_tokens)
        ):
            self.flush(client)
            sent += 1
            joined = self.log.wait(tag, max_tokens=max_tokens, timeout_s=FLUSH_WAIT_S)

    def close(self) -> None:
        self._fh.close()


def response_id(row: dict[str, Any]) -> str | None:
    body = row.get("response")
    if row.get("status") == 200 and isinstance(body, dict):
        rid = body.get("id")
        return str(rid) if rid else None
    return None


def content_of(row: dict[str, Any]) -> str | None:
    body = row.get("response") or {}
    try:
        msg = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    c = msg.get("content")
    return None if c is None else str(c)


def token_ids_of(row: dict[str, Any]) -> list[int] | None:
    body = row.get("response") or {}
    try:
        ids = body["choices"][0].get("token_ids")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return [int(t) for t in ids] if isinstance(ids, list) else None


def completion_tokens(row: dict[str, Any]) -> int | None:
    body = row.get("response") or {}
    try:
        return int(body["usage"]["completion_tokens"])
    except (KeyError, TypeError, ValueError):
        return None


def cd_xargs(arm: str, payload: str, tag: str) -> dict[str, Any]:
    """``CdRequest.to_xargs()`` with the stock bound/threshold (no calibration: none is sent)."""
    s = arms.spec(arm)
    assert s.mode is not None and s.self_conditioning is not None
    return CdRequest(
        mode=s.mode,
        self_conditioning=s.self_conditioning,
        automaton=payload,
        automaton_sha256=sha(payload),
        record_tag=tag,
    ).to_xargs()


# --------------------------------------------------------------------------- grammars


def pick_case(task: str, skip: int = 0) -> tuple[Any, Any, str]:
    """The first dev case (after ``skip``) whose automaton passes the T-0070 gate and the cap."""
    from local_astra.decoding import report
    from local_astra.decoding.artifact import encode_automaton

    for case in list(cuts.load(task, "dev"))[skip:]:
        automaton = replay.compile_automaton(task, case, whitespace="json")
        try:
            report.gate(automaton, MAX_STATES)
        except report.TooLarge:
            continue
        payload = encode_automaton(automaton)
        if len(payload) <= MAX_AUTOMATON_CHARS:
            return case, automaton, payload
    die(2, f"no {task} dev case passes the {MAX_STATES}-state gate and the cap")
    raise AssertionError  # unreachable


def literal_automaton() -> tuple[list[int], Any, str]:
    """A synthetic literal of ``LITERAL_LEN`` token ids (a chain DFA), compiled with the vendored
    compiler's back half (stop augmentation, ``d``, classes, bucketing): bucket 512 by construction."""
    from local_astra.decoding import vocab
    from local_astra.decoding._vendor.dgfa_compile.automaton import compile_automaton
    from local_astra.decoding._vendor.dgfa_compile.minimize import Dfa
    from local_astra.decoding.artifact import encode_automaton

    tok = vocab.load_tokenizer()
    reserved = set(vocab.stop_token_ids()) | {0}
    ids: list[int] = []
    text = ""
    while len(ids) < LITERAL_LEN:
        text += LITERAL_TEXT
        ids = [t for t in tok.encode(text, add_special_tokens=False).ids if t not in reserved]
    ids = ids[:LITERAL_LEN]
    dfa = Dfa(
        n_states=LITERAL_LEN + 1,
        transitions=tuple((i, t, i + 1) for i, t in enumerate(ids)),
        start=0,
        finals=frozenset({LITERAL_LEN}),
    )
    automaton = compile_automaton(
        dfa,
        name="t0069-literal",
        end_tokens=vocab.stop_token_ids(),
        vocab_size=int(tok.get_vocab_size(with_added_tokens=True)),
        tokenizer_hash=vocab.tokenizer_hash(),
        compiler_version="t0069-literal",
    )
    if int(automaton.n_states_bucket) != 512:
        die(2, f"the literal compiled to bucket {automaton.n_states_bucket}, not 512")
    return ids, automaton, encode_automaton(automaton)


def altered(payload: str) -> tuple[str, str]:
    """X's payload: a valid one truncated (whole base64 quanta) -- the host must predict a loud code."""
    for cut in (len(payload) // 2, len(payload) // 3, 64):
        bad = payload[: cut - cut % 4]
        code = predicted_code(
            {
                "astra_cd_mode": "end_projection",
                "astra_cd_automaton": bad,
                "astra_cd_automaton_sha256": sha(bad),
                "astra_cd_record_tag": "probe",
            }
        )
        if code in smoke.LOUD_CODES:
            return bad, code
    die(2, "no truncation of the payload gives bad_encoding or structure on the host")
    raise AssertionError  # unreachable


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boot-log", type=Path, default=REPO / "logs" / "vllm.log")
    ap.add_argument("--profile", default="lean")
    args = ap.parse_args()

    # ---- refusals (no inference request before these pass)
    dirty = git("status", "--porcelain").strip()
    if dirty:
        die(2, f"refusing: the tree is dirty:\n{dirty}")
    drift = smoke.vllm_sh_problems(
        git("diff", f"{REFERENCE_COMMIT}..HEAD", "--", "services/vllm.sh")
    )
    if drift:
        die(2, "refusing: " + "; ".join(drift))
    if not REFERENCE.is_file():
        die(
            2,
            f"refusing: the boot reference {REFERENCE} is missing (copy it from the t0011 worktree)",
        )
    if not args.boot_log.is_file():
        die(2, f"refusing: no engine log at {args.boot_log}")
    try:
        last_boot(args.boot_log.read_text(errors="replace"))
    except ValueError as exc:
        die(2, f"refusing: {exc}")
    sha_head = git("rev-parse", "--short", "HEAD").strip()
    out = REPO / "logs" / f"t0069-smoke-{sha_head}"
    if out.exists():
        die(2, f"refusing: {out} exists; move it aside first")

    client = httpx.Client(timeout=900.0)
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

    prep = Prep()  # host compiles; a refusal here (exit 2) leaves no evidence directory
    out.mkdir(parents=True)
    mem_boot = gpu_memory()
    print(f"device memory after boot: {mem_boot}")
    load = os.getloadavg()
    regime = (
        f"T-0069 tier 3 smoke: profile {args.profile} (engine only), commit {sha_head} clean, {model} "
        f"rev 2ea8372, vllm v0.29.0, canvas 256, steps 48, TRITON_ATTN, bf16 KV 4 GiB, max_model_len "
        f"16384, max_num_seqs 3, plugin Tap∘CD, tap {TAP} at 30 Hz, enable_thinking=false, max_tokens "
        f"256 ({C6_MAX_TOKENS} in c6), stock bound/threshold (no calibration), JSON whitespace, "
        f"concurrency 1 (2 in c5), no seed, n=1 per arm per task, {2 * HEADER_PER_TASK} header "
        f"records, host load {load[0]:.2f}/{load[1]:.2f}/{load[2]:.2f}, python "
        f"{platform.python_version()}; no latency or accuracy claim"
    )
    print(regime)

    verdicts: dict[str, list[str]] = {}
    info: list[str] = []
    run = Run(out, args.boot_log, model)
    c8_window = ""
    crashed: str | None = None
    refused: str | None = None
    try:
        c8_window = drive(run, prep, client, info)
    except SubscriberNotReady as exc:
        refused = str(exc)
        print(f"cd_smoke: {refused}", file=sys.stderr)
    except Exception:
        crashed = traceback.format_exc()
        print(crashed, file=sys.stderr)
    finally:
        time.sleep(1.0)
        if run.sub is not None:
            run.sub.close()
        run.close()
        client.close()

    # ---- evidence files (always written)
    received = run.sub.snapshot() if run.sub is not None else []
    # ruling O7: the warm-up's frames are evidence of no criterion
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
    (out / "c8-window.log").write_text(c8_window)
    mem_end = gpu_memory()

    if crashed is None and refused is None:
        try:
            evaluate(run, prep.automata(), frames, engine_log, c8_window, verdicts, info)
        except Exception:
            crashed = traceback.format_exc()
            print(crashed, file=sys.stderr)

    failed = sorted(k for k, v in verdicts.items() if v)
    lines = [
        regime,
        f"device memory after boot: {mem_boot}",
        f"device memory at the end: {mem_end}",
        f"requests: {len(run.rows)} ({run.flushes} flush requests); frames: {len(frames)} "
        f"({len(received) - len(frames)} warm-up frames excluded, ruling O7)",
    ]
    if refused is not None:
        lines += [f"REFUSED: {refused}", "verdict: REFUSED (no criterion ran)"]
        report = "\n".join(lines) + "\n"
        (out / "report.txt").write_text(report)
        print(report)
        return 2
    for key in sorted(verdicts):
        problems = verdicts[key]
        lines.append(f"{key}: {'FAIL' if problems else 'PASS'}")
        lines.extend(f"  {p}" for p in problems[:30])
    lines.extend(info)
    if crashed is not None:
        lines.append("CRASHED (the run did not complete):")
        lines.extend("  " + ln for ln in crashed.splitlines())
    ok = crashed is None and not failed and bool(verdicts)
    lines.append(
        f"verdict: {'PASS' if ok else 'FAIL'}" + (f" ({', '.join(failed)})" if failed else "")
    )
    report = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(report)
    print(report)
    return 0 if ok else 1


class SubscriberNotReady(Exception):
    """Ruling O7: no frame of the warm-up request reached the subscriber; the run is refused (exit 2)."""


def warm_subscription(run: Run, client: httpx.Client, info: list[str]) -> None:
    """Ruling O7: the tap's PUB socket is created lazily on the first publish, so the first request after
    boot loses frames to any subscriber (ZMQ slow joiner). Before any criterion, send one small
    unconstrained ``warmup:sub`` request (retried up to ``WARMUP_ATTEMPTS`` times, all within
    ``WARMUP_BUDGET_S``) until at least one of its frames is received. Its frames are excluded from every
    criterion (``main`` drops them from ``frames.jsonl``); its row stays in ``requests.jsonl`` under a name
    no check reads."""
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
        row = run.post(
            "warmup:sub",
            body,
            task=None,
            dataset_id=None,
            arm="unconstrained",
            record_tag=None,
            client=client,
            attempt=attempt,
        )
        rid = response_id(row)
        if rid is not None:
            run.warmup_ids.append(rid)
        # each attempt waits its own share of the budget, then the next attempt is sent: attempt 1 is the
        # request the slow joiner loses, so it must not consume the whole budget. A frame of ANY warm-up
        # attempt counts (a late frame of attempt 1 still proves the subscription is live).
        attempt_end = min(deadline, time.monotonic() + per_attempt)
        while run.warmup_ids:
            frames = run.sub.snapshot()[start:]
            if any(
                smoke.belongs(str(f.get("request_id", "")), w)
                for f in frames
                for w in run.warmup_ids
            ):
                info.append(f"subscriber warmed by warmup:sub attempt {attempt} (ruling O7)")
                return
            if time.monotonic() >= attempt_end:
                break
            time.sleep(0.1)
    raise SubscriberNotReady(
        f"no frame of {len(run.warmup_ids)} warmup:sub request(s) reached the subscriber within "
        f"{WARMUP_BUDGET_S:.0f} s ({WARMUP_ATTEMPTS} attempts max): the tap subscription is not live; "
        "refusing before any criterion runs (ruling O7)"
    )


class Prep:
    """Every grammar of the run, compiled before the evidence directory exists (a refusal here leaves
    nothing behind) and before the first request (so the requests run back to back)."""

    def __init__(self) -> None:
        self.c4 = {task: pick_case(task) for task in TASKS}
        self.g_case, self.g_automaton, self.g_payload = pick_case("sudoku", skip=1)
        self.x_payload, self.x_predicted = altered(self.c4["bfcl"][2])
        self.literal_ids, self.literal_automaton, self.literal_payload = literal_automaton()

    def automata(self) -> dict[str, Any]:
        return {
            "sudoku": self.c4["sudoku"][1],
            "bfcl": self.c4["bfcl"][1],
            "G": self.g_automaton,
            "literal": self.literal_automaton,
        }


def drive(run: Run, prep: Prep, client: httpx.Client, info: list[str]) -> str:
    """Send every request of the run; returns criterion 8's log window."""
    c4_cases = prep.c4
    g_case, g_payload = prep.g_case, prep.g_payload
    x_payload, x_predicted = prep.x_payload, prep.x_predicted
    literal_ids, literal_payload = prep.literal_ids, prep.literal_payload

    run.sub = Subscriber()
    time.sleep(0.5)  # SUB slow-joiner
    warm_subscription(run, client, info)

    # ---- c8: quiet window, first (no constrained request before it)
    quiet_case = cuts.load("sudoku", "dev")[0]
    offset = run.boot_log.stat().st_size
    quiet = run.post(
        "c8:quiet",
        prompts.request_body(run.model, prompts.build_prompt("sudoku", quiet_case)),
        task="sudoku",
        dataset_id=str(quiet_case.id),
        arm="unconstrained",
        record_tag=None,
        client=client,
    )
    run.flush(client, name="c8:flush")
    rid = response_id(quiet)
    window = ""
    deadline = time.monotonic() + 10.0
    while True:
        with run.boot_log.open("rb") as fh:
            fh.seek(offset)
            window = fh.read().decode("utf-8", errors="replace")
        if rid is None or f"canvas tap release: req_id={rid}-" in window:
            break
        if time.monotonic() > deadline:
            break
        time.sleep(0.2)

    # ---- c9: 20 unconstrained dev records
    k = 0
    for task in TASKS:
        for case in list(cuts.load(task, "dev"))[:HEADER_PER_TASK]:
            run.post(
                f"c9:header:{k}",
                prompts.request_body(run.model, prompts.build_prompt(task, case)),
                task=task,
                dataset_id=str(case.id),
                arm="unconstrained",
                record_tag=None,
                client=client,
            )
            k += 1

    # ---- c4: every arm end to end, both tasks
    for task in TASKS:
        case, _automaton, payload = c4_cases[task]
        prompt = prompts.build_prompt(task, case)
        for arm in C4_ARMS:
            tag = run.tag("c4", task, arm)
            run.post(
                f"c4:{task}:{arm}",
                prompts.request_body(run.model, prompt, cd_xargs(arm, payload, tag)),
                task=task,
                dataset_id=str(case.id),
                arm=arm,
                record_tag=tag,
                client=client,
            )
            run.await_records(tag, prompts.MAX_TOKENS, client)

    # ---- c5: G valid, X loud while G denoises, then the engine keeps serving
    assert run.sub is not None
    g_tag, x_tag = run.tag("c5", "G"), run.tag("c5", "X")
    g_body = prompts.request_body(
        run.model,
        prompts.build_prompt("sudoku", g_case),
        cd_xargs("perstep_mar", g_payload, g_tag),
    )
    x_xargs = {
        "astra_cd_mode": "end_projection",
        "astra_cd_automaton": x_payload,
        "astra_cd_automaton_sha256": sha(x_payload),
        "astra_cd_record_tag": x_tag,
    }
    x_body = prompts.request_body(
        run.model, prompts.build_prompt("bfcl", c4_cases["bfcl"][0]), x_xargs
    )
    time.sleep(1.0)  # late frames of the previous request publish before the snapshot
    known = [rid for r in list(run.rows.values()) if (rid := response_id(r))]
    before = len(run.sub.snapshot())
    g_done = threading.Event()

    def send_g() -> None:
        with httpx.Client(timeout=900.0) as gc:
            run.post(
                "c5:G",
                g_body,
                task="sudoku",
                dataset_id=str(g_case.id),
                arm="perstep_mar",
                record_tag=g_tag,
                client=gc,
            )
        g_done.set()

    g_thread = threading.Thread(target=send_g, daemon=True)
    g_thread.start()
    trigger: int | None = None
    deadline = time.monotonic() + G_DENOISE_TIMEOUT_S
    while trigger is None and time.monotonic() < deadline and not g_done.is_set():
        frames = run.sub.snapshot()
        # G's id is unknown until it returns: its first denoise frame is the first one after the
        # snapshot that belongs to no earlier response (the post-hoc check joins by prefix)
        trigger = next(
            (
                i
                for i in range(before, len(frames))
                if frames[i].get("phase") == "denoise"
                and not any(smoke.belongs(str(frames[i].get("request_id", "")), k) for k in known)
            ),
            None,
        )
        if trigger is None:
            time.sleep(0.005)
    if trigger is None:
        info.append("c5: no denoise frame of G arrived before G finished; X sent after G")
    run.post(
        "c5:X",
        x_body,
        task="bfcl",
        dataset_id=str(c4_cases["bfcl"][0].id),
        arm="end_projection",
        record_tag=x_tag,
        client=client,
        predicted=x_predicted,
        trigger_frame_index=trigger,
    )
    g_thread.join(900.0)
    run.await_records(g_tag, prompts.MAX_TOKENS, client)
    run.await_records(x_tag, prompts.MAX_TOKENS, client)
    next_case = cuts.load("sudoku", "dev")[2]
    run.post(
        "c5:next",
        prompts.request_body(run.model, prompts.build_prompt("sudoku", next_case)),
        task="sudoku",
        dataset_id=str(next_case.id),
        arm="unconstrained",
        record_tag=None,
        client=client,
    )
    try:
        health = client.get(f"{API}/health").status_code
    except httpx.HTTPError:
        health = 0
    run.write(
        {
            "name": "c5:health",
            "task": None,
            "dataset_id": None,
            "arm": "unconstrained",
            "record_tag": None,
            "max_tokens": None,
            "xargs": None,
            "status": health,
            "response": None,
        }
    )

    # ---- c6: a bucket-512 literal across two canvases
    lit_prompt = "Repeat the following text exactly, and nothing else:\n\n" + LITERAL_TEXT * 2
    for arm in C6_ARMS:
        tag = run.tag("c6", arm)
        body = prompts.request_body(run.model, lit_prompt, cd_xargs(arm, literal_payload, tag))
        body["max_tokens"] = C6_MAX_TOKENS
        run.post(
            f"c6:{arm}",
            body,
            task="literal",
            dataset_id=None,
            arm=arm,
            record_tag=tag,
            client=client,
            literal_ids=literal_ids,
        )
        run.await_records(tag, C6_MAX_TOKENS, client)

    # ---- c7: transport at the cap, hand-built (W4 exception, this row only)
    for name, length in (("near", MAX_AUTOMATON_CHARS - 512), ("over", MAX_AUTOMATON_CHARS + 1)):
        tag = run.tag("c7", name)
        payload = "A" * length
        xargs = {
            "astra_cd_mode": "end_projection",
            "astra_cd_automaton": payload,
            "astra_cd_automaton_sha256": sha(payload),
            "astra_cd_record_tag": tag,
        }
        row = run.post(
            f"c7:{name}",
            prompts.request_body(run.model, FLUSH_PROMPT, xargs),
            task=None,
            dataset_id=None,
            arm="end_projection",
            record_tag=tag,
            client=client,
            predicted=predicted_code(xargs),
        )
        if row["status"] == 200:
            run.await_records(tag, prompts.MAX_TOKENS, client)
    return window


def evaluate(
    run: Run,
    automata: dict[str, Any],
    frames: list[dict[str, Any]],
    engine_log: str,
    window: str,
    verdicts: dict[str, list[str]],
    info: list[str],
) -> None:
    """Every verdict from the saved evidence: records re-read from ``engine.log`` by ``record_tag``."""
    log = records.EngineLog(run.out / "engine.log", from_start=True)
    log.poll()
    if log.malformed:
        info.append(
            f"engine.log: {log.malformed} ASTRA_CD lines did not validate as CdCanvasRecord"
        )

    def recs(tag: str | None) -> list[dict[str, Any]]:
        if not tag:
            return []
        return [r.model_dump(mode="json") for r in log.records(tag)]

    releases = parse_tap_release_lines(engine_log)

    def accepts(automaton: Any) -> Callable[[Sequence[int]], bool]:
        return lambda ids: replay.accepts(automaton, ids)

    verdicts["c4 every frame of the run validates"] = smoke.frame_problems(frames)
    if not frames:
        verdicts["c4 every frame of the run validates"].append("no tap frame was recorded")

    # ---- c4 (+ c9's constrained token_ids)
    preempted = 0
    included: dict[str, bool | None] = {}
    token_problems: list[str] = []
    for task in TASKS:
        for arm in C4_ARMS:
            name = f"c4:{task}:{arm}"
            row = run.rows.get(name)
            p: list[str] = []
            if row is None:
                verdicts[name] = ["not sent"]
                continue
            rid = response_id(row)
            if rid is None:
                p.append(f"HTTP {row['status']}")
            rs = recs(row["record_tag"])
            preempted += smoke.preempted_count(rs)
            p += smoke.record_problems(rs, record_tag=row["record_tag"], arm=arm)
            if rs:
                p += smoke.replay_problems(rs, accepts(automata[task]))
            if rid is not None:
                p += smoke.release_problems(releases, rid)
                p += smoke.commit_frame_problems(smoke.frames_for(frames, rid), rs)
                tp, inc = smoke.token_ids_problems(token_ids_of(row), rs)
                token_problems += [f"{name}: {x}" for x in tp]
                included[name] = inc
            verdicts[name] = p
    info.append(f"preempted records (T-0067 S10, reported): {preempted} of 8 c4 requests")

    # ---- c5
    g, x = run.rows.get("c5:G"), run.rows.get("c5:X")
    p5: list[str] = []
    if g is None or x is None:
        p5.append("G or X was not sent")
    else:
        g_recs = recs(g["record_tag"])
        p5 += [
            f"G: {s}"
            for s in smoke.record_problems(g_recs, record_tag=g["record_tag"], arm="perstep_mar")
        ]
        if g_recs:
            p5 += [f"G: {s}" for s in smoke.replay_problems(g_recs, accepts(automata["G"]))]
        if response_id(g) is None:
            p5.append(f"G: HTTP {g['status']}")
        if x["status"] != 200:
            p5.append(f"X: HTTP {x['status']}")
        p5 += [
            f"X: {s}"
            for s in smoke.loud_problems(
                content_of(x), recs(x["record_tag"]), predicted=str(x.get("predicted"))
            )
        ]
        gid, xid = response_id(g), response_id(x)
        if gid and xid:
            gi = [
                i for i, f in enumerate(frames) if smoke.belongs(str(f.get("request_id", "")), gid)
            ]
            xi = [
                i for i, f in enumerate(frames) if smoke.belongs(str(f.get("request_id", "")), xid)
            ]
            gd = next((i for i in gi if frames[i].get("phase") == "denoise"), None)
            if not xi or gd is None or not gd < xi[0] < gi[-1]:
                p5.append(
                    f"X did not overlap G's denoising (G frames {gi[:1]}..{gi[-1:]}, first denoise {gd}, "
                    f"X first {xi[:1]})"
                )
    nxt, health = run.rows.get("c5:next"), run.rows.get("c5:health")
    p5 += smoke.serving_problems(
        next_status=int(nxt["status"]) if nxt else 0,
        health_status=int(health["status"]) if health else 0,
    )
    verdicts["c5 loud, non-fatal"] = p5

    # ---- c6
    for arm in C6_ARMS:
        row = run.rows.get(f"c6:{arm}")
        if row is None:
            verdicts[f"c6:{arm}"] = ["not sent"]
            continue
        rs = recs(row["record_tag"])
        p6: list[str] = [] if row["status"] == 200 else [f"HTTP {row['status']}"]
        p6 += smoke.multicanvas_problems(
            rs,
            literal_ids=row["literal_ids"],
            completion_tokens=completion_tokens(row) or 0,
            max_tokens=C6_MAX_TOKENS,
        )
        if rs:
            p6 += smoke.replay_problems(rs, accepts(automata["literal"]))
        verdicts[f"c6:{arm}"] = p6

    # ---- c7
    cases = {}
    for name in ("near", "over"):
        row = run.rows.get(f"c7:{name}") or {}
        xa = row.get("xargs") or {}
        cases[name] = {
            "length": len(xa.get("astra_cd_automaton", "")),
            "status": row.get("status"),
            "records": recs(row.get("record_tag")),
            "predicted": row.get("predicted"),
        }
    verdicts["c7 transport at the cap"] = smoke.transport_problems(
        cases["near"], cases["over"], cap=MAX_AUTOMATON_CHARS
    )

    # ---- c8
    p8 = smoke.boot_problems(engine_log, REFERENCE.read_text())
    quiet = run.rows.get("c8:quiet")
    qid = response_id(quiet) if quiet else None
    if qid is None:
        p8.append("c8:quiet failed")
    else:
        if f"canvas tap release: req_id={qid}-" not in window:
            p8.append("the window does not hold the quiet request's release line")
        p8 += smoke.quiet_window_problems(window)
        qf = smoke.frames_for(frames, qid)
        p8 += smoke.frame_problems(qf)
        p8 += smoke.commit_count_problems(qf, completion_tokens(quiet) or 0)
    verdicts["c8 boot, vllm.sh, quiet window"] = p8

    # ---- c9
    rows = [run.rows.get(f"c9:header:{k}") or {} for k in range(2 * HEADER_PER_TASK)]
    hdr = smoke.header_report(
        [
            {"task": r.get("task"), "dataset_id": r.get("dataset_id"), "token_ids": token_ids_of(r)}
            for r in rows
        ]
    )
    info.append(
        f"c9 header-prefix rate (reported, not gated): {hdr['header_prefix']}/{hdr['n']} = "
        f"{hdr['rate']:.2f}; first 8 ids: {hdr['first8']}"
    )
    info.append(f"c9 stop id included in constrained token_ids: {included}")
    verdicts["c9 token ids"] = list(hdr["problems"]) + token_problems
    walls = [
        f"{r['name']} {r.get('wall_ms_not_a_measurement')} ms"
        for r in run.rows.values()
        if r.get("wall_ms_not_a_measurement") is not None and not r["name"].startswith("warmup:")
    ]
    info.append("wall times (logged, not a measurement): " + ", ".join(walls))


if __name__ == "__main__":
    raise SystemExit(main())
