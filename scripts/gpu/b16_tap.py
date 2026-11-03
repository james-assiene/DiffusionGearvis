"""T-0011 tier 3 -- B16, the canvas tap's cost, and the first real trajectory. `/launch` ONLY.

    uv run python scripts/gpu/b16_tap.py [--boot-log logs/vllm.log] [--profile lean]

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine.
It never boots one. Before it is run: `astra up --profile lean` with the plugin wheel built from
this commit (STT and conductor down -- engine only), a clean tree, a /launch token.

What it does (docs/tasks/T-0011.md criteria 13-14):

1. Refuses: a dirty tree; `git diff bd8001b..HEAD -- services/vllm.sh` non-empty (then take a fresh
   tap-less reference boot first); no engine on 127.0.0.1:8001; an engine NOT on the diffusion path
   (a `temperature` probe must be rejected 400).
2. Extracts the engine config from the boot log with `local_astra.bench.b16_tap.boot_config_lines`,
   writes it to `logs/t0011-tier3-<sha>/boot-reference.txt` (T-0069 criterion 4's reference) and
   compares it with the set from `logs/vllm-coldboot-bd8001b.log` (plugin on, no tap; the committed
   excerpt `tests/fixtures/engine/vllm-coldboot-bd8001b.excerpt.log` if the log is gone).
3. Subscribes to the tap (`tcp://127.0.0.1:8002`, SUB, topic `astra.canvas.step`).
4. Builds a fixed text-only prompt of ~512 tokens (sized with the engine's own `/tokenize`), thinking
   on, `max_tokens` 768, no seed (a `seed` is a 400 on this path).
5. Three interleaved arms at concurrency 1 (R17): A = `vllm_xargs={"astra_tap": false}`,
   B = `vllm_xargs={"astra_tap": true}` (identical xargs, only the tap differs), D = no xargs (the
   default path). 3 warm-up rounds (discarded), then 30 rounds; the arm order rotates per round
   (round r runs `ARMS[r % 3:] + ARMS[:r % 3]`: ABD, BDA, DAB, ...), so each arm follows each other
   arm 10 times and A_r/B_r are paired by round index. The xargs-leak check:
   A must publish no frames, B and D must publish frames; otherwise the run fails and the report
   says to fall back to two boots toggling `ASTRA_TAP`.
6. B16 gate = A vs B through `local_astra.bench.b16_tap.evaluate` (control from A, as before).
   D vs B is one informational line (medians, p95, delta), not gated. Per tap-on request: frames,
   frame rate, commit frames against committed canvases (`ceil(completion_tokens / 256)`) with every
   mismatch listed, canvas_idx gaps.
7. Trajectory (criterion 14, R17): after the rounds, up to 3 tap-on requests with a long-answer
   prompt (thinking on, `max_tokens` 1280 = 5 canvases, checked against `max_model_len` with the
   prompt); the first that closes >= 3 canvases is written to
   `artifacts/canvas/<ts>-<regime>.msgpack` (every frame and the regime line). Canvases per attempt
   are printed. These requests are not part of B16, but their frames are validated and their
   release lines checked like the rest. Every frame of the run is validated as `AstraCanvasStep`
   with `accepted`/`entropy` absent and the per-phase step range (R15: denoise
   `0 <= step <= max_steps`, commit `1 <= step <= max_steps`).
8. Every tap-on request must have a tap release line (R12, from the last boot of the engine log,
   R16) and every release line `commit_overrun == 0`. The report is written before the exit code;
   the exit code is the verdict.

No byte-identity claim across arms: the sampler noise is unseeded.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import msgpack

from local_astra.bench.b16_tap import (
    boot_config_lines,
    evaluate,
    format_report,
    last_boot,
    parse_tap_release_lines,
)
from local_astra.contracts.protocol import AstraCanvasStep, AstraMessageAdapter

REPO = Path(__file__).resolve().parents[2]
API = "http://127.0.0.1:8001"
TAP = "tcp://127.0.0.1:8002"
TOPIC = b"astra.canvas.step"
REFERENCE_COMMIT = "bd8001b"
REFERENCE_LOG = REPO / "logs" / "vllm-coldboot-bd8001b.log"
REFERENCE_EXCERPT = REPO / "tests" / "fixtures" / "engine" / "vllm-coldboot-bd8001b.excerpt.log"
CANVAS = 256
STEPS = 48
PROMPT_TOKENS = 512
MAX_TOKENS = 768
ROUNDS = 30
WARMUP = 3
MAX_MODEL_LEN = 16384
#: R17: A and B carry identical xargs and differ only in the tap; D is the default (no xargs)
ARMS: tuple[tuple[str, dict[str, bool] | None], ...] = (
    ("A", {"astra_tap": False}),
    ("B", {"astra_tap": True}),
    ("D", None),
)
ROTATION = (
    "arm order rotated per round (round r runs ARMS[r % 3:] + ARMS[:r % 3]: ABD, BDA, DAB, ...; "
    "each arm follows each other arm 10 times over the measured rounds; A_r and B_r paired by round)"
)
TAP_ON_ARMS = ("B", "D", "T")  # T: the trajectory requests
TRAJ_MAX_TOKENS = 1280  # 5 canvases at CL 256
TRAJ_ATTEMPTS = 3
TRAJ_MIN_CANVASES = 3
TRAJ_PROMPT = (
    "Write a long, detailed answer of at least eight full paragraphs: explain how a lighthouse "
    "keeper in the nineteenth century would have kept the light burning through a winter storm -- "
    "the fuel, the lens, the clockwork, the daily routine, the dangers, the log book, the relief "
    "boats, and what changed when the lights were automated. Do not summarise; write it all out."
)
THINKING = {"enable_thinking": True}  # chat_template_kwargs: <|think|> on, as the conductor sends
GAP_S = 1.0  # quiet time after each request so its release flush lands in its own window

SENTENCE = (
    "A lighthouse keeper writes down the weather every evening, the colour of the sea, the ships "
    "that passed, and anything unusual the gulls did. "
)


def die(code: int, msg: str) -> None:
    print(f"b16_tap: {msg}", file=sys.stderr)
    raise SystemExit(code)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


# --------------------------------------------------------------------------- preflight


def preflight(client: httpx.Client) -> tuple[str, str]:
    dirty = git("status", "--porcelain")
    if dirty:
        die(2, f"refusing: the tree is dirty:\n{dirty}")
    drift = git("diff", f"{REFERENCE_COMMIT}..HEAD", "--", "services/vllm.sh")
    if drift:
        die(
            2,
            f"refusing: services/vllm.sh changed since {REFERENCE_COMMIT}, so the bd8001b boot is no "
            "longer the plugin-on reference. Take a fresh tap-less reference boot first.",
        )
    try:
        model = client.get(f"{API}/v1/models").json()["data"][0]["id"]
    except Exception as exc:
        die(2, f"no engine on {API} ({exc}); boot it with `astra up --profile lean` first")
    probe = client.post(
        f"{API}/v1/chat/completions",
        json={
            "model": model,
            "max_tokens": 8,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    if probe.status_code != 400:
        die(
            2,
            f"temperature returned {probe.status_code}, not 400: the engine is not on the diffusion path",
        )
    return model, git("rev-parse", "--short", "HEAD")


def boot_reference(boot_log: Path, out_dir: Path) -> tuple[list[str], list[str]]:
    if not boot_log.exists():
        die(2, f"no boot log at {boot_log}; pass --boot-log")
    # R16: logs/vllm.log is append-mode (every boot); read only the last one
    got = boot_config_lines(last_boot(boot_log.read_text(errors="replace")))
    ref_path = REFERENCE_LOG if REFERENCE_LOG.exists() else REFERENCE_EXCERPT
    ref = boot_config_lines(ref_path.read_text(errors="replace"))
    (out_dir / "boot-reference.txt").write_text(
        f"# boot_config_lines({boot_log}) at {git('rev-parse', 'HEAD')}\n"
        f"# compared with {ref_path.relative_to(REPO)} ({REFERENCE_COMMIT}, plugin on, no tap)\n"
        + "\n".join(got)
        + "\n"
    )
    return got, ref


# --------------------------------------------------------------------------- the tap subscriber


class Subscriber:
    def __init__(self) -> None:
        import zmq

        self._zmq = zmq
        self.frames: list[tuple[float, dict[str, Any]]] = []
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
                self.frames.append((time.monotonic(), msgpack.unpackb(payload, raw=False)))

    def between(self, t0: float, t1: float) -> list[dict[str, Any]]:
        return [f for t, f in list(self.frames) if t0 <= t < t1]

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2.0)
        self._sock.close(linger=0)


# --------------------------------------------------------------------------- requests


def build_prompt(client: httpx.Client, model: str) -> tuple[list[dict[str, str]], int]:
    def count(k: int) -> tuple[list[dict[str, str]], int]:
        messages = [
            {
                "role": "user",
                "content": "Summarise this log in three sentences.\n\n" + SENTENCE * k,
            }
        ]
        return messages, count_tokens(client, model, messages)

    lo, hi = 1, 64
    best = count(lo)
    while lo <= hi:
        mid = (lo + hi) // 2
        got = count(mid)
        if abs(got[1] - PROMPT_TOKENS) < abs(best[1] - PROMPT_TOKENS):
            best = got
        if got[1] < PROMPT_TOKENS:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def count_tokens(client: httpx.Client, model: str, messages: list[dict[str, str]]) -> int:
    r = client.post(
        f"{API}/tokenize",
        json={"model": model, "messages": messages, "chat_template_kwargs": THINKING},
    )
    r.raise_for_status()
    return int(r.json()["count"])


def one_request(
    client: httpx.Client,
    model: str,
    messages: list[dict[str, str]],
    *,
    arm: str,
    xargs: dict[str, bool] | None,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "chat_template_kwargs": THINKING,
    }
    if xargs is not None:
        body["vllm_xargs"] = dict(xargs)
    t0 = time.monotonic()
    r = client.post(f"{API}/v1/chat/completions", json=body)
    t1 = time.monotonic()
    r.raise_for_status()
    usage = r.json()["usage"]
    time.sleep(GAP_S)
    return {
        "arm": arm,
        "t0": t0,
        "t1": t1,
        "t_end": time.monotonic(),
        "ms": 1000.0 * (t1 - t0),
        "id": r.json().get("id"),
        "completion_tokens": int(usage["completion_tokens"]),
        "prompt_tokens": int(usage["prompt_tokens"]),
    }


# --------------------------------------------------------------------------- checks


def frame_problems(frames: list[dict[str, Any]]) -> list[str]:
    bad: list[str] = []
    for f in frames:
        try:
            m = AstraMessageAdapter.validate_python(f)
        except Exception as exc:
            bad.append(f"invalid frame {f.get('request_id')}#{f.get('canvas_idx')}: {exc}")
            continue
        if not isinstance(m, AstraCanvasStep):
            bad.append(f"not a canvas step: {f.get('type')}")
        elif m.accepted is not None or m.entropy is not None:
            bad.append(f"accepted/entropy filled on {m.request_id}#{m.canvas_idx}")
        elif m.phase == "denoise" and not 0 <= m.step <= m.max_steps:  # R15: the cap included
            bad.append(f"denoise step {m.step} outside 0..{m.max_steps}")
        elif m.phase == "commit" and not 0 <= m.step <= m.max_steps:  # R14 as amended by O6
            bad.append(f"commit step {m.step} outside 0..{m.max_steps}")
    return bad


def per_request(req: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, Any]:
    ids = sorted({f["request_id"] for f in frames})
    commits = [f for f in frames if f["phase"] == "commit"]
    idx = [f["canvas_idx"] for f in commits]
    expected = math.ceil(req["completion_tokens"] / CANVAS)
    dur = max(req["t1"] - req["t0"], 1e-9)
    return {
        **{k: req[k] for k in ("arm", "ms", "id", "completion_tokens", "prompt_tokens")},
        "engine_request_ids": ids,
        "frames": len(frames),
        "frame_hz": len(frames) / dur,
        "commit_frames": len(commits),
        "committed_canvases": expected,
        "commit_idx": idx,
        "idx_gaps": [b for a, b in itertools.pairwise(idx) if b != a + 1],
    }


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boot-log", type=Path, default=REPO / "logs" / "vllm.log")
    ap.add_argument("--profile", default="lean")
    args = ap.parse_args()

    with httpx.Client(timeout=900.0) as client:
        model, sha = preflight(client)
        out = REPO / "logs" / f"t0011-tier3-{sha}"
        out.mkdir(parents=True, exist_ok=True)
        got, ref = boot_reference(args.boot_log, out)
        boot_same = got == ref

        messages, prompt_len = build_prompt(client, model)
        traj_messages = [{"role": "user", "content": TRAJ_PROMPT}]
        traj_prompt_len = count_tokens(client, model, traj_messages)
        if traj_prompt_len + TRAJ_MAX_TOKENS > MAX_MODEL_LEN:
            die(2, f"trajectory prompt {traj_prompt_len} + {TRAJ_MAX_TOKENS} > {MAX_MODEL_LEN}")

        sub = Subscriber()
        time.sleep(0.5)  # SUB slow-joiner
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        load = os.getloadavg()
        regime = (
            f"profile {args.profile} (engine only), commit {sha} clean, {model} rev 2ea8372, "
            f"vllm v0.29.0, driver {driver}, canvas {CANVAS}, steps {STEPS}, TRITON_ATTN, bf16 KV "
            f"4 GiB, max_model_len {MAX_MODEL_LEN}, max_num_seqs 3, plugin+tap on, hz 30, prompt "
            f"{prompt_len} tokens text-only, thinking on, max_tokens {MAX_TOKENS}, concurrency 1, "
            f"arms A=xargs astra_tap false / B=xargs astra_tap true / D=no xargs, {WARMUP} warm-up "
            f"rounds + {ROUNDS} interleaved rounds (n={ROUNDS} per arm), {ROTATION}, trajectory: up to "
            f"{TRAJ_ATTEMPTS} tap-on requests, prompt {traj_prompt_len} tokens, max_tokens "
            f"{TRAJ_MAX_TOKENS}; no seed, host load {load[0]:.2f}/{load[1]:.2f}/{load[2]:.2f}, "
            f"python {platform.python_version()}"
        )
        print(regime)

        reqs: list[dict[str, Any]] = []
        for rnd in range(WARMUP + ROUNDS):
            # rotate the arm order every round (warm-ups included): ABD, BDA, DAB, ... so over the
            # 30 measured rounds each arm follows each other arm 10 times, and a tap after-effect
            # that outlasts the gap cannot always land on the tap-off request
            for arm, xargs in ARMS[rnd % 3 :] + ARMS[: rnd % 3]:
                r = one_request(client, model, messages, arm=arm, xargs=xargs)
                reqs.append({**r, "warmup": rnd < WARMUP, "round": rnd})

        # criterion 14 / R17: dedicated long-answer requests, not part of B16
        traj_req: dict[str, Any] | None = None
        attempts: list[str] = []
        for attempt in range(1, TRAJ_ATTEMPTS + 1):
            r = one_request(
                client,
                model,
                traj_messages,
                arm="T",
                xargs={"astra_tap": True},
                max_tokens=TRAJ_MAX_TOKENS,
            )
            r["warmup"] = False
            r["round"] = WARMUP + ROUNDS + attempt - 1  # after the B16 rounds; not part of B16
            reqs.append(r)
            canvases = math.ceil(r["completion_tokens"] / CANVAS)
            line = (
                f"trajectory attempt {attempt}: {canvases} canvases "
                f"(completion_tokens {r['completion_tokens']})"
            )
            print(line)
            attempts.append(line)
            if canvases >= TRAJ_MIN_CANVASES:
                traj_req = r
                break
        time.sleep(1.0)
        sub.close()

    rows = [
        per_request(r, sub.between(r["t0"], r["t_end"]))
        | {"warmup": r["warmup"], "round": r["round"]}
        for r in reqs
    ]

    # the xargs-leak check: A publishes nothing; B and D (tap on) publish
    leaks = [
        f"{r['id']}: tap-off (A) request published {r['frames']} frames"
        for r in rows
        if r["arm"] == "A" and r["frames"]
    ]
    leaks += [
        f"{r['id']}: tap-on ({r['arm']}) request published no frames"
        for r in rows
        if r["arm"] in TAP_ON_ARMS and not r["frames"]
    ]

    measured = [r for r in rows if not r["warmup"]]
    # sorted by round index, so A_r and B_r (and D_r) stay paired for evaluate whatever the order
    # inside a round was
    by_arm = {
        arm: sorted((r for r in measured if r["arm"] == arm), key=lambda r: r["round"])
        for arm in ("A", "B", "D", "T")
    }
    a = [r["ms"] for r in by_arm["A"]]
    b = [r["ms"] for r in by_arm["B"]]
    d = [r["ms"] for r in by_arm["D"]]
    observed_hz = max(r["frame_hz"] for r in by_arm["B"])
    result = evaluate(a, b, observed_hz=observed_hz)
    # D vs B: informational only (the xargs path against the default path), never gated
    info = evaluate(d, b)
    info_line = (
        f"D (no xargs) vs B (xargs astra_tap true), informational, not gated: median D "
        f"{info.median_a_ms:.1f} ms, B {info.median_b_ms:.1f} ms; p95 D {info.p95_a_ms:.1f} ms, "
        f"B {info.p95_b_ms:.1f} ms; delta (B vs D) {info.delta_pct:+.2f} %"
    )

    tap_on = [r for r in rows if r["arm"] in TAP_ON_ARMS]
    all_frames = [f for _, f in sub.frames]
    problems = frame_problems(all_frames)
    mismatches = [
        f"{r['arm']} {r['id']}: {r['commit_frames']} commit frames vs {r['committed_canvases']} "
        f"committed canvases (completion_tokens {r['completion_tokens']}), canvas_idx gaps "
        f"{r['idx_gaps']}"
        for r in tap_on
        if r["commit_frames"] != r["committed_canvases"] or r["idx_gaps"]
    ]

    # R12: the tap's per-release counter lines from the last boot of the engine log (R16); every
    # tap-on request, trajectory attempts included, must have one, and every line overrun 0
    releases = parse_tap_release_lines(last_boot(args.boot_log.read_text(errors="replace")))
    released = {str(x["req_id"]) for x in releases}
    overrun_lines = [x for x in releases if x["commit_overrun"] != 0]
    unreleased = [
        f"{r['arm']} {r['id']} (engine ids {r['engine_request_ids']})"
        for r in tap_on
        if not set(r["engine_request_ids"]) & released
    ]

    artifact = None
    if traj_req is not None:
        traj_row = next(r for r in rows if r["id"] == traj_req["id"])
        frames = sub.between(traj_req["t0"], traj_req["t_end"])
        ts = time.strftime("%Y%m%dT%H%M%S")
        artifact = REPO / "artifacts" / "canvas" / f"{ts}-lean-cl{CANVAS}-s{STEPS}-{sha}.msgpack"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(
            msgpack.packb(
                {"regime": regime, "request": traj_row, "frames": frames}, use_bin_type=True
            )
        )

    tap_frames = [r["frames"] for r in by_arm["B"] + by_arm["D"]]
    report = "\n".join(
        [
            format_report(result, regime=regime),
            f"arm order: {ROTATION}",
            info_line,
            f"frames per tap-on request (B, D): median {statistics.median(tap_frames)}, "
            f"max rate {max(r['frame_hz'] for r in tap_on):.1f} Hz",
            f"xargs-leak check: {len(leaks)} problems"
            + "".join(f"\n  {x}" for x in leaks[:20])
            + ("\n  -> fall back to two boots toggling ASTRA_TAP (criterion 13)" if leaks else ""),
            f"invalid frames: {len(problems)}" + "".join(f"\n  {p}" for p in problems[:20]),
            f"commit mismatches: {len(mismatches)}" + "".join(f"\n  {m}" for m in mismatches),
            f"tap release lines: {len(releases)}, with commit_overrun != 0: {len(overrun_lines)}"
            + "".join(f"\n  {x}" for x in overrun_lines[:20])
            + "".join(f"\n  no release line for {u}" for u in unreleased),
            f"boot config equals the {REFERENCE_COMMIT} reference: {boot_same}"
            + "".join(f"\n  only now: {x}" for x in sorted(set(got) - set(ref)))
            + "".join(f"\n  only in reference: {x}" for x in sorted(set(ref) - set(got))),
            f"boot reference: {out / 'boot-reference.txt'}",
            *attempts,
            f"trajectory: {artifact if artifact else f'NONE (no attempt closed >= {TRAJ_MIN_CANVASES} canvases)'}",
        ]
    )
    print(report)
    (out / "report.txt").write_text(report + "\n")
    (out / "requests.json").write_text(json.dumps(rows, indent=1))
    ok = (
        result.verdict == "pass"
        and not leaks
        and not problems
        and not mismatches
        and bool(releases)
        and not overrun_lines
        and not unreleased
        and boot_same
        and artifact is not None
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
