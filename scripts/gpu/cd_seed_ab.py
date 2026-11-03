"""T-0081 tier 2 -- BFCL dev seeded repair on the engine (criteria 8-9). `/launch` ONLY.

    uv run --group cd python scripts/gpu/cd_seed_ab.py [--boot-log logs/vllm.log] [--limit N]

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one; ``scripts/locked/t0081_seed.sh`` does (engine only, ``services/vllm.sh``, the plugin wheel built from this
commit).

Refusals (exit 2, before any measured request): a dirty tree (paths named); ``services/vllm.sh`` changed since
806da12 (the T-0071 shared regime); no engine log; the evidence directory exists; ``docs/cd/calibration-bfcl.json``
not a frozen BFCL ``perstep_mar`` calibration (``calibration.load_frozen`` raises); no engine on 127.0.0.1:8001; an
engine not on the diffusion path (a ``temperature`` probe must be a 400).

Per record of the BFCL dev cut (130-229, n = 100), concurrency 1, thinking off, ``max_tokens`` 256:

1. **Draft** (always first): one unconstrained request; its generated ids (``return_token_ids``), cut to the canvas
   length, are the draft. Its own score splits the report (draft schema-valid or not).
2. **Nine constrained arms**, in an order rotated by the record's index (paired by record): ``perstep_mar`` at
   T-0079's frozen knobs (``docs/cd/calibration-bfcl.json``; its sha256 is in the regime line) from scratch and seeded at (noise, start step) in
   {0, 0.25, 0.5} x {0, 24}; ``end_projection`` from scratch and seeded at (0.25, 0). Grammar
   ``channel_header=False, fence=True``, JSON whitespace (the drafts' own format; T-0081 alignment note), compiled
   once per record. Every request carries a unique ``record_tag``; its ``ASTRA_CD`` records are joined from the
   engine log (with up to two tiny unconstrained flush requests, as the runner's ruling E9), the detokenised
   committed ids must match the served text on the answer region (``replay.join_matches`` against
   ``served_text(reasoning, content)``, as the runner does; otherwise ``join_error``), the committed ids are
   replayed through the driver's own automaton (CS never comes from the plugin) and scored with the runner's
   scorer.

Executed denoise steps per canvas: the record's ``steps`` for every canvas, minus the start step on canvas 0 (SD9:
``steps`` = k0 + executed on the seeded first canvas; later canvases start from stock's re-init at step 0).

Copy (``is_copy``): the committed ids equal the draft up to and including the draft's first stop id (1, 50, 106),
since the records carry the whole canvas (stop id, then padding) while ``token_ids`` end at the stop. Knobs
(``load_knobs``): ``calibration.load_frozen(path, task="bfcl", arm="perstep_mar")``; any refusal exits 2.

Stop rule (criterion 8): a loud CD error, a failed join, an HTTP error or CS = 0 on any constrained row stops the
run; the report is written with the PROBLEM lines and the exit code is 1. The first seeded request doubles as the
task's ~1.4 KB ``vllm_xargs`` check (its draft key length is printed; a refusal would stop the run).

Evidence: ``logs/t0081-seed-<sha>/`` -- ``rows.jsonl`` (one line per request), ``report.txt`` (regime, per-arm
table split by draft validity, copy rate, repair rate on invalid drafts, the criterion-9 reading with exact
``Fraction`` arithmetic and the anchoring McNemar), ``engine.log`` (the engine log from the driver's start).
``report.txt`` ends with ``run: COMPLETE`` only when every record ran every arm with no problem.

Exit code: 0 = run complete (whatever the reading says -- criterion 9 is reported, not gated); 1 = the run
stopped (a problem above); 2 = refused.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
REFERENCE_COMMIT = "806da12"
API = "http://127.0.0.1:8001"
V1 = API + "/v1"
CANVAS = 256
MAX_STEPS = 48
#: T-0079's frozen perstep_mar knobs (criterion 8), loaded with ``calibration.load_frozen`` in ``main``
CALIBRATION = REPO / "docs" / "cd" / "calibration-bfcl.json"
RECORD_WAIT_S = 5.0
FLUSH_WAIT_S = 2.0
MAX_FLUSHES = 2
TOLERANCE = Fraction(1, 20)
STEP_RATIO = Fraction(3, 5)
WALL_RATIO = Fraction(4, 5)


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    mode: str
    noise: float | None  # None: from scratch
    start: int | None

    @property
    def seeded(self) -> bool:
        return self.noise is not None


SCRATCH_MAR = Arm("mar_scratch", "perstep_mar", None, None)
SCRATCH_EP = Arm("ep_scratch", "end_projection", None, None)
SEEDED_MAR = tuple(
    Arm(f"mar_n{noise:g}_s{start}", "perstep_mar", noise, start)
    for noise in (0.0, 0.25, 0.5)
    for start in (0, 24)
)
SEEDED_EP = Arm("ep_n0.25_s0", "end_projection", 0.25, 0)
ARMS: tuple[Arm, ...] = (SCRATCH_MAR, *SEEDED_MAR, SCRATCH_EP, SEEDED_EP)


def rotation(index: int) -> list[Arm]:
    """The arm order of record ``index``: ``ARMS`` rotated by ``index`` (every arm in every slot every 9 records)."""
    k = index % len(ARMS)
    return list(ARMS[k:] + ARMS[:k])


# --------------------------------------------------------------------------- small helpers


def die(code: int, msg: str) -> None:
    print(f"cd_seed_ab: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def post(body: Mapping[str, Any], timeout_s: float = 1800.0) -> tuple[int, Any]:
    req = urllib.request.Request(
        V1 + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status), json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return int(exc.code), None
    except (OSError, ValueError):
        return 0, None


def nearest_rank(xs: Sequence[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = max(1, -(-int(q * 100) * len(s) // 100))  # ceil(q * n), at least 1
    return float(s[min(k, len(s)) - 1])


#: ``local_astra.eval.cd.replay.STOP_IDS`` (EOS, ``<turn|>``, ``<|tool_response>``)
STOPS = frozenset({1, 50, 106})


def is_copy(committed: Sequence[int], draft: Sequence[int]) -> bool:
    """The commit copies the draft: equal up to and including the draft's first stop id (all of a draft without
    one). The records carry the whole canvas (stop id, then padding); ``token_ids`` end at the stop."""
    head: list[int] = []
    for t in draft:
        head.append(int(t))
        if int(t) in STOPS:
            break
    return [int(t) for t in committed[: len(head)]] == head


def load_knobs(path: Path | str) -> dict[str, float]:
    """T-0079's frozen BFCL ``perstep_mar`` knobs; any other file (arm, task, edited, missing) exits 2."""
    from local_astra.eval.cd import calibration as _calibration

    try:
        knobs, _sha = _calibration.load_frozen(path, task="bfcl", arm="perstep_mar")
    except ValueError as exc:
        die(2, f"refusing: perstep_mar knobs: {exc}")
    return {
        "entropy_bound": float(knobs["entropy_bound"]),
        "confidence_threshold": float(knobs["confidence_threshold"]),
    }


def message(payload: Any) -> tuple[str | None, str]:
    """``(reasoning, content)`` of a chat completion, as the runner reads them."""
    msg = payload["choices"][0].get("message") or {}
    reasoning = msg.get("reasoning")
    if reasoning is None:
        reasoning = msg.get("reasoning_content")
    return reasoning, str(msg.get("content") or "")


def b64_ids(ids: Sequence[int]) -> str:
    import base64

    import numpy as np

    return base64.b64encode(np.asarray(list(ids), dtype="<i4").tobytes()).decode("ascii")


# --------------------------------------------------------------------------- the run


class Run:
    def __init__(self, out: Path, engine_log: Path, knobs: Mapping[str, float]) -> None:
        from local_astra.eval.cd import records as _records

        self.out = out
        self.knobs = dict(knobs)
        self.run_id = uuid.uuid4().hex[:10]
        self.log = _records.EngineLog(engine_log)
        self.engine_log = engine_log
        self.model = ""
        self.rows: list[dict[str, Any]] = []
        self.problems: list[str] = []
        self.flushes = 0
        self.xargs_check: str | None = None
        self._fh = (out / "rows.jsonl").open("w")

    def write(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def flush_request(self) -> None:
        from local_astra.eval.cd import prompts as _prompts

        body = _prompts.request_body(self.model, "Reply with OK.")
        body["max_tokens"] = 1
        body.pop("return_token_ids", None)
        post(body)
        self.flushes += 1

    def await_records(self, tag: str) -> Any:
        from local_astra.eval.cd import prompts as _prompts
        from local_astra.eval.cd import records as _records

        joined = self.log.wait(tag, max_tokens=_prompts.MAX_TOKENS, timeout_s=RECORD_WAIT_S)
        sent = 0
        while sent < MAX_FLUSHES and (
            joined is None or not _records.is_complete(joined.records, _prompts.MAX_TOKENS)
        ):
            self.flush_request()
            sent += 1
            joined = self.log.wait(tag, max_tokens=_prompts.MAX_TOKENS, timeout_s=FLUSH_WAIT_S)
        return joined

    def draft(self, index: int, rec: Any) -> dict[str, Any]:
        from local_astra.eval.cd import cuts as _cuts
        from local_astra.eval.cd import prompts as _prompts
        from local_astra.eval.cd import replay as _replay

        body = _prompts.request_body(self.model, _prompts.build_prompt("bfcl", rec))
        t0 = time.perf_counter()
        status, payload = post(body)
        wall = (time.perf_counter() - t0) * 1000.0
        row: dict[str, Any] = {"id": rec.id, "index": index, "arm": "draft", "wall_ms": wall}
        ids = None
        if status == 200 and payload is not None:
            got = payload["choices"][0].get("token_ids")
            ids = [int(t) for t in got] if isinstance(got, list) else None
        if not ids:
            row["error"] = (
                f"no draft (HTTP {status}, token_ids {'missing' if ids is None else 'empty'})"
            )
            self.problems.append(f"{rec.id}: {row['error']}")
            self.write(row)
            return row
        text = _replay.detokenize(ids)
        s = _cuts.score("bfcl", rec, text)
        row.update(
            draft_ids=ids[:CANVAS],
            draft_len=min(len(ids), CANVAS),
            text=text,
            schema_valid=bool(s["schema_valid"]),
            exact_call=bool(s["exact_call"]),
            error=None,
        )
        self.write(row)
        return row

    def constrained(
        self, index: int, rec: Any, arm: Arm, automaton: Any, payload: str, draft: Sequence[int]
    ) -> dict[str, Any]:
        from local_astra.contracts.cd import CdRequest
        from local_astra.eval.cd import cuts as _cuts
        from local_astra.eval.cd import prompts as _prompts
        from local_astra.eval.cd import records as _records
        from local_astra.eval.cd import replay as _replay

        tag = f"t0081.{self.run_id}.{index}.{arm.name}"
        fields: dict[str, Any] = {
            "mode": arm.mode,
            "self_conditioning": "model",
            "automaton": payload,
            "automaton_sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
            "record_tag": tag,
        }
        if arm.mode == "perstep_mar":
            fields.update(self.knobs)
        if arm.seeded:
            fields.update(draft=b64_ids(draft), draft_noise=arm.noise, draft_start_step=arm.start)
        xargs = CdRequest(**fields).to_xargs()
        if arm.seeded and self.xargs_check is None:
            self.xargs_check = (
                f"first seeded request {tag}: astra_cd_draft is {len(xargs['astra_cd_draft'])} "
                f"characters ({len(draft)} ids)"
            )
            print(self.xargs_check, flush=True)
        body = _prompts.request_body(self.model, _prompts.build_prompt("bfcl", rec), xargs)
        t0 = time.perf_counter()
        status, resp = post(body)
        wall = (time.perf_counter() - t0) * 1000.0
        row: dict[str, Any] = {
            "id": rec.id,
            "index": index,
            "arm": arm.name,
            "mode": arm.mode,
            "noise": arm.noise,
            "start": arm.start,
            "tag": tag,
            "wall_ms": wall,
            "error": None,
        }
        if status != 200 or resp is None:
            row["error"] = f"http {status}"
            return row
        try:
            reasoning, content = message(resp)
        except (KeyError, IndexError, TypeError, ValueError):
            row["error"] = "http 200 without a message"
            return row
        joined = self.await_records(tag)
        if joined is None or not joined.valid:
            row["error"] = "join_error"
            return row
        if joined.error is not None:
            row["error"] = f"loud: {joined.error}"
            return row
        if not _records.is_complete(joined.records, _prompts.MAX_TOKENS):
            row["error"] = "join_error (incomplete)"
            return row
        committed = list(joined.committed_ids)
        steps = list(joined.steps)
        k0 = arm.start or 0
        executed = [s - (k0 if i == 0 else 0) for i, s in enumerate(steps)]
        text = _replay.detokenize(committed)
        if not _replay.join_matches(text, _replay.served_text(reasoning, content)):
            row["error"] = "join_error (committed ids do not match the served text)"
            return row
        s = _cuts.score("bfcl", rec, text)
        row.update(
            committed_ids=committed,
            steps=steps,
            executed=executed,
            cs=bool(_replay.accepts(automaton, committed)),
            schema_valid=bool(s["schema_valid"]),
            exact_call=bool(s["exact_call"]),
            copy=is_copy(committed, draft),
            text=text,
        )
        if min(executed, default=0) < 0:
            row["error"] = f"executed steps {executed} < 0 (record steps below the start step)"
        return row

    def record(self, index: int, rec: Any) -> bool:
        """One record: the draft, then the nine arms in rotated order. False: the run must stop."""
        from local_astra.decoding import artifact as _artifact
        from local_astra.decoding import grammars as _grammars
        from local_astra.decoding import report as _report

        d = self.draft(index, rec)
        if d.get("error"):
            return False
        automaton = _grammars.compile_bfcl(rec.function, whitespace="json", channel_header=False)
        try:
            _report.gate(automaton, 512)
        except _report.TooLarge:
            self.problems.append(
                f"{rec.id}: automaton above 512 states; the record cannot be served"
            )
            return False
        payload = _artifact.encode_automaton(automaton)
        for arm in rotation(index):
            row = self.constrained(index, rec, arm, automaton, payload, d["draft_ids"])
            row["draft_schema_valid"] = d["schema_valid"]
            row["draft_exact_call"] = d["exact_call"]
            self.write(row)
            if row["error"] is not None or not row.get("cs", False):
                self.problems.append(
                    f"{rec.id} {arm.name}: {row['error'] or 'CS = 0 (host replay rejects the commit)'}"
                )
                return False
        return True

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------- the report


def _cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ex = [e for r in rows for e in r.get("executed", [])]
    walls = [float(r["wall_ms"]) for r in rows]
    return {
        "n": n,
        "exact": Fraction(sum(bool(r.get("exact_call")) for r in rows), n) if n else None,
        "schema": Fraction(sum(bool(r.get("schema_valid")) for r in rows), n) if n else None,
        "cs": Fraction(sum(bool(r.get("cs")) for r in rows), n) if n else None,
        "steps_mean": Fraction(sum(ex), len(ex)) if ex else None,
        "steps_p90": nearest_rank(ex, 0.9),
        "wall_p50": nearest_rank(walls, 0.5),
        "wall_p90": nearest_rank(walls, 0.9),
        "copy": Fraction(sum(bool(r.get("copy")) for r in rows), n) if n else None,
    }


def _f(x: Fraction | float | None, nd: int = 3) -> str:
    return "-" if x is None else f"{float(x):.{nd}f}"


def reading(by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    """Criterion 9 on the complete records: exact Fraction arithmetic, the H2-style choice, the anchoring test."""
    from local_astra.eval.cd import stats as _stats

    out = ["## criterion 9 reading (pre-registered; reported, not gated)"]
    base = _cell(by_arm[SCRATCH_MAR.name])
    if base["exact"] is None or base["steps_mean"] is None or base["wall_p50"] is None:
        return [*out, "no from-scratch perstep_mar rows: no reading"]
    passing: list[tuple[Fraction, float, int, str]] = []
    worth = False  # criterion 9: some seeded cell meets all three conditions
    for arm in SEEDED_MAR:
        c = _cell(by_arm[arm.name])
        if c["exact"] is None or c["steps_mean"] is None or c["wall_p50"] is None:
            out.append(f"{arm.name}: no rows")
            continue
        tol = c["exact"] >= base["exact"] - TOLERANCE
        steps_ok = c["steps_mean"] <= STEP_RATIO * base["steps_mean"]
        wall_ok = Fraction(c["wall_p50"]) <= WALL_RATIO * Fraction(base["wall_p50"])
        out.append(
            f"{arm.name}: exact {c['exact']} vs scratch {base['exact']} - 1/20 -> {tol}; mean steps "
            f"{c['steps_mean']} vs 3/5 x {base['steps_mean']} -> {steps_ok}; p50 wall {c['wall_p50']:.1f} vs "
            f"4/5 x {base['wall_p50']:.1f} -> {wall_ok}"
        )
        worth = worth or (tol and steps_ok and wall_ok)
        if tol:
            passing.append(
                (c["steps_mean"], float(arm.noise or 0.0), int(arm.start or 0), arm.name)
            )
    if passing:
        chosen = min(passing)
        out.append(
            f"chosen cell (tolerance, then fewest steps, then lower noise, then lower start): {chosen[3]}"
        )
    else:
        out.append("chosen cell: none within the exact-call tolerance")
    out.append(f"worth wiring into T-0080c: {'YES' if worth else 'NO'}")
    out.append(
        "## anchoring: exact_call on schema-invalid drafts, seeded vs from scratch (McNemar, exact)"
    )
    scratch = {
        r["id"]: bool(r["exact_call"])
        for r in by_arm[SCRATCH_MAR.name]
        if not r["draft_schema_valid"]
    }
    for arm in SEEDED_MAR:
        seeded = {
            r["id"]: bool(r["exact_call"]) for r in by_arm[arm.name] if not r["draft_schema_valid"]
        }
        ids = sorted(set(scratch) & set(seeded))
        b = sum(scratch[i] and not seeded[i] for i in ids)
        c2 = sum(seeded[i] and not scratch[i] for i in ids)
        n_s, n_d = sum(scratch[i] for i in ids), sum(seeded[i] for i in ids)
        p = _stats.mcnemar_exact(b, c2) if ids else None
        out.append(
            f"{arm.name}: n {len(ids)}, exact scratch {n_s}, seeded {n_d}, discordant {b}/{c2}, "
            f"p {_f(p, 4)}; seeded lower: {'yes' if n_d < n_s else 'no'}"
        )
    return out


def report(run: Run, regime: str, complete: bool) -> str:
    drafts = {r["id"]: r for r in run.rows if r["arm"] == "draft" and not r.get("error")}
    done = {r["id"] for r in run.rows if r["arm"] != "draft"}
    full = [
        i
        for i in done
        if sum(1 for r in run.rows if r["id"] == i and r["arm"] != "draft") == len(ARMS)
    ]
    rows = [r for r in run.rows if r["arm"] != "draft" and r["id"] in full]
    by_arm: dict[str, list[dict[str, Any]]] = {
        a.name: [r for r in rows if r["arm"] == a.name] for a in ARMS
    }
    lines = ["# T-0081 tier 2: seeded repair, BFCL dev", f"regime: {regime}"]
    if run.xargs_check:
        lines.append(f"xargs check: {run.xargs_check} -- accepted (no refusal record)")
    nd = len(drafts)
    nv = sum(1 for d in drafts.values() if d["schema_valid"])
    lines.append(
        f"drafts: {nd}; schema-valid {nv}; exact_call {sum(1 for d in drafts.values() if d['exact_call'])}; "
        f"records with all {len(ARMS)} arms: {len(full)}; flush requests {run.flushes}"
    )
    head = (
        f"{'arm':<14} {'split':<8} {'n':>4} {'exact':>6} {'schema':>6} {'CS':>6} {'steps':>7} {'p90':>5} "
        f"{'wall50':>8} {'wall90':>8} {'copy':>6}"
    )
    lines.append(head)
    for arm in ARMS:
        for split, pred in (
            ("all", lambda r: True),
            ("valid", lambda r: r["draft_schema_valid"]),
            ("invalid", lambda r: not r["draft_schema_valid"]),
        ):
            c = _cell([r for r in by_arm[arm.name] if pred(r)])
            lines.append(
                f"{arm.name:<14} {split:<8} {c['n']:>4} {_f(c['exact']):>6} {_f(c['schema']):>6} "
                f"{_f(c['cs']):>6} {_f(c['steps_mean'], 2):>7} {_f(c['steps_p90'], 0):>5} "
                f"{_f(c['wall_p50'], 1):>8} {_f(c['wall_p90'], 1):>8} {_f(c['copy']):>6}"
            )
    lines.append("## repair rate on schema-invalid drafts (seeded output schema-valid)")
    for arm in (*SEEDED_MAR, SEEDED_EP):
        inv = [r for r in by_arm[arm.name] if not r["draft_schema_valid"]]
        k = sum(bool(r.get("schema_valid")) for r in inv)
        lines.append(f"{arm.name}: {k}/{len(inv)}")
    lines += reading(by_arm)
    for p in run.problems:
        lines.append(f"PROBLEM: {p}")
    lines.append("run: COMPLETE" if complete and not run.problems else "run: STOPPED")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--boot-log", type=Path, default=REPO / "logs" / "vllm.log")
    ap.add_argument("--limit", type=int, default=None, help="first N dev records (dry runs only)")
    args = ap.parse_args()

    dirty = git("status", "--porcelain")
    if dirty:
        die(2, f"refusing: the tree is dirty:\n{dirty}")
    if git("diff", f"{REFERENCE_COMMIT}..HEAD", "--", "services/vllm.sh"):
        die(
            2, f"refusing: services/vllm.sh changed since {REFERENCE_COMMIT} (T-0071 shared regime)"
        )
    if not args.boot_log.is_file():
        die(2, f"refusing: no engine log at {args.boot_log}")
    head = git("rev-parse", "HEAD")
    out = REPO / "logs" / f"t0081-seed-{head[:7]}"
    if out.exists():
        die(2, f"refusing: {out} exists; move it aside first")
    knobs = load_knobs(CALIBRATION)
    knobs_sha = hashlib.sha256(CALIBRATION.read_bytes()).hexdigest()
    try:
        with urllib.request.urlopen(V1 + "/models", timeout=10) as resp:
            model = str(json.loads(resp.read())["data"][0]["id"])
    except Exception as exc:
        die(2, f"no engine on {API} ({exc}); the locked script boots it")
    status, _ = post(
        {
            "model": model,
            "max_tokens": 8,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "hi"}],
        },
        timeout_s=120.0,
    )
    if status != 400:
        die(2, f"temperature returned {status}, not 400: not the diffusion path")

    from local_astra.eval.cd import cuts as _cuts

    recs = list(_cuts.load("bfcl", "dev"))
    if args.limit is not None:
        recs = recs[: args.limit]
    out.mkdir(parents=True)
    start_offset = args.boot_log.stat().st_size
    regime = (
        f"T-0081 tier 2, commit {head[:7]} clean, {model} rev 2ea8372, vllm v0.29.0, CL {CANVAS}, "
        f"{MAX_STEPS} steps, t 0.8->0.4, TRITON_ATTN, engine only via services/vllm.sh (unchanged since "
        f"{REFERENCE_COMMIT}), enable_thinking=false, max_tokens 256, BFCL dev 130-229 n {len(recs)}, JSON "
        f"whitespace, channel_header=False fence=True, perstep_mar knobs {knobs['entropy_bound']}/"
        f"{knobs['confidence_threshold']} from {CALIBRATION.relative_to(REPO)} sha256 {knobs_sha}, arms {[a.name for a in ARMS]} rotated by record index after the "
        f"draft, no sampler seed, CD generator seed fixed (plugin), concurrency 1, python "
        f"{platform.python_version()}"
    )
    print(regime, flush=True)
    run = Run(out, args.boot_log, knobs)
    run.model = model
    complete = True
    try:
        for index, rec in enumerate(recs):
            if not run.record(index, rec):
                complete = False
                break
            print(f"{index + 1}/{len(recs)} {rec.id} ok", flush=True)
    except Exception as exc:  # the report must still be written; the run stops
        run.problems.append(f"driver crashed: {type(exc).__name__}: {exc}")
        complete = False
    finally:
        run.close()
        with args.boot_log.open("rb") as fh:
            fh.seek(start_offset)
            (out / "engine.log").write_bytes(fh.read())
    text = report(run, regime, complete)
    (out / "report.txt").write_text(text)
    sys.stdout.write(text)
    return 0 if complete and not run.problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
