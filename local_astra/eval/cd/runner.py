"""The CD evaluation runner (T-0070 criteria 2, 4-6, 8, 9, 12).

``run(RunConfig(...))`` runs one arm on one cut against the **loopback** vLLM server and writes
``<out_dir>/<ts>-<task>-<cut>-<arm>.jsonl``: a regime line, one row per record, and a trailer (summary,
header-prefix rate, ``stopped``). Every record of the cut stays in the artifact whatever happens to it
(HTTP error, timeout, ``join_error``, ``too_large``, loud CD error): failures are scored as the empty
emission, so ``n`` and BFCL's ``arg_total`` never shrink.

Constrained rows: the harness compiles the record's automaton (BFCL with standard JSON whitespace,
T-0062 decision 11; ``whitespace="pretty"`` is the JAX-link parity arm, restricted to records with
|S| <= ``max_states`` under pretty whitespace), gates it at ``max_states``, sends it as a T-0064
``CdRequest`` with a unique ``record_tag``, joins the ``ASTRA_CD`` records on that tag, checks the join on
the answer region, replays the committed ids through its own automaton (CS never comes from the plugin),
and scores the detokenised committed ids. Unconstrained rows score the detokenised generated ids
(``return_token_ids``), or the served text rebuilt per ruling E3 when the server returns none.

Ruling E9: the plugin may emit a canvas's record only at the next sampler call (T-0067 also flushes on
request release). After ``record_wait_s`` with an incomplete join, the runner sends up to two tiny
unconstrained flush requests (``max_tokens`` 1, no ``astra_cd_*``), waiting ``flush_wait_s`` after each,
before declaring ``join_error``. Every join is settled before the next record is sent, so the last record
of a run gets the same treatment. Flush requests are never rows and never in any denominator; the row
records how many it needed (``flush_requests``), the trailer the run's total.

Running this against the real engine is GPU time: go through ``/launch`` (``scripts/gpu/cd_eval.py``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import hashlib
import ipaddress
import json
import socket
import subprocess
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any

from pydantic import ValidationError

from local_astra.contracts.cd import MAX_AUTOMATON_CHARS
from local_astra.eval.cd import arms as _arms
from local_astra.eval.cd import calibration as _calibration
from local_astra.eval.cd import cuts as _cuts
from local_astra.eval.cd import hardstop as _hardstop
from local_astra.eval.cd import prompts as _prompts
from local_astra.eval.cd import records as _records
from local_astra.eval.cd import replay as _replay

__all__ = ["RunConfig", "RunResult", "check_loopback", "read_artifact", "read_rows", "run"]

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
EXIT_OK, EXIT_STOPPED, EXIT_REFUSED = 0, 3, 2
_LOOPBACK_NAMES = frozenset({"localhost"})
#: Ruling E9: at most this many flush requests per incomplete join.
MAX_FLUSHES = 2
FLUSH_PROMPT = "Reply with OK."

BFCL_SCORE_KEYS = ("schema_valid", "arg_correct", "arg_total", "exact_call")
SUDOKU_SCORE_KEYS = ("solved", "why")


@dataclasses.dataclass(frozen=True)
class RunConfig:
    task: str
    cut: str
    arm: str
    base_url: str
    engine_log: Path
    out_dir: Path
    repo: Path
    expected_commit: str
    calibration: Path | None = None
    limit: int | None = None
    budget_s: float = 14_400.0
    request_timeout_s: float = 600.0
    whitespace: str = "json"
    max_states: int = 512
    step_cap: int = 48
    #: The calibration driver's grid point (dev cut only); test-cut Mar runs need ``calibration``.
    mar_knobs: Mapping[str, float] | None = None
    #: How long to wait for a request's ``ASTRA_CD`` records after its response.
    record_wait_s: float = 5.0
    #: How long to wait again after each flush request (ruling E9).
    flush_wait_s: float = 2.0


@dataclasses.dataclass(frozen=True)
class RunResult:
    exit_code: int
    artifact: Path
    stopped: str | None
    n: int
    summary: Mapping[str, Any]


# --------------------------------------------------------------------------- guards


def check_loopback(base_url: str) -> None:
    """Refuse anything but a loopback literal (or ``localhost``) without resolving a name."""
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"base_url must be http(s): {base_url!r}")
    host = parsed.hostname or ""
    if host in _LOOPBACK_NAMES:
        return
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"base_url host {host!r} is not a loopback address") from None
    if not addr.is_loopback:
        raise ValueError(f"base_url host {host!r} is not a loopback address")


def _validate(cfg: RunConfig) -> tuple[dict[str, float] | None, str | None]:
    """Static refusals (``ValueError``) and the Mar knobs with the calibration file's sha."""
    if cfg.task not in _cuts.TASKS or cfg.cut not in _cuts.CUTS:
        raise ValueError(f"unknown task/cut {cfg.task!r}/{cfg.cut!r}")
    spec = _arms.spec(cfg.arm)
    if cfg.whitespace not in _replay.WHITESPACES:
        raise ValueError(f"whitespace must be one of {_replay.WHITESPACES}")
    if cfg.whitespace == "pretty" and (cfg.task != "bfcl" or spec.mode is None):
        raise ValueError("the pretty-whitespace parity arm is a constrained BFCL arm")
    mar: dict[str, float] | None = None
    sha: str | None = None
    if cfg.calibration is not None:
        # T-0076 criterion 8: a calibrated arm runs only on its calibration source's frozen file
        mar, sha = _calibration.load_frozen(
            cfg.calibration, task=cfg.task, arm=cfg.arm if spec.needs_calibration else None
        )
    if spec.needs_calibration:
        if mar is None:
            if cfg.mar_knobs is None or cfg.cut != "dev":
                raise ValueError(
                    f"{cfg.arm} needs a frozen calibration.json (criterion 10); run the "
                    "calibration driver on the dev cut first"
                )
            mar = {k: float(cfg.mar_knobs[k]) for k in _calibration.KNOBS}
    else:
        mar = None
    return mar, sha


# --------------------------------------------------------------------------- I/O helpers


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


def _versions(model: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {"served_model": model, "harness": "local_astra.eval.cd (T-0070)"}
    lock = _REPO / "services" / "versions.lock"
    if lock.is_file():
        raw = lock.read_bytes()
        out["versions_lock_sha256"] = hashlib.sha256(raw).hexdigest()
        with contextlib.suppress(tomllib.TOMLDecodeError):
            out["versions_lock"] = tomllib.loads(raw.decode())
    try:
        from local_astra.decoding import grammars, vocab

        out["dgfa_commit"] = grammars.DGFA_COMMIT
        out["compiler_version"] = grammars.compiler_version()
        out["tokenizer_sha256"] = vocab.TOKENIZER_SHA256
        out["model_revision"] = vocab.MODEL_REVISION
    except ImportError:  # the cd group is absent: the regime says so rather than guessing
        out["compiler_version"] = None
    return out


def _served_model(base_url: str, timeout_s: float) -> str | None:
    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/models", timeout=min(timeout_s, 10.0)
        ) as resp:
            data = json.loads(resp.read())
        return str(data["data"][0]["id"])
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def _post(base_url: str, body: Mapping[str, Any], timeout_s: float) -> tuple[Any, str | None]:
    """``(payload, None)`` or ``(None, "timeout" | "http_error")``."""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read()), None
    except TimeoutError:
        return None, "timeout"
    except urllib.error.HTTPError:
        return None, "http_error"
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            return None, "timeout"
        return None, "http_error"
    except (OSError, ValueError):
        return None, "http_error"


def _timestamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _artifact_path(cfg: RunConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        path = cfg.out_dir / f"{_timestamp()}-{cfg.task}-{cfg.cut}-{cfg.arm}.jsonl"
        if not path.exists():
            return path
        time.sleep(0.001)


def _write(fh: IO[str], obj: Mapping[str, Any]) -> None:
    fh.write(json.dumps(obj) + "\n")
    fh.flush()


def read_artifact(
    path: Path | str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """``(regime, record rows, other lines)``."""
    lines = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    rest = lines[1:]
    return lines[0], [x for x in rest if "id" in x], [x for x in rest if "id" not in x]


def read_rows(path: Path | str) -> list[dict[str, Any]]:
    return read_artifact(path)[1]


# --------------------------------------------------------------------------- scoring


def _scores(task: str, record: Any, text: str, **opts: bool) -> tuple[dict[str, Any], bool]:
    s = _cuts.score(task, record, text, **opts)
    keys = BFCL_SCORE_KEYS if task == "bfcl" else SUDOKU_SCORE_KEYS
    return {k: s[k] for k in keys}, bool(s["parsed"])


def _message(payload: Any) -> tuple[str | None, str, list[int] | None]:
    choice = payload["choices"][0]
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning")
    if reasoning is None:
        reasoning = msg.get("reasoning_content")
    ids = choice.get("token_ids")
    token_ids = [int(t) for t in ids] if isinstance(ids, list) else None
    return reasoning, str(msg.get("content") or ""), token_ids


# --------------------------------------------------------------------------- the run


class _Run:
    def __init__(self, cfg: RunConfig, mar: dict[str, float] | None) -> None:
        self.cfg = cfg
        self.mar = mar
        self.spec = _arms.spec(cfg.arm)
        self.constrained = self.spec.mode is not None
        self.run_id = uuid.uuid4().hex[:12]
        self.model = DEFAULT_MODEL
        self.log: _records.EngineLog | None = None
        self.flushes = 0

    def row_base(self, rid: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": rid,
            "text": "",
            "committed_ids": None,
            "parse": False,
            "scores": {},
            "cs": False if self.constrained else None,
            "steps": None,
            "wall_ms": 0.0,
            "bucket": None,
            "error": None,
            "knobs": _arms.arm_knobs(self.cfg.arm, self.mar) if self.constrained else {},
            "in_final": None,
        }
        if self.cfg.arm == "end_projection":
            row.update(shadow_argmax_ids=None, argmax_in_language=None, map_equals_argmax=None)
        return row

    def fail(self, row: dict[str, Any], record: Any, error: str) -> None:
        row["error"] = error
        row["scores"], _ = _scores(self.cfg.task, record, "")
        row["parse"] = False
        if self.cfg.task == "bfcl" and not self.constrained:
            row["scores_published"], _ = _scores(
                self.cfg.task, record, "", ci_enums=False, expand_wildcard=False
            )

    def flush(self) -> None:
        """Ruling E9: one tiny unconstrained request, so a plugin that emits a canvas's records at the
        next sampler call writes them. Never a row, never in a denominator."""
        body = _prompts.request_body(self.model, FLUSH_PROMPT)
        body["max_tokens"] = 1
        body.pop("return_token_ids", None)
        _post(self.cfg.base_url, body, self.cfg.request_timeout_s)
        self.flushes += 1

    def await_records(self, tag: str, row: dict[str, Any]) -> _records.Joined | None:
        """Wait ``record_wait_s`` for the tag's canvases; while incomplete, send up to
        ``MAX_FLUSHES`` flush requests, waiting ``flush_wait_s`` after each (ruling E9)."""
        assert self.log is not None
        joined = self.log.wait(
            tag, max_tokens=_prompts.MAX_TOKENS, timeout_s=self.cfg.record_wait_s
        )
        sent = 0
        while sent < MAX_FLUSHES and (
            joined is None or not _records.is_complete(joined.records, _prompts.MAX_TOKENS)
        ):
            self.flush()
            sent += 1
            joined = self.log.wait(
                tag, max_tokens=_prompts.MAX_TOKENS, timeout_s=self.cfg.flush_wait_s
            )
        row["flush_requests"] = sent
        return joined

    def unconstrained(self, index: int, record: Any, row: dict[str, Any]) -> dict[str, Any]:
        body = _prompts.request_body(self.model, _prompts.build_prompt(self.cfg.task, record))
        t0 = time.perf_counter()
        payload, err = _post(self.cfg.base_url, body, self.cfg.request_timeout_s)
        row["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        header: bool | None = None
        if err is not None:
            self.fail(row, record, err)
        else:
            try:
                reasoning, content, ids = _message(payload)
            except (KeyError, IndexError, TypeError, ValueError):
                self.fail(row, record, "http_error")
                return {"opens_with_header": None}
            if ids:
                text = _replay.detokenize(ids)
                header = ids[0] == _replay.CHANNEL_OPEN_ID
            else:
                text = _replay.served_text(reasoning, content)
            row["committed_ids"] = ids
            row["text"] = text
            row["scores"], row["parse"] = _scores(self.cfg.task, record, text)
            if self.cfg.task == "bfcl":
                row["scores_published"], _ = _scores(
                    self.cfg.task, record, text, ci_enums=False, expand_wildcard=False
                )
        return {"opens_with_header": header}

    def constrained_row(
        self, index: int, record: Any, row: dict[str, Any], automaton: Any
    ) -> dict[str, Any]:
        from local_astra.decoding import artifact as _artifact
        from local_astra.decoding import report as _report

        obs: dict[str, Any] = {}
        try:
            _report.gate(automaton, self.cfg.max_states)
        except _report.TooLarge:
            row["bucket"] = int(automaton.n_states_bucket)
            self.fail(row, record, "too_large")
            return obs
        payload = _artifact.encode_automaton(automaton)
        tag = f"t0070.{self.run_id}.{index}"
        try:
            xargs = _arms.cd_xargs(self.cfg.arm, payload, tag, self.mar)
        except ValidationError:
            too_big = len(payload) > MAX_AUTOMATON_CHARS
            self.fail(row, record, "payload_too_large" if too_big else "cd_request_invalid")
            return obs
        row["knobs"] = _arms.row_knobs(xargs)
        body = _prompts.request_body(
            self.model, _prompts.build_prompt(self.cfg.task, record), xargs
        )
        t0 = time.perf_counter()
        resp, err = _post(self.cfg.base_url, body, self.cfg.request_timeout_s)
        row["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        if err is not None:
            self.fail(row, record, err)
            return obs
        try:
            reasoning, content, _ids = _message(resp)
        except (KeyError, IndexError, TypeError, ValueError):
            self.fail(row, record, "http_error")
            return obs
        served = _replay.served_text(reasoning, content)
        assert self.log is not None
        joined = self.await_records(tag, row)
        if joined is None or not joined.valid:
            row["text"] = served
            self.fail(row, record, "join_error")
            return obs
        row["steps"] = joined.steps
        row["bucket"] = joined.bucket
        row["in_final"] = joined.in_final
        if self.cfg.arm == "end_projection":
            row["shadow_argmax_ids"] = [
                list(x) if x is not None else None for x in joined.per_canvas("shadow_argmax_ids")
            ]
            row["argmax_in_language"] = joined.per_canvas("argmax_in_language")
            row["map_equals_argmax"] = joined.per_canvas("map_equals_argmax")
        committed = joined.committed_ids
        row["committed_ids"] = committed
        if joined.error is not None:
            row["text"] = served
            self.fail(row, record, joined.error)
            return obs
        text = _replay.detokenize(committed)
        row["text"] = text
        if not _records.is_complete(joined.records, _prompts.MAX_TOKENS) or not (
            _replay.join_matches(text, served)
        ):
            self.fail(row, record, "join_error")
            row["text"] = text
            return obs
        accepted = _replay.accepts(automaton, committed)
        row["cs"] = accepted
        row["scores"], row["parse"] = _scores(self.cfg.task, record, text)
        return {
            "replay": accepted,
            "plugin_in_final": joined.in_final,
            "is_terminated": _replay.terminated(committed),
        }


def _population(
    cfg: RunConfig,
) -> tuple[list[Any], dict[str, Any], list[str]]:
    """The records to run, the pre-compiled automata (constrained arms) and the excluded ids (parity
    arm only: |S| above ``max_states`` under pretty whitespace)."""
    recs = list(_cuts.load(cfg.task, cfg.cut))
    if cfg.limit is not None:
        recs = recs[: cfg.limit]
    automata: dict[str, Any] = {}
    excluded: list[str] = []
    if _arms.spec(cfg.arm).mode is None:
        return recs, automata, excluded
    kept = []
    for rec in recs:
        a = _replay.compile_automaton(cfg.task, rec, whitespace=cfg.whitespace)
        if cfg.whitespace == "pretty" and int(a.n_states) > cfg.max_states:
            excluded.append(rec.id)
            continue
        automata[rec.id] = a
        kept.append(rec)
    return kept, automata, excluded


def run(cfg: RunConfig) -> RunResult:
    """Run one arm on one cut. Raises ``ValueError`` for a non-loopback URL or a refused configuration
    before any socket is opened; a hard stop returns a non-zero ``exit_code`` and marks the artifact."""
    check_loopback(cfg.base_url)
    mar, calib_sha = _validate(cfg)
    t_start = time.monotonic()
    state = _Run(cfg, mar)
    path = _artifact_path(cfg)
    head = _git(cfg.repo, "rev-parse", "HEAD") or None
    regime: dict[str, Any] = {
        "kind": "regime",
        "task": cfg.task,
        "cut": cfg.cut,
        "arm": cfg.arm,
        "knobs": _arms.arm_knobs(cfg.arm, mar),
        "whitespace": cfg.whitespace if cfg.task == "bfcl" and state.constrained else None,
        "label": "jax_link_parity"
        if cfg.whitespace == "pretty"
        else ("calibration" if cfg.mar_knobs is not None and cfg.calibration is None else None),
        "commit": head,
        "expected_commit": cfg.expected_commit,
        "enable_thinking": False,
        "max_tokens": _prompts.MAX_TOKENS,
        "calibration_sha": calib_sha,
        "calibration_path": str(cfg.calibration) if cfg.calibration else None,
        "max_states": cfg.max_states,
        "step_cap": cfg.step_cap,
        "budget_s": cfg.budget_s,
        "request_timeout_s": cfg.request_timeout_s,
        "run_id": state.run_id,
        "started_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "score_normalisation": {"ci_enums": True, "expand_wildcard": True},
    }
    with path.open("w") as fh:
        stop = _hardstop.check_tree(cfg.repo, cfg.expected_commit)
        if stop is not None:
            regime.update(n=0, versions=_versions(None), stopped=stop)
            _write(fh, regime)
            _write(fh, {"kind": "trailer", "stopped": stop})
            return RunResult(EXIT_REFUSED, path, stop, 0, {})
        recs, automata, excluded = _population(cfg)
        served = _served_model(cfg.base_url, cfg.request_timeout_s)
        state.model = served or DEFAULT_MODEL
        regime.update(n=len(recs), excluded_ids=excluded, versions=_versions(served))
        _write(fh, regime)
        state.log = _records.EngineLog(cfg.engine_log)
        monitor = _hardstop.Monitor(
            cfg.task, cfg.arm, constrained=state.constrained, step_cap=cfg.step_cap
        )
        stop = None
        for index, rec in enumerate(recs):
            stop = _hardstop.check_budget(time.monotonic() - t_start, cfg.budget_s)
            if stop is not None:
                break
            row = state.row_base(rec.id)
            if state.constrained:
                obs = state.constrained_row(index, rec, row, automata[rec.id])
            else:
                obs = state.unconstrained(index, rec, row)
            _write(fh, row)
            stop = monitor.observe(row, **obs)
            if stop is not None:
                break
        if stop is None:
            stop = _hardstop.check_budget(time.monotonic() - t_start, cfg.budget_s)
        summary = _summary(cfg.task, read_rows_from(path))
        _write(
            fh,
            {
                "kind": "trailer",
                "summary": summary,
                "header_prefix_rate": monitor.header_prefix_rate(),
                "header_prefix_window": _hardstop.HEADER_WINDOW,
                "malformed_records": state.log.malformed,
                "flush_requests": state.flushes,
                "elapsed_s": time.monotonic() - t_start,
                "stopped": stop,
            },
        )
    return RunResult(EXIT_STOPPED if stop else EXIT_OK, path, stop, len(recs), summary)


def read_rows_from(path: Path) -> list[dict[str, Any]]:
    return [
        x
        for x in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
        if "id" in x
    ]


def _summary(task: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {
        "rows": n,
        "parsed": sum(bool(r["parse"]) for r in rows),
        "errors": sum(bool(r["error"]) for r in rows),
        "cs": sum(r["cs"] is True for r in rows),
    }
    if task == "bfcl":
        for k in ("schema_valid", "exact_call"):
            out[k] = sum(bool(r["scores"].get(k)) for r in rows)
        out["arg_correct"] = sum(int(r["scores"].get("arg_correct", 0)) for r in rows)
        out["arg_total"] = sum(int(r["scores"].get("arg_total", 0)) for r in rows)
    else:
        out["solved"] = sum(bool(r["scores"].get("solved")) for r in rows)
    return out
