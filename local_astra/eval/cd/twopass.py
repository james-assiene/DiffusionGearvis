"""The two-pass tool-call driver (T-0080b criteria 2-5): pass 1 once per case, every pass-2 cell on its output.

``run(TwoPassConfig(...))`` runs one stage (``dev`` or ``test`` split of ``<set_dir>/cases.jsonl``, file order,
first ``limit``) against the **loopback** vLLM server and writes ``<out_dir>/<ts>-toolcall-<stage>.jsonl``:

- a regime line (stage, cells, the system prompt and its sha256, ``tools_v1.json``'s sha256, the cases file's
  sha256, the frozen calibration files' sha256 and knobs, commit, versions, the bucket histogram);
- per case, one pass-1 row (``"pass": 1``), then one pass-2 row (``"pass": 2``) per cell, in an order rotated
  per routed case (T-0011 R17);
- a trailer line (``"kind": "trailer"``) with ``stopped``.

**Pass 1** is the system prompt (tool instructions, the v1 declarations as text, the ``<cc>`` convention, Q3)
plus the user message (with the case's screenshot as one ``image_url`` part), thinking on, ``max_tokens``
1024, ``return_token_ids``, no ``tools`` field and no ``vllm_xargs``. The generated ids are decoded on the host
(up to the first stop id) and routed with ``toolcall.router.route``: routing is on the model's answer, never
on the case label, and never on the served (parsed) fields.

**Pass 2** (only when pass 1 routed ``toolcall``), per cell ``"<thought>/<scope>/<arm>"``:

- ``T1`` = pass 1's body + ``vllm_xargs``, ``max_tokens`` 1024; ``T2`` = the same with 256; ``T4`` = pass 1's
  messages + ``{"role": "assistant", "content": spoken_prefix + "<cc>"}``, ``continue_final_message`` true,
  ``add_generation_prompt`` false, ``max_tokens`` 256, ``vllm_xargs``;
- scope ``named`` (K-D2): the grammar of the tool the draft names (``Route.draft_name``); when it names none,
  the name picker -- one request under ``scope="name"``, the picked name read from its decoded output, then
  one request under the picked tool's grammar with the same body (the row's ``requests`` is 2). ``union`` is
  accepted for reporting only (``select`` never chooses it);
- the CD keys are ``arms.cd_xargs(arm, payload, record_tag, knobs)`` with the knobs of
  ``calibration.load_frozen(path, task="bfcl", arm=arm)`` (a file that is not ``arm``'s source is refused).
- arm ``perstep_mar+seed`` (T-0081b; ``named`` only): ``perstep_mar``'s keys and frozen knobs plus the pass-1
  draft as a seed of the first canvas (``seed_ids``; ``astra_cd_draft*`` re-validated through ``CdRequest``),
  noise ``SEED_DRAFT_NOISE`` and start step ``SEED_DRAFT_START_STEP``. A case through the name picker is not
  seeded. Executed steps subtract the start step from the first canvas only (``executed_steps``).

Each pass-2 request's ``ASTRA_CD`` records are joined on its ``record_tag`` (ruling E9 flushes, as the T-0070
runner). CS is the host replay of the committed ids through the request's own automaton; the call is parsed
and scored from the host-decoded ``return_token_ids`` (which must equal the committed ids up to the stop),
never from ``content``/``reasoning``. Prefix-cache hit tokens are the delta of ``vllm:prefix_cache_hits_total``
on ``/metrics`` around the request (``None`` when the counters are unavailable -- never guessed).

**Stops** (the trailer's ``stopped``, the report is still written): an ``http_error``/``timeout`` (engine stop,
T-0071b C9/C12) and missing ``return_token_ids`` exit 1; a constrained terminated output the replay rejects,
replay disagreeing with the plugin's ``in_final``, a ``join_error`` or a loud CD error (``z0``,
``empty_delta``, ``infeasible``, ``internal``) exit 1; so do, on a seeded request only, a seed the contract
refuses (``seed_invalid``, never sent unseeded), a ``draft_*`` or ``missing_key`` record reason, and a first
canvas whose step counter is at or below the start step (``seed_accounting``). A dirty tree or a commit
mismatch exits 2 before any request; a non-loopback URL, an unknown cell (a ``union`` seed cell included), a
wrong calibration file or a calibration entry keyed by the seed arm raise ``ValueError`` before any socket.

``derive(rows, policy)`` builds per-case policy rows offline (``draft_only``, ``cd_on_invalid:<cell>``,
``cd_always:<cell>`` or a bare cell), ``select(rows, rule)`` is the pre-registered choice of T-0080c
(``stage1``/``stage2``), and ``report(paths)`` recomputes every figure from artifacts only.
``rescore(rows, tools, cases)`` (T-0082a) re-routes and re-scores stored pass-1 rows from their ``text`` under
the current router and scorer (pass-2 rows untouched), refusing a ``text`` that is not the decode of its ids or
a re-route that changes ``kind``/``tag``/``spoken_prefix``/``draft_name``. Pass-1 rows record the router's
``draft_close`` (a terminal ``</cc>`` removed from the draft, T-0082a).

**Replay** (T-0082b, ``pass1_from``/``pass1_sha256``): no pass-1 request is sent. The pass-1 rows are the stored
rows of that earlier artifact re-scored with ``rescore`` (``wall_ms`` and the prefix fields kept, plus
``replayed_from``); pass 2 runs exactly as live (same bodies, seeds, order and stops), each routed case preceded by
one warm-up (its pass-1 body with ``max_tokens`` 1, a ``"pass": "warmup"`` row, counted in the trailer; ruling R10).
A calibration absent from the source is recorded (``calibration_absent_in_source``, R9). The source is refused
(``ValueError``, before any socket) on a sha mismatch, another stage, ``stopped``, a pass-1 error, a regime
difference (``_replay_source``) or a ``rescore`` refusal. The regime names the source.

**The native format** (T-0084c): a cell ``native/<thought>/<scope>/<arm>`` runs the model's own call format beside
the ``<cc>`` cells, paired by case. A run sends a native pass 1 iff it has a native cell (``NATIVE_SYSTEM_PROMPT``,
the same user content, ``tools`` = ``native.tools_field``, ``tool_choice`` ``"none"``) and a cc pass 1 iff it has a
cc cell; the two pass 1s of a case alternate (cc first on even case indexes). Native pass-2 bodies keep ``tools``;
T4 continues ``spoken_prefix + "<|tool_call>"``; the grammars are ``compile_toolcall(..., call_format="native")``;
drafts and outputs are parsed by ``native.parse_call``; the named scope is ``native.draft_name``. A native pass 1
routed with another ``toolcall`` tag is ``wrong_tag`` (called, invalid; QN7) and, since H-T84-0 (T-0084d criterion 7),
is retried like a native draft without a name: the native picker, then one constrained native pass 2. Every native
request is
checked against its served ``prompt_token_ids`` (``prompt_guard``: ``missing``, ``declarations``, ``prefix``,
``continuation``; each a stop). T-0085: before any T4 request (routed or picker, both formats) the continuation
``spoken_prefix + tag`` that ``pass2_body`` appends is checked (``continuation_guard``): the request is not sent
and the run stops when the pinned template's ``strip_thinking`` would change it (``template``, mirrored by
``router.template_content``) or the spoken part carries a structural marker (``marker``; ``template`` wins when
both hold). A stopped mixed report chooses no format. Every row carries ``format`` (``row_format``: absent reads as cc); ``derive`` pairs
a policy with its own format's pass-1 rows (``draft_only`` / ``draft_only:native``). A replay refuses native rows
and native cells. A cc-only run is unchanged apart from ``format: "cc"`` on its rows.

Running this against the real engine is GPU time: ``scripts/gpu/cd_toolcall.py`` via ``/launch``.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as _dt
import functools
import hashlib
import json
import math
import re
import statistics
import struct
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import IO, Any

from pydantic import ValidationError

from local_astra.eval.cd import arms as _arms
from local_astra.eval.cd import calibration as _calibration
from local_astra.eval.cd import hardstop as _hardstop
from local_astra.eval.cd import records as _records
from local_astra.eval.cd import runner as _runner
from local_astra.eval.cd import stats as _stats
from local_astra.toolcall import native as _native
from local_astra.toolcall import router as _router
from local_astra.toolcall import score as _score
from local_astra.toolcall import tools as _tools

__all__ = [
    "CELL_ARMS",
    "LOUD_ERRORS",
    "NATIVE_SYSTEM_PROMPT",
    "STAGE1_TOLERANCE",
    "STAGE2_TOLERANCE",
    "SYSTEM_PROMPT",
    "WALL_TOLERANCE",
    "FormatChoice",
    "TwoPassConfig",
    "TwoPassResult",
    "derive",
    "executed_steps",
    "pass1_body",
    "render_markdown",
    "report",
    "rescore",
    "row_format",
    "run",
    "seed_ids",
    "select",
    "select_format",
    "system_prompt",
]

EXIT_OK, EXIT_STOPPED, EXIT_REFUSED = 0, 1, 2
STAGES = ("dev", "test")
THOUGHTS = ("T1", "T2", "T4")
SCOPES = ("named", "union")
#: T-0081b: the draft-seeded arm. Nothing outside this module knows it: it maps to its base arm for
#: ``arms.spec``, ``arms.cd_xargs`` and ``calibration.load_frozen``.
SEED_ARM = "perstep_mar+seed"
SEED_BASE_ARM = "perstep_mar"
#: The seed cell's draft knobs, fixed (not CLI options): T-0081 tier 2 at dad1382, cell ``mar_n0_s24``
#: (BFCL dev n 100, thinking off: exact 0.51, 3.31 steps, p50 213.7 ms vs from scratch 0.49 / 6.40 / 370.7 ms).
SEED_DRAFT_NOISE = 0.0
SEED_DRAFT_START_STEP = 24
#: The engine-global canvas length (``--diffusion-config``): the seed is cut to it.
CANVAS_LENGTH = 256
#: Arms a pass-2 cell may use (T-0080c stage 2; K-D2: stage 1 is ``perstep_mar`` only).
CELL_ARMS = ("end_projection", "perstep_mar", "perstep_mar_qargmax", SEED_ARM)
PASS1_MAX_TOKENS = 1024
PASS2_MAX_TOKENS = {"T1": 1024, "T2": 256, "T4": 256}
#: Criterion 4: loud CD errors that stop the run (T-0071 criterion 11 pattern); refusals such as
#: ``bucket_too_large`` are recorded and the run goes on.
LOUD_ERRORS = frozenset({*_hardstop.STOP_B_CODES, "internal"})
ENGINE_ERRORS = frozenset({"http_error", "timeout"})
#: Q7 tolerances, exact.
STAGE1_TOLERANCE = Fraction(1, 10)
STAGE2_TOLERANCE = Fraction(1, 20)
WALL_TOLERANCE = Fraction(105, 100)
CC = _router.TAGS["cc"]
CHANNEL_CLOSE_ID = 101
#: T-0084c: the call formats, the native cell prefix, the native markers and the declaration ids (46/47 around each
#: ``<|tool>declaration:...<tool|>`` the template renders in the system turn)
FORMATS = ("cc", "native")
NATIVE_PREFIX = "native/"
TOOL_CALL_OPEN = _native.TOOL_CALL_OPEN
TOOL_CALL_CLOSE = _native.TOOL_CALL_CLOSE
TOOL_CALL_OPEN_ID = 48
DECLARATION_OPEN_ID, DECLARATION_CLOSE_ID = 46, 47
WRONG_TAG = "wrong_tag"
PROMPT_GUARD = "prompt_guard"
#: T-0085 criterion 3: the pre-send T4 continuation guard and the structural markers its (b) refuses in the spoken part
CONTINUATION_GUARD = "continuation_guard"
STRUCTURAL_MARKERS = (
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|tool>",
    "<tool|>",
    "<|think|>",
    "<bos>",
    "<eos>",
)
METRIC_QUERIES = "vllm:prefix_cache_queries_total"
METRIC_HITS = "vllm:prefix_cache_hits_total"
MAX_FLUSHES = _runner.MAX_FLUSHES

# --------------------------------------------------------------------------- the system prompt

_PROMPT_HEAD = """You are Gearvis, a voice assistant that can see the user's screen and act on it.

Answer the user briefly. When an action is needed, first say in one short sentence what you will do, then write the tag <cc> followed by exactly one JSON tool call of the form {"name": "<tool name>", "arguments": {...}} and nothing after it. Use at most one call per answer, and only the tools declared below. When no action is needed, answer normally and never write <cc>.

Screen coordinates x and y are integers from 0 to 999, normalised to the attached screenshot: (0, 0) is its top-left corner and (999, 999) its bottom-right corner.

Tools (JSON declarations):
"""


def system_prompt(tools: _tools.ToolSet | None = None) -> str:
    """The one fixed system prompt of both passes (criterion 2): instructions, the declarations as text."""
    ts = tools if tools is not None else _tools.load()
    decls = json.dumps(list(ts.declarations), ensure_ascii=False, indent=2)
    return _PROMPT_HEAD + decls + "\n"


SYSTEM_PROMPT = system_prompt()

#: T-0084c: the native pass-1 system prompt (the task's wording, no trailing newline; the human approves it at
#: H-T84-0). The template appends the declarations from the request's ``tools`` field.
NATIVE_SYSTEM_PROMPT = (
    "You are Gearvis, a voice assistant that can see the user's screen and act on it.\n\n"
    "Answer the user briefly. When an action is needed, first say in one short sentence what you will do, then "
    "call exactly one of the declared tools. Use at most one call per answer. When no action is needed, answer "
    "normally and call no tool.\n\n"
    "Screen coordinates x and y are integers from 0 to 999, normalised to the attached screenshot: (0, 0) is its "
    "top-left corner and (999, 999) its bottom-right corner."
)


def tools_field_json(tools: _tools.ToolSet) -> str:
    """The canonical JSON of the native ``tools`` field (its sha256 is in the regime; ruling C1-C7)."""
    return json.dumps(
        _native.tools_field(tools), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def row_format(row: Mapping[str, Any]) -> str:
    """The one place a row's format is read: its ``format`` field; rows written before T-0084c have none and
    are cc."""
    return str(row.get("format") or "cc")


def cell_format(name: str) -> str:
    """The format of a cell name (``native/...`` is native, everything else cc)."""
    return "native" if name.startswith(NATIVE_PREFIX) else "cc"


# --------------------------------------------------------------------------- config


@dataclasses.dataclass(frozen=True)
class TwoPassConfig:
    stage: str
    cells: Sequence[str]
    base_url: str
    engine_log: Path
    out_dir: Path
    repo: Path
    expected_commit: str
    calibration: Mapping[str, Path]
    set_dir: Path
    limit: int | None = None
    request_timeout_s: float = 600.0
    record_wait_s: float = 5.0
    flush_wait_s: float = 2.0
    max_states: int = 512
    whitespace: str = "json"
    #: T-0082b: replay mode -- pass 1 is not sent; its rows are the re-scored pass-1 rows of this earlier
    #: artifact, whose sha256 must be ``pass1_sha256`` (both or neither)
    pass1_from: Path | None = None
    pass1_sha256: str | None = None


@dataclasses.dataclass(frozen=True)
class TwoPassResult:
    exit_code: int
    artifact: Path
    stopped: str | None
    n: int


@dataclasses.dataclass(frozen=True)
class Cell:
    name: str
    thought: str
    scope: str
    arm: str
    #: T-0084c: ``cc`` or ``native`` (``cell_format`` of the name)
    fmt: str = "cc"


def parse_cell(name: str) -> Cell:
    """``<thought>/<scope>/<arm>`` (cc) or ``native/<thought>/<scope>/<arm>`` (T-0084c); anything else, an unknown
    format included, raises ``ValueError``."""
    fmt = cell_format(name)
    parts = name.split("/")
    if fmt == "native":
        parts = parts[1:]
    if len(parts) != 3:
        raise ValueError(
            f"a cell is '<thought>/<scope>/<arm>' or 'native/<thought>/<scope>/<arm>', not {name!r}"
        )
    thought, scope, arm = parts
    if thought not in THOUGHTS:
        raise ValueError(f"{name}: thought must be one of {THOUGHTS}")
    if scope not in SCOPES:
        raise ValueError(f"{name}: scope must be one of {SCOPES}")
    if arm not in CELL_ARMS:
        raise ValueError(f"{name}: arm must be one of {CELL_ARMS}")
    if arm == SEED_ARM and scope != "named":
        raise ValueError(f"{name}: {SEED_ARM} is a named-scope arm only (T-0081b)")
    return Cell(name, thought, scope, arm, fmt)


def base_arm(arm: str) -> str:
    """The arm whose spec, CD keys and frozen knobs a cell arm uses (the seed arm's is ``perstep_mar``)."""
    return SEED_BASE_ARM if arm == SEED_ARM else arm


def executed_steps(steps: Sequence[int], k0: int = 0) -> list[int]:
    """Executed steps per canvas = the record's step counter at commit, minus the seeded draft's start step
    ``k0`` (T-0081 ``draft_start_step``; 0 without a seeded draft) on the first canvas only: only the first
    canvas is seeded, later canvases start at step 0 (T-0081 criteria 4-5). Total; the ``steps <= k0`` check
    is a stop (``_Run.check``). The one place this is computed."""
    out = [int(s) for s in steps]
    if out:
        out[0] -= int(k0)
    return out


# --------------------------------------------------------------------------- the draft seed (T-0081b)


def _seed_parts(
    thought: str, pass1_ids: Sequence[int], fmt: str = "cc"
) -> tuple[list[int], str, str]:
    """``(H, T, seed_source)`` of criterion 2; T is ``<cc>`` (cc) or ``<|tool_call>`` (native, T-0084c QN6) for
    T1/T2."""
    from local_astra.decoding import report as _report

    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}, not {fmt!r}")
    tag = CC if fmt == "cc" else TOOL_CALL_OPEN
    ids = [int(t) for t in pass1_ids]
    header = list(_report._PHASE0_HEADER)
    if thought == "T1":
        if ids and ids[0] == header[0] and CHANNEL_CLOSE_ID in ids:
            return ids[: ids.index(CHANNEL_CLOSE_ID) + 1], tag, "t1_thought"
        return header, tag, "t1_empty_header"
    if thought == "T2":
        return header, tag, "t2_header"
    if thought == "T4":
        return [], "", "t4_call"
    raise ValueError(f"thought must be one of {THOUGHTS}, not {thought!r}")


def seed_source(thought: str, pass1_ids: Sequence[int]) -> str:
    """Where the seed's head comes from: ``t1_thought``, ``t1_empty_header``, ``t2_header`` or ``t4_call``."""
    return _seed_parts(thought, pass1_ids)[2]


def seed_ids(
    thought: str,
    pass1_ids: Sequence[int],
    draft_call_text: str,
    canvas_length: int = CANVAS_LENGTH,
    *,
    fmt: str = "cc",
) -> list[int]:
    """The draft seed of a pass-2 canvas (T-0081b criterion 2)::

        seed(thought) = (H(thought) + encode(T(thought) + draft_call_text.rstrip()) + S)[:canvas_length]

    H(T1) = pass-1 ids verbatim through the first ``101`` when they start with ``100`` and contain a ``101``,
    else the empty-thought header; H(T2) = the empty-thought header; H(T4) = nothing. T = ``<cc>`` for T1/T2
    (``fmt="cc"``) or ``<|tool_call>`` (``fmt="native"``, T-0084c QN6), ``""`` for T4. ``draft_call_text`` is used
    raw apart from trailing whitespace (ruling R2: no grammar admits whitespace between the call and the stop; no
    parse, other trim or repair) and re-tokenised with the pinned tokenizer (the canvas starts at a fresh token
    boundary). S = the first stop id of pass 1, if any. ``ValueError`` for another ``fmt``."""
    from local_astra.decoding import vocab

    head, tag, _source = _seed_parts(thought, pass1_ids, fmt)
    stops = _replay_stop_ids()
    stop = next((int(t) for t in pass1_ids if int(t) in stops), None)
    tail = [stop] if stop is not None else []
    out = [*head, *(int(t) for t in vocab.encode(tag + draft_call_text.rstrip())), *tail]
    return out[: int(canvas_length)]


def seeded_xargs(xargs: Mapping[str, Any], seed: Sequence[int]) -> dict[str, Any]:
    """``xargs`` (a ``CdRequest.to_xargs()``) plus the draft keys, re-validated through ``CdRequest``.
    Raises ``pydantic.ValidationError`` when the contract refuses the seed."""
    from local_astra.contracts.cd import CD_XARGS_PREFIX, CdRequest

    fields = {k[len(CD_XARGS_PREFIX) :]: v for k, v in xargs.items()}
    draft = base64.b64encode(struct.pack(f"<{len(seed)}i", *seed)).decode("ascii")
    req = CdRequest(
        **fields,
        draft=draft,
        draft_noise=SEED_DRAFT_NOISE,
        draft_start_step=SEED_DRAFT_START_STEP,
    )
    return req.to_xargs()


# --------------------------------------------------------------------------- the eval set


def _cases(cfg: TwoPassConfig) -> tuple[list[dict[str, Any]], str]:
    path = Path(cfg.set_dir) / "cases.jsonl"
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    manifest = Path(cfg.set_dir) / "MANIFEST.json"
    if manifest.is_file():
        pinned = json.loads(manifest.read_text()).get("cases_sha256")
        if pinned != sha:
            raise ValueError(f"{path} sha256 {sha} != MANIFEST's {pinned}")
    else:
        raise ValueError(f"no MANIFEST.json beside {path}")
    cases = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    chosen = [c for c in cases if c.get("split") == cfg.stage]
    if cfg.limit is not None:
        chosen = chosen[: cfg.limit]
    return chosen, sha


def _image_part(set_dir: Path, screen: str) -> dict[str, Any]:
    data = (Path(set_dir) / screen).read_bytes()
    url = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": url}}


def pass1_body(model: str, system: str, case: Mapping[str, Any], set_dir: Path) -> dict[str, Any]:
    """Criterion 2: system + user (image first, then the text, for a case with a screen), thinking on."""
    user: Any = str(case["user"])
    if case.get("screen"):
        user = [_image_part(set_dir, str(case["screen"])), {"type": "text", "text": user}]
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens": PASS1_MAX_TOKENS,
        "return_token_ids": True,
    }


def native_pass1_body(
    model: str, case: Mapping[str, Any], set_dir: Path, tools: _tools.ToolSet
) -> dict[str, Any]:
    """T-0084c criterion 2: the cc pass-1 body of the case with ``NATIVE_SYSTEM_PROMPT`` as the system message,
    ``tools`` = ``native.tools_field(tools)`` and ``tool_choice`` ``"none"`` (QN3: rendered, the server's parser
    bypassed)."""
    body = pass1_body(model, NATIVE_SYSTEM_PROMPT, case, set_dir)
    body["tools"] = _native.tools_field(tools)
    body["tool_choice"] = "none"
    return body


def pass2_body(
    p1: Mapping[str, Any],
    thought: str,
    spoken_prefix: str,
    xargs: Mapping[str, Any],
    fmt: str = "cc",
) -> dict[str, Any]:
    """A pass-2 body: a deep copy of the format's pass-1 body (a native one keeps ``tools``) + ``vllm_xargs`` and
    the thought's ``max_tokens``; T4 continues ``spoken_prefix`` + ``<cc>`` / ``<|tool_call>``."""
    body: dict[str, Any] = json.loads(json.dumps(p1))  # a deep copy: pass 1 is never mutated
    body["max_tokens"] = PASS2_MAX_TOKENS[thought]
    body["vllm_xargs"] = dict(xargs)
    if thought == "T4":
        tag = CC if fmt == "cc" else TOOL_CALL_OPEN
        body["messages"].append({"role": "assistant", "content": spoken_prefix + tag})
        body["continue_final_message"] = True
        body["add_generation_prompt"] = False
    return body


def continuation_guard(continuation: str, tag: str) -> str | None:
    """T-0085 criterion 3 on the exact T4 continuation ``spoken + tag`` (the last assistant content of the body):
    ``"template"`` when the pinned template's rendering changes it (``router.template_content(c) != c.strip()``),
    else ``"marker"`` when the spoken part holds a structural marker, else ``None`` (K1: ``template`` wins)."""
    if _router.template_content(continuation) != continuation.strip():
        return "template"
    spoken = continuation.removesuffix(tag)
    if any(m in spoken for m in _guarded_tokens()):
        return "marker"
    return None


@functools.cache
def _guarded_tokens() -> frozenset[str]:
    """K8: the named markers plus every special token of the pinned tokenizer (added tokens with ``special``)."""
    from local_astra.decoding import vocab

    added = vocab.load_tokenizer().get_added_tokens_decoder().values()
    return frozenset(STRUCTURAL_MARKERS) | frozenset(str(a.content) for a in added if a.special)


# --------------------------------------------------------------------------- I/O helpers


def _metrics(base_url: str) -> tuple[int, int] | None:
    """``(queries, hits)`` from ``/metrics`` at the server root (summed over label sets); ``None`` if the
    endpoint or either counter is unavailable. Reading it spends nothing (gpu-services rule)."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=10.0) as resp:
            if resp.status != 200:
                return None
            text = resp.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    out: list[int] = []
    for name in (METRIC_QUERIES, METRIC_HITS):
        pat = rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$"
        vals = [float(m.group(1)) for m in re.finditer(pat, text, re.MULTILINE)]
        if not vals:
            return None
        out.append(round(sum(vals)))
    return out[0], out[1]


def _delta(
    before: tuple[int, int] | None, after: tuple[int, int] | None
) -> tuple[int | None, int | None]:
    if before is None or after is None:
        return None, None
    return after[1] - before[1], after[0] - before[0]


def _post(
    base_url: str, body: Mapping[str, Any], timeout_s: float
) -> tuple[Any, str | None, float]:
    """``(payload, error, wall_ms)``: send to last byte, loopback, concurrency 1."""
    t0 = time.perf_counter()
    payload, err = _runner._post(base_url, body, timeout_s)
    return payload, err, (time.perf_counter() - t0) * 1000.0


def _token_ids(payload: Any) -> list[int] | None:
    try:
        ids = payload["choices"][0].get("token_ids")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return [int(t) for t in ids] if isinstance(ids, list) else None


def _until_stop(ids: Sequence[int]) -> list[int]:
    """The ids before the first stop id (the stop and anything after it dropped)."""
    out: list[int] = []
    for t in ids:
        if int(t) in _replay_stop_ids():
            break
        out.append(int(t))
    return out


def _replay_stop_ids() -> frozenset[int]:
    from local_astra.eval.cd import replay

    return replay.STOP_IDS


def _decode(ids: Sequence[int]) -> str:
    from local_astra.eval.cd import replay

    return replay.detokenize(_until_stop(ids))


def _timestamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _write(fh: IO[str], obj: Mapping[str, Any]) -> None:
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fh.flush()


def call_text_of(text: str, fmt: str = "cc") -> str:
    """The call part of a pass-2 output: the answer channel (closed thoughts removed; an unclosed thought
    leaves nothing), a leading ``<cc>`` (cc) or ``<|tool_call>`` (native) dropped."""
    scanned = _router._strip_closed_thoughts(text)
    if scanned is None:
        return ""
    tag = CC if fmt == "cc" else TOOL_CALL_OPEN
    s = scanned.lstrip()
    return s[len(tag) :] if s.startswith(tag) else scanned


def score_call(
    text: str, tools: _tools.ToolSet, gt: Mapping[str, Any] | None, fmt: str = "cc"
) -> tuple[str, dict[str, Any] | None, bool, bool]:
    """``(parse, call, schema_valid, exact)``; ``parse`` is ``"ok"`` or the parse failure's reason. A native call
    is parsed by ``native.parse_call`` (the stop already removed, ruling B5); both are scored unchanged."""
    parsed = _score.parse_call(text) if fmt == "cc" else _native.parse_call(text)
    if isinstance(parsed, _score.ParseFailure):
        return parsed.reason, None, False, False
    valid = _score.schema_valid(parsed, tools)
    ex = bool(gt) and _score.exact(parsed, gt or {})
    return "ok", parsed, valid, ex


def _open_thought_text(text: str) -> str | None:
    """The text from the first ``<|channel>`` no later ``<channel|>`` closes (the router's walk); ``None`` when
    every thought closes."""
    pos = 0
    while True:
        i = text.find(_router.CHANNEL_OPEN, pos)
        if i < 0:
            return None
        j = text.find(_router.CHANNEL_CLOSE, i + len(_router.CHANNEL_OPEN))
        if j < 0:
            return text[i:]
        pos = j + len(_router.CHANNEL_CLOSE)


def pass1_fields(
    text: str, fmt: str, tools: _tools.ToolSet, gt: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The routed and scored fields of a pass-1 row from its decoded ``text`` (live and ``rescore`` alike).

    cc: as T-0080b/T-0082a (any ``toolcall`` tag parsed by the JSON scorer). native (T-0084c): a ``native`` route is
    parsed by ``native.parse_call``; another ``toolcall`` tag is ``wrong_tag`` (called, invalid, not exact; QN7);
    plus ``text_after_close`` (the answer after ``<tool_call|>``, ``None`` without one), ``spoken_prefix_chars`` and
    ``open_thought_call`` (no route, and a ``<|tool_call>`` inside the unclosed thought; ruling C6)."""
    r = _router.route(text)
    out: dict[str, Any] = {
        "route": {
            "kind": r.kind,
            "tag": r.tag,
            "draft_name": r.draft_name,
            "spoken_prefix": r.spoken_prefix,
        },
        "draft_call_text": r.draft_call_text,
        "draft_close": r.draft_close,
    }
    if r.kind != "toolcall":
        out.update(draft_parse=None, draft_call=None, draft_schema_valid=None, draft_exact=False)
    elif fmt == "native" and r.tag != "native":
        out.update(
            draft_parse=WRONG_TAG, draft_call=None, draft_schema_valid=False, draft_exact=False
        )
    else:
        parse, call, valid, ex = score_call(r.draft_call_text, tools, gt, fmt)
        out.update(draft_parse=parse, draft_call=call, draft_schema_valid=valid, draft_exact=ex)
    if fmt == "native":
        after: str | None = None
        if r.tag == "native" and TOOL_CALL_CLOSE in r.draft_call_text:
            draft = r.draft_call_text
            after = draft[draft.index(TOOL_CALL_CLOSE) + len(TOOL_CALL_CLOSE) :]
        tail = _open_thought_text(text) if r.kind == "none" else None
        out.update(
            text_after_close=after,
            spoken_prefix_chars=len(r.spoken_prefix),
            open_thought_call=tail is not None and TOOL_CALL_OPEN in tail,
        )
    return out


def sends_pass2(p1row: Mapping[str, Any], fmt: str) -> bool:
    """Whether a pass-1 row of ``fmt`` leads to pass 2: routed ``toolcall``, with any tag, in both formats (cc as
    ``main``; a native ``wrong_tag`` goes to the native picker since H-T84-0, T-0084d criterion 7)."""
    route = p1row.get("route")
    route_map: Mapping[str, Any] = route if isinstance(route, Mapping) else {}
    return route_map.get("kind") == "toolcall"


def _prompt_ids(payload: Any) -> list[int] | None:
    """The served ``prompt_token_ids`` (top level, vLLM v0.29.0 with ``return_token_ids``); ``None`` if absent."""
    try:
        ids = payload.get("prompt_token_ids")
    except AttributeError:
        return None
    return [int(t) for t in ids] if isinstance(ids, list) else None


# --------------------------------------------------------------------------- the run


class _Run:
    def __init__(
        self, cfg: TwoPassConfig, knobs: Mapping[str, Mapping[str, float] | None], tools: Any
    ) -> None:
        self.cfg = cfg
        self.knobs = knobs
        self.tools = tools
        self.run_id = uuid.uuid4().hex[:12]
        self.log = _records.EngineLog(cfg.engine_log)
        self.flushes = 0
        self.warmups = 0
        self.model = _runner.DEFAULT_MODEL
        self.system = system_prompt(tools)
        #: T-0084c: per case id, the served prompt ids of its native pass 1 (the T1/T2 prefix guard)
        self.pass1_prompts: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ helpers
    def automaton(self, scope: str, thought: str, fmt: str = "cc") -> Any:
        from local_astra.decoding import toolcall as _tc

        if fmt == "native":
            return _tc.compile_toolcall(
                self.tools,
                scope=scope,
                thought=thought,
                whitespace=self.cfg.whitespace,
                call_format="native",
            )
        return _tc.compile_toolcall(
            self.tools, scope=scope, thought=thought, whitespace=self.cfg.whitespace
        )

    def pass1_body(self, case: Mapping[str, Any], fmt: str) -> dict[str, Any]:
        if fmt == "native":
            return native_pass1_body(self.model, case, Path(self.cfg.set_dir), self.tools)
        return pass1_body(self.model, self.system, case, Path(self.cfg.set_dir))

    def prompt_guard(
        self, payload: Any, thought: str | None, case_id: str
    ) -> tuple[str | None, list[int] | None]:
        """T-0084c criterion 6 on a native response: ``(reason, served prompt ids)``, reason ``None`` when the guard
        passes. ``thought`` None is the native pass 1 (5 x 46 and 5 x 47, one per declaration); T1/T2 must equal the
        case's native pass-1 prompt; T4 must end in 48."""
        prompt = _prompt_ids(payload)
        if prompt is None:
            return "missing", None
        if thought is None:
            n = len(self.tools.declarations)
            if prompt.count(DECLARATION_OPEN_ID) != n or prompt.count(DECLARATION_CLOSE_ID) != n:
                return "declarations", prompt
            return None, prompt
        if thought == "T4":
            ok = bool(prompt) and prompt[-1] == TOOL_CALL_OPEN_ID
            return (None if ok else "continuation"), prompt
        return (None if prompt == self.pass1_prompts.get(case_id) else "prefix"), prompt

    def flush(self) -> None:
        """Ruling E9 (as the T-0070 runner): a tiny unconstrained request, never a row."""
        from local_astra.eval.cd import prompts

        body = prompts.request_body(self.model, _runner.FLUSH_PROMPT)
        body["max_tokens"] = 1
        body.pop("return_token_ids", None)
        _runner._post(self.cfg.base_url, body, self.cfg.request_timeout_s)
        self.flushes += 1

    def await_records(self, tag: str, max_tokens: int) -> tuple[_records.Joined | None, int]:
        joined = self.log.wait(tag, max_tokens=max_tokens, timeout_s=self.cfg.record_wait_s)
        sent = 0
        while sent < MAX_FLUSHES and (
            joined is None or not _records.is_complete(joined.records, max_tokens)
        ):
            self.flush()
            sent += 1
            joined = self.log.wait(tag, max_tokens=max_tokens, timeout_s=self.cfg.flush_wait_s)
        return joined, sent

    # ------------------------------------------------------------------ pass 1
    def pass1(
        self, case: Mapping[str, Any], body: Mapping[str, Any], fmt: str = "cc"
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "pass": 1,
            "case": case["id"],
            "kind": case.get("kind"),
            "tool": case.get("tool"),
            "error": None,
            "format": fmt,
        }
        before = _metrics(self.cfg.base_url)
        payload, err, wall = _post(self.cfg.base_url, body, self.cfg.request_timeout_s)
        after = _metrics(self.cfg.base_url)
        row["wall_ms"] = wall
        row["prefix_hit_tokens"], row["prefix_query_tokens"] = _delta(before, after)
        if err is not None:
            row["error"] = err
            row["route"] = {"kind": "none", "tag": None, "draft_name": None, "spoken_prefix": ""}
            return row
        if fmt == "native":
            reason, prompt = self.prompt_guard(payload, None, str(case["id"]))
            row["prompt_len"] = len(prompt) if prompt is not None else None
            if reason is not None:
                row["error"] = PROMPT_GUARD
                row[PROMPT_GUARD] = reason
                row["route"] = {
                    "kind": "none",
                    "tag": None,
                    "draft_name": None,
                    "spoken_prefix": "",
                }
                return row
            self.pass1_prompts[str(case["id"])] = list(prompt or [])
        ids = _token_ids(payload)
        if ids is None:
            row["error"] = "no_token_ids"
            row["route"] = {"kind": "none", "tag": None, "draft_name": None, "spoken_prefix": ""}
            return row
        text = _decode(ids)
        row.update(token_ids=ids, text=text, **pass1_fields(text, fmt, self.tools, case.get("gt")))
        return row

    # ------------------------------------------------------------------ the replay's warm-up (T-0082b R10)
    def warmup(self, case: Mapping[str, Any], body: Mapping[str, Any]) -> dict[str, Any]:
        """Ruling R10 (human): on a replay, before a routed case's first pass-2 request, the case's own pass-1 body
        with ``max_tokens`` 1, so the prefix cache is in the state a live pass 1 leaves. Logged as a
        ``"pass": "warmup"`` row (never scored, outside every policy)."""
        warm = dict(body)
        warm["max_tokens"] = 1
        before = _metrics(self.cfg.base_url)
        _payload, err, wall = _post(self.cfg.base_url, warm, self.cfg.request_timeout_s)
        after = _metrics(self.cfg.base_url)
        self.warmups += 1
        row: dict[str, Any] = {"pass": "warmup", "case": case["id"], "error": err, "wall_ms": wall}
        row["prefix_hit_tokens"], row["prefix_query_tokens"] = _delta(before, after)
        return row

    # ------------------------------------------------------------------ one constrained request
    def request(
        self,
        base: Mapping[str, Any],
        cell: Cell,
        scope: str,
        spoken: str,
        tag: str,
        seed: Sequence[int] | None = None,
        case_id: str = "",
    ) -> dict[str, Any]:
        """One pass-2 request under ``scope``'s automaton, seeded with ``seed`` when given (T-0081b). A native cell's
        request uses the native automaton and body and is checked against its served prompt (T-0084c)."""
        from local_astra.decoding import artifact as _artifact
        from local_astra.decoding import report as _report
        from local_astra.eval.cd import replay

        a = self.automaton(scope, cell.thought, cell.fmt)
        out: dict[str, Any] = {
            "scope": scope,
            "record_tag": tag,
            "bucket": int(a.n_states_bucket),
            "n_states": int(a.n_states),
            "error": None,
            "cs": False,
            "steps": None,
            "wall_ms": 0.0,
            "prefix_hit_tokens": None,
            "prefix_query_tokens": None,
            "token_ids": None,
            "committed_ids": None,
            "text": "",
            "close_101_pos": None,
            "flush_requests": 0,
            "k0": SEED_DRAFT_START_STEP if seed is not None else 0,
        }
        try:
            _report.gate(a, self.cfg.max_states)
        except _report.TooLarge:
            out["error"] = "too_large"
            return out
        payload = _artifact.encode_automaton(a)
        arm = base_arm(cell.arm)
        try:
            xargs = _arms.cd_xargs(arm, payload, tag, self.knobs.get(arm))
        except ValidationError:
            out["error"] = "cd_request_invalid"
            return out
        # before the draft keys are added: the row stores the seed once, as ids (``seed_ids``)
        out["knobs"] = _arms.row_knobs(xargs)
        if seed is not None:
            try:
                xargs = seeded_xargs(xargs, seed)
            except ValidationError:
                out["error"] = "seed_invalid"  # a stop (criterion 6c): never sent unseeded
                return out
        body = pass2_body(base, cell.thought, spoken, xargs, cell.fmt)
        if cell.thought == "T4":
            # T-0085 criterion 3: before the request is sent, on the exact string pass2_body appended
            reason = continuation_guard(
                str(body["messages"][-1]["content"]), CC if cell.fmt == "cc" else TOOL_CALL_OPEN
            )
            if reason is not None:
                out["error"] = CONTINUATION_GUARD
                out[CONTINUATION_GUARD] = reason
                return out
        max_tokens = int(body["max_tokens"])
        before = _metrics(self.cfg.base_url)
        resp, err, wall = _post(self.cfg.base_url, body, self.cfg.request_timeout_s)
        after = _metrics(self.cfg.base_url)
        out["wall_ms"] = wall
        out["prefix_hit_tokens"], out["prefix_query_tokens"] = _delta(before, after)
        if err is not None:
            out["error"] = err
            return out
        if cell.fmt == "native":
            # before any flush: a failing guard is the last request sent (criterion 6)
            reason, _prompt = self.prompt_guard(resp, cell.thought, case_id)
            if reason is not None:
                out["error"] = PROMPT_GUARD
                out[PROMPT_GUARD] = reason
                return out
        ids = _token_ids(resp)
        out["token_ids"] = ids
        joined, flushes = self.await_records(tag, max_tokens)
        out["flush_requests"] = flushes
        if joined is None or not joined.valid:
            out["error"] = "join_error"
            return out
        out["steps"] = executed_steps(joined.steps, out["k0"])
        out["bucket"] = joined.bucket if joined.bucket is not None else out["bucket"]
        committed = joined.committed_ids
        out["committed_ids"] = committed
        out["in_final"] = joined.in_final
        if joined.error is not None:
            out["error"] = joined.error
            return out
        if not _records.is_complete(joined.records, max_tokens):
            out["error"] = "join_error"
            return out
        if ids is None:
            out["error"] = "no_token_ids"
            return out
        if _until_stop(ids) != _until_stop(committed):
            out["error"] = "join_error"  # the served ids are not what the records committed
            return out
        accepted = replay.accepts(a, committed)
        out["cs"] = accepted
        out["terminated"] = replay.terminated(committed)
        out["text"] = _decode(ids)
        head = _until_stop(committed)
        out["close_101_pos"] = head.index(CHANNEL_CLOSE_ID) if CHANNEL_CLOSE_ID in head else None
        return out

    # ------------------------------------------------------------------ pass 2 of one cell
    def pass2(
        self,
        case: Mapping[str, Any],
        p1row: Mapping[str, Any],
        body: Mapping[str, Any],
        cell: Cell,
        tag_base: str,
    ) -> tuple[dict[str, Any], str | None]:
        route = p1row["route"]
        spoken = str(route.get("spoken_prefix") or "")
        reqs: list[dict[str, Any]] = []
        draft = str(p1row.get("draft_call_text") or "")
        if cell.fmt == "cc":
            named = route.get("draft_name")
        elif p1row.get("draft_parse") == WRONG_TAG:
            named = None  # T-0084d criterion 7: no native draft name exists -> the native picker
        else:
            named = _native.draft_name(draft)
        if cell.scope == "union":
            scopes = ["union"]
        elif named:
            scopes = [str(named)]
        else:
            scopes = ["name"]
        seed: list[int] | None = None
        seed_row: dict[str, Any] = {}
        if cell.arm == SEED_ARM:
            if scopes[0] == "name":
                seed_row = {"seed": None, "seed_reason": "picker"}
            else:
                p1_ids = [int(t) for t in p1row.get("token_ids") or []]
                seed = [
                    int(t)
                    for t in seed_ids(cell.thought, p1_ids, draft, CANVAS_LENGTH, fmt=cell.fmt)
                ]
                seed_row = {
                    "seed": True,
                    "seed_reason": None,
                    "seed_ids": seed,
                    "seed_len": len(seed),
                    "seed_source": seed_source(cell.thought, p1_ids),
                    "seed_trimmed": len(draft) - len(draft.rstrip()),  # R2: characters trimmed
                    "seed_noise": SEED_DRAFT_NOISE,
                    "seed_start_step": SEED_DRAFT_START_STEP,
                }
        stop: str | None = None
        while True:
            scope = scopes[len(reqs)]
            req = self.request(
                body,
                cell,
                scope,
                spoken,
                f"{tag_base}.{len(reqs)}",
                seed if not reqs else None,  # only the routed request is seeded, never the picker's
                str(case["id"]),
            )
            reqs.append(req)
            stop = self.check(req, case["id"], cell.name)
            if stop is not None or req["error"] is not None:
                break
            if scope == "name":
                if cell.fmt == "native":
                    picked = _native.draft_name(call_text_of(req["text"], "native"))
                else:
                    picked = _router.route(CC + call_text_of(req["text"])).draft_name
                if picked is None:
                    req["error"] = "picker_no_name"
                    break
                scopes.append(picked)
                continue
            break
        final = reqs[-1]
        error = next((r["error"] for r in reqs if r["error"] is not None), None)
        if error is None:
            parse, call, valid, ex = score_call(
                call_text_of(final["text"], cell.fmt), self.tools, case.get("gt"), cell.fmt
            )
        else:
            parse, call, valid, ex = "error", None, False, False
        hits = [r["prefix_hit_tokens"] for r in reqs]
        queries = [r["prefix_query_tokens"] for r in reqs]
        steps = [s for r in reqs for s in (r["steps"] or [])]
        row: dict[str, Any] = {
            "pass": 2,
            "case": case["id"],
            "format": cell.fmt,
            "cell": cell.name,
            "thought": cell.thought,
            "scope": cell.scope,
            "arm": cell.arm,
            "scopes": [r["scope"] for r in reqs],
            "requests": len(reqs),
            "record_tags": [r["record_tag"] for r in reqs],
            "text": final["text"],
            "texts": [r["text"] for r in reqs],
            "committed_ids": [r["committed_ids"] for r in reqs],
            "parse": parse,
            "call": call,
            "schema_valid": valid,
            "exact": ex,
            "cs": all(bool(r["cs"]) for r in reqs),
            "error": error,
            "wall_ms": sum(float(r["wall_ms"]) for r in reqs),
            "wall_ms_per_request": [float(r["wall_ms"]) for r in reqs],
            "steps": steps,
            "steps_per_request": [r["steps"] for r in reqs],
            "bucket": final["bucket"],
            "buckets": [r["bucket"] for r in reqs],
            "n_states": [r["n_states"] for r in reqs],
            "close_101_pos": final["close_101_pos"] if cell.thought == "T1" else None,
            "prefix_hit_tokens": None if any(h is None for h in hits) else sum(hits),
            "prefix_query_tokens": None if any(q is None for q in queries) else sum(queries),
            "flush_requests": sum(int(r["flush_requests"]) for r in reqs),
            "knobs": final.get("knobs"),
            **seed_row,
        }
        return row, stop

    def check(self, req: Mapping[str, Any], case_id: str, cell: str) -> str | None:
        err = req["error"]
        where = f"case {case_id}, cell {cell}, scope {req['scope']}"
        if err == PROMPT_GUARD:
            return f"{PROMPT_GUARD}: {req.get(PROMPT_GUARD)} (the served prompt; {where})"
        if err == CONTINUATION_GUARD:
            return (
                f"{CONTINUATION_GUARD}: {req.get(CONTINUATION_GUARD)} "
                f"(the T4 continuation, not sent; {where})"
            )
        if err in ENGINE_ERRORS:
            return f"engine: {err} ({where})"
        if err == "no_token_ids":
            return f"engine: no return_token_ids in the response ({where})"
        if err == "join_error":
            return f"join_error: the ASTRA_CD records of {req['record_tag']} do not join ({where})"
        seeded = int(req.get("k0") or 0) > 0
        if err == "seed_invalid":
            return f"seed_invalid: the CdRequest contract refuses the draft seed ({where})"
        if seeded and isinstance(err, str) and (err.startswith("draft_") or err == "missing_key"):
            return f"seed refused: the plugin's record reason {err!r} on a seeded request ({where})"
        if err in LOUD_ERRORS:
            return f"loud CD error {err!r} ({where})"
        if err is None:
            steps = req.get("steps") or []
            if seeded and steps and int(steps[0]) <= 0:
                return (
                    f"seed_accounting: canvas 0 committed at step {int(steps[0]) + int(req['k0'])} "
                    f"<= the start step {req['k0']} ({where})"
                )
            replay_ok = bool(req["cs"])
            if req.get("in_final") is not None and replay_ok != bool(req["in_final"]):
                return (
                    f"cs: replay {'accepts' if replay_ok else 'rejects'} but the plugin says "
                    f"in_final={req['in_final']} ({where})"
                )
            if req.get("terminated") and not replay_ok:
                return f"cs: terminated output rejected by the host replay, CS != 1 ({where})"
        return None


def _knobs(cfg: TwoPassConfig, cells: Sequence[Cell]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Per base arm: the frozen knobs (``None`` for an arm without calibration) and their provenance. The seed
    arm has no calibration of its own (an entry keyed by it is refused); it uses ``perstep_mar``'s."""
    if SEED_ARM in cfg.calibration:
        raise ValueError(
            f"{SEED_ARM} has no calibration of its own: its knobs are {SEED_BASE_ARM}'s frozen file"
        )
    knobs: dict[str, Any] = {}
    prov: dict[str, Any] = {}
    for arm in dict.fromkeys(base_arm(c.arm) for c in cells):
        spec = _arms.spec(arm)
        if not spec.needs_calibration:
            knobs[arm] = None
            continue
        path = cfg.calibration.get(arm)
        if path is None:
            raise ValueError(
                f"{arm} needs a frozen calibration file (docs/cd/calibration-bfcl*.json)"
            )
        values, sha = _calibration.load_frozen(path, task="bfcl", arm=arm)
        knobs[arm] = values
        prov[arm] = {"path": str(path), "sha256": sha, **values}
    return knobs, prov


def bucket_histogram(
    cells: Sequence[Cell], cases: Sequence[Mapping[str, Any]], tools: Any, compile_: Any
) -> dict[str, Any]:
    """Criterion 4: per thought, the bucket of every (scope, thought) the cells can send, and the histogram
    over the stage's tool cases by their labelled tool (the named scope's real bucket is known only after
    pass 1). Native cells (T-0084c) are keyed ``native/<thought>`` and compiled as ``compile_(scope, thought,
    "native")``."""
    out: dict[str, Any] = {}
    for fmt, thought in dict.fromkeys((c.fmt, c.thought) for c in cells):
        mine = [c for c in cells if c.fmt == fmt and c.thought == thought]
        scopes: list[str] = []
        if any(c.scope == "named" for c in mine):
            scopes += ["name", *tools.names]
        if any(c.scope == "union" for c in mine):
            scopes.append("union")
        if fmt == "cc":
            per_scope = {s: int(compile_(s, thought).n_states_bucket) for s in scopes}
        else:
            per_scope = {s: int(compile_(s, thought, fmt).n_states_bucket) for s in scopes}
        hist: Counter[int] = Counter()
        for case in cases:
            tool = case.get("tool")
            if case.get("kind") == "tool" and tool in per_scope:
                hist[per_scope[str(tool)]] += 1
        out[thought if fmt == "cc" else NATIVE_PREFIX + thought] = {
            "scope_buckets": per_scope,
            "labelled_tool_histogram": {str(b): hist[b] for b in sorted(hist)},
        }
    return out


def _print_histogram(hist: Mapping[str, Any]) -> None:
    for thought, h in hist.items():
        print(
            f"twopass: bucket histogram thought={thought} scope buckets {h['scope_buckets']} "
            f"labelled-tool cases per bucket {h['labelled_tool_histogram']}",
            flush=True,
        )


def run(cfg: TwoPassConfig) -> TwoPassResult:
    """Run one stage. ``ValueError`` (before any socket) for a non-loopback URL, an unknown stage or cell,
    a missing or wrong frozen calibration file, or an eval set that does not match its MANIFEST."""
    _runner.check_loopback(cfg.base_url)
    if cfg.stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {cfg.stage!r}")
    if not cfg.cells:
        raise ValueError("no cells")
    cells = [parse_cell(c) for c in cfg.cells]
    if len({c.name for c in cells}) != len(cells):
        raise ValueError(f"duplicate cells in {list(cfg.cells)}")
    replay = cfg.pass1_from is not None or cfg.pass1_sha256 is not None
    if replay and any(c.fmt == "native" for c in cells):
        raise ValueError(
            "a replay (T-0082b) re-uses cc pass-1 rows only: native cells need a live native pass 1"
        )
    knobs, calib = _knobs(cfg, cells)
    tools = _tools.load()
    system = system_prompt(tools)
    cases, cases_sha = _cases(cfg)
    replayed: dict[str, dict[str, Any]] | None = None
    source: dict[str, Any] = {}
    if replay:
        replayed, source = _replay_source(cfg, calib, tools, system, cases, cases_sha)
    t_start = time.monotonic()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.out_dir / f"{_timestamp()}-toolcall-{cfg.stage}.jsonl"
    while path.exists():
        time.sleep(0.001)
        path = cfg.out_dir / f"{_timestamp()}-toolcall-{cfg.stage}.jsonl"
    regime: dict[str, Any] = {
        "kind": "regime",
        "task": "toolcall",
        "stage": cfg.stage,
        "cells": [c.name for c in cells],
        "commit": _runner._git(cfg.repo, "rev-parse", "HEAD") or None,
        "expected_commit": cfg.expected_commit,
        "system_prompt": system,
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "tools_path": str(tools.path),
        "tools_sha256": tools.version,
        "cases_sha256": cases_sha,
        "set_dir": str(cfg.set_dir),
        "n_cases": len(cases),
        "n_tool_cases": sum(1 for c in cases if c.get("kind") == "tool"),
        "limit": cfg.limit,
        "calibration": calib,
        "knobs": {a: _arms.arm_knobs(a, knobs[a]) for a in knobs},
        "enable_thinking": True,
        "pass1_max_tokens": PASS1_MAX_TOKENS,
        "pass2_max_tokens": PASS2_MAX_TOKENS,
        "whitespace": cfg.whitespace,
        "max_states": cfg.max_states,
        "concurrency": 1,
        "seed": None,
        "request_timeout_s": cfg.request_timeout_s,
        "started_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    regime.update(source)  # T-0082b: a replay names its source; a live run adds nothing
    if any(c.fmt == "native" for c in cells):  # T-0084c; a cc-only regime is unchanged
        regime["native"] = {
            "system_prompt": NATIVE_SYSTEM_PROMPT,
            "system_prompt_sha256": hashlib.sha256(NATIVE_SYSTEM_PROMPT.encode()).hexdigest(),
            "tools_field_sha256": hashlib.sha256(tools_field_json(tools).encode()).hexdigest(),
            "tools_field_canonical_json": "sort_keys, separators (',', ':'), ensure_ascii False",
            "tool_choice": "none",
            "call_format": "native",
            "pass1_order": "cc first on even case indexes, native first on odd ones",
        }
    if any(c.arm == SEED_ARM for c in cells):
        regime["seed_draft"] = {
            "arm": SEED_ARM,
            "base_arm": SEED_BASE_ARM,
            "draft_noise": SEED_DRAFT_NOISE,
            "draft_start_step": SEED_DRAFT_START_STEP,
            "canvas_length": CANVAS_LENGTH,
            "source": "T-0081 tier 2, dad1382, mar_n0_s24",
        }
    with path.open("w") as fh:
        stop = _hardstop.check_tree(cfg.repo, cfg.expected_commit)
        if stop is not None:
            regime.update(versions=_runner._versions(None), stopped=stop)
            _write(fh, regime)
            _write(fh, {"kind": "trailer", "stopped": stop})
            return TwoPassResult(EXIT_REFUSED, path, stop, 0)
        state = _Run(cfg, knobs, tools)
        regime["run_id"] = state.run_id
        crash: str | None = None
        regime_written = False
        try:
            hist = bucket_histogram(cells, cases, tools, state.automaton)
            regime["bucket_histogram"] = hist
            _print_histogram(hist)
            served = _runner._served_model(cfg.base_url, cfg.request_timeout_s)
            regime["versions"] = _runner._versions(served)
            _write(fh, regime)
            regime_written = True
            if served is None:
                stop = (
                    f"engine: GET {cfg.base_url.rstrip('/')}/models did not answer (before pass 1)"
                )
            else:
                state.model = served
                stop = _loop(state, cfg, cells, cases, fh, replayed)
        except Exception:
            # Reviewer advisory: once the artifact is open, a failure is a stop (exit 1, trailer, report),
            # never a pre-request refusal (exit 2).
            crash = traceback.format_exc()
            stop = f"crash: {crash.strip().splitlines()[-1]}"
            print(crash, file=sys.stderr, flush=True)
            if not regime_written:
                regime.setdefault("versions", None)
                _write(fh, regime)
        trailer = {
            "kind": "trailer",
            "stopped": stop,
            "flush_requests": state.flushes,
            "malformed_records": state.log.malformed,
            "elapsed_s": time.monotonic() - t_start,
        }
        if replayed is not None:
            trailer["warmup_requests"] = state.warmups  # R10; a live trailer is unchanged
        if crash is not None:
            trailer["traceback"] = crash
        _write(fh, trailer)
    if stop is not None:
        print(f"twopass: STOPPED: {stop}", flush=True)
    return TwoPassResult(EXIT_STOPPED if stop else EXIT_OK, path, stop, len(cases))


def williams_orders(k: int) -> list[list[int]]:
    """Ruling R-W: the Williams design over ``k`` cells (indices). Row 0 is ``0, 1, k-1, 2, k-2, ...``; row ``i``
    adds ``i`` mod ``k``. For even ``k`` these ``k`` rows put every cell immediately before every other exactly
    once; for odd ``k`` the ``k`` mirrored rows are appended (``2k`` rows, each ordered pair twice). Each cell is
    also first equally often. The driver cycles the rows by routed-case count."""
    if k < 1:
        raise ValueError("no cells")
    first = [0]
    lo, hi = 1, k - 1
    while len(first) < k:
        first.append(lo)
        lo += 1
        if len(first) < k:
            first.append(hi)
            hi -= 1
    rows = [[(x + i) % k for x in first] for i in range(k)]
    if k % 2 == 1:
        rows += [list(reversed(r)) for r in rows]
    return rows


def _loop(
    state: _Run,
    cfg: TwoPassConfig,
    cells: Sequence[Cell],
    cases: Sequence[Mapping[str, Any]],
    fh: IO[str],
    replayed: Mapping[str, Mapping[str, Any]] | None = None,
) -> str | None:
    """Pass 1 then pass 2 per case. With ``replayed`` (T-0082b) pass 1 is not sent: its row is the stored,
    re-scored row of the case; the body pass 2 is built from is still ``pass1_body`` (as live).

    T-0084c: one pass 1 per format the cells use, cc first on even case indexes and native first on odd ones. The
    pass-2 order is the R-W Williams row over the full cell list, indexed by the count of cases routed in at least
    one format; a cell whose format did not route (``sends_pass2``) is skipped."""
    routed = 0
    formats = [f for f in FORMATS if any(c.fmt == f for c in cells)]
    for ci, case in enumerate(cases):
        order_f = formats if ci % 2 == 0 else list(reversed(formats))
        bodies: dict[str, dict[str, Any]] = {}
        p1s: dict[str, dict[str, Any]] = {}
        for fmt in order_f:
            body = state.pass1_body(case, fmt)
            if replayed is None:
                p1 = state.pass1(case, body, fmt)
            else:
                p1 = dict(replayed[str(case["id"])])
            _write(fh, p1)
            if p1["error"] in ENGINE_ERRORS:
                return f"engine: {p1['error']} (case {case['id']}, pass 1)"
            if p1["error"] == "no_token_ids":
                return f"engine: no return_token_ids in the pass-1 response (case {case['id']})"
            if p1["error"] == PROMPT_GUARD:
                return (
                    f"{PROMPT_GUARD}: {p1.get(PROMPT_GUARD)} (the served prompt; case {case['id']}, "
                    "native pass 1)"
                )
            bodies[fmt], p1s[fmt] = body, p1
        go = {f for f in order_f if sends_pass2(p1s[f], f)}
        if not go:
            continue
        if replayed is not None:
            warm = state.warmup(case, bodies["cc"])
            _write(fh, warm)
            if warm["error"] is not None:
                return f"engine: {warm['error']} (case {case['id']}, warm-up)"
        seqs = williams_orders(len(cells))
        order = [cells[i] for i in seqs[routed % len(seqs)]]
        routed += 1
        for cell in order:
            if cell.fmt not in go:
                continue
            j = cells.index(cell)
            tag = f"t0080b.{state.run_id}.{ci}.{j}"
            row, stop = state.pass2(case, p1s[cell.fmt], bodies[cell.fmt], cell, tag)
            _write(fh, row)
            if stop is not None:
                return stop
    return None


# --------------------------------------------------------------------------- reading artifacts


def read_artifact(
    path: Path | str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """``(regime, pass rows, other lines)``."""
    lines = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    rest = lines[1:]
    return (
        lines[0],
        [x for x in rest if x.get("pass") in (1, 2)],
        [x for x in rest if x.get("pass") not in (1, 2)],
    )


_ROUTE_KEYS = ("kind", "tag", "spoken_prefix", "draft_name")


def rescore(
    rows: Iterable[Mapping[str, Any]],
    tools: _tools.ToolSet,
    cases: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """T-0082a criterion 10: the rows, in input order, with each pass-1 row's ``route``, ``draft_call_text``,
    ``draft_close``, ``draft_parse``, ``draft_call``, ``draft_schema_valid`` and ``draft_exact`` recomputed from
    its stored ``text`` under the current router and scorer (ground truth from ``cases``, by id; ``pass1_fields`` of
    the row's own format, T-0084c). Pass-2 rows and
    pass-1 rows without a ``text`` (a pass-1 error) come back as they are.

    Raises ``ValueError`` if a stored ``text`` is not ``_decode(token_ids)``, if a case id is not in ``cases``, or
    if re-routing changes ``kind``, ``tag``, ``spoken_prefix`` or ``draft_name``."""
    by_id = {str(c["id"]): c for c in cases}
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("pass") != 1 or "text" not in row:
            out.append(dict(row))
            continue
        case_id = str(row["case"])
        text = row["text"]
        decoded = _decode([int(t) for t in row.get("token_ids") or []])
        if text != decoded:
            raise ValueError(f"case {case_id}: the stored text is not the decode of its token_ids")
        if case_id not in by_id:
            raise ValueError(f"case {case_id}: not in the cases given")
        fields = pass1_fields(text, row_format(row), tools, by_id[case_id].get("gt"))
        route = fields["route"]
        stored = row.get("route")
        stored_map: Mapping[str, Any] = stored if isinstance(stored, Mapping) else {}
        changed = [k for k in _ROUTE_KEYS if stored_map.get(k) != route[k]]
        if changed:
            raise ValueError(f"case {case_id}: re-routing the stored text changes {changed}")
        new = dict(row)
        new.update(fields)
        out.append(new)
    return out


#: T-0082b criterion 2: regime fields a replay source must share with the current run (top level; under
#: ``versions``)
REPLAY_REGIME_KEYS = (
    "system_prompt_sha256",
    "tools_sha256",
    "cases_sha256",
    "pass1_max_tokens",  # R11
    "enable_thinking",  # R11
)
REPLAY_VERSION_KEYS = ("versions_lock_sha256", "model_revision", "tokenizer_sha256")


def _replay_source(
    cfg: TwoPassConfig,
    calib: Mapping[str, Mapping[str, Any]],
    tools: _tools.ToolSet,
    system: str,
    cases: Sequence[Mapping[str, Any]],
    cases_sha: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """T-0082b: the pass-1 rows of ``cfg.pass1_from`` re-scored under the current router (``rescore``), each with
    ``replayed_from`` (the source's sha256), by case id; and the regime fields naming the source.

    ``ValueError`` (before any socket) when: only one of ``pass1_from``/``pass1_sha256`` is given; the file's sha256
    is not ``pass1_sha256``; it has no regime line or no trailer; its ``stage`` is not ``cfg.stage``; any line has
    ``stopped`` set; a pass-1 row has an ``error``; a field of ``REPLAY_REGIME_KEYS``/``REPLAY_VERSION_KEYS`` or the
    calibration sha256 of an arm the cells use differs from this run's; its pass-1 cases are not this stage's
    cases at ``cfg.limit``, in order (ruling R4: the source's limit); or ``rescore`` refuses."""
    if cfg.pass1_from is None or not cfg.pass1_sha256:
        raise ValueError("a replay needs both pass1_from and pass1_sha256")
    path = Path(cfg.pass1_from)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != cfg.pass1_sha256:
        raise ValueError(f"{path}: sha256 {sha} != the pinned {cfg.pass1_sha256}")
    lines = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if not lines or lines[0].get("kind") != "regime":
        raise ValueError(f"{path}: no regime line")
    regime, rest = lines[0], lines[1:]
    native_rows = [
        str(x.get("case")) for x in rest if x.get("pass") in (1, 2) and row_format(x) == "native"
    ]
    if native_rows:
        raise ValueError(
            f"{path}: rows of the native format (cases {sorted(set(native_rows))}); a replay re-uses cc "
            "pass-1 rows only (T-0084c criterion 10)"
        )
    if regime.get("stage") != cfg.stage:
        raise ValueError(f"{path}: stage {regime.get('stage')!r} != {cfg.stage!r}")
    stopped = [x.get("stopped") for x in lines if x.get("stopped")]
    if stopped:
        raise ValueError(f"{path}: a stopped run ({stopped[0]})")
    if not any(x.get("kind") == "trailer" for x in rest):
        raise ValueError(f"{path}: no trailer (an incomplete run)")
    versions = _runner._versions(None)
    current: dict[str, Any] = {
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "tools_sha256": tools.version,
        "cases_sha256": cases_sha,
        "pass1_max_tokens": PASS1_MAX_TOKENS,
        "enable_thinking": True,
    }
    stored_versions = regime.get("versions")
    stored_v: Mapping[str, Any] = stored_versions if isinstance(stored_versions, Mapping) else {}
    differ = [k for k in REPLAY_REGIME_KEYS if regime.get(k) != current[k]]
    differ += [k for k in REPLAY_VERSION_KEYS if stored_v.get(k) != versions.get(k)]
    stored_calib = regime.get("calibration")
    stored_c: Mapping[str, Any] = stored_calib if isinstance(stored_calib, Mapping) else {}
    absent: list[str] = []
    for arm, prov in calib.items():
        entry = stored_c.get(arm)
        if entry is None:
            absent.append(
                arm
            )  # R9: pass-1 rows do not depend on calibration; recorded, not refused
        elif not isinstance(entry, Mapping) or entry.get("sha256") != prov.get("sha256"):
            differ.append(f"calibration[{arm}].sha256")
    if differ:
        raise ValueError(f"{path}: the regime differs from this run's in {differ}")
    p1 = [x for x in rest if x.get("pass") == 1]
    errors = [str(x.get("case")) for x in p1 if x.get("error")]
    if errors:
        raise ValueError(f"{path}: pass-1 rows with an error: {errors}")
    rows = rescore(p1, tools, cases)
    got = [str(r["case"]) for r in rows]
    want = [str(c["id"]) for c in cases]
    if got != want:
        raise ValueError(
            f"{path}: its pass-1 cases {got} are not this stage's cases {want} "
            f"(replay with the source's limit, {regime.get('limit')})"
        )
    named = {
        "pass1_from": str(path),
        "pass1_sha256": sha,
        "pass1_source": {
            k: regime.get(k) for k in ("commit", "run_id", "started_utc", "limit", "cells")
        },
        "calibration_absent_in_source": absent,
    }
    return {str(r["case"]): {**r, "format": "cc", "replayed_from": sha} for r in rows}, named


# --------------------------------------------------------------------------- derived policies


DRAFT_ONLY_NATIVE = "draft_only:native"


def _policy_parts(policy: str) -> tuple[str, str | None]:
    """``(kind, cell)``: kind ``draft_only``, ``cd_always`` or ``cd_on_invalid``."""
    if policy in ("draft_only", DRAFT_ONLY_NATIVE):
        return "draft_only", None
    for kind in ("cd_on_invalid", "cd_always"):
        if policy.startswith(kind + ":"):
            return kind, policy[len(kind) + 1 :]
    if policy.count("/") == (3 if cell_format(policy) == "native" else 2):
        return "cd_always", policy
    raise ValueError(f"unknown policy {policy!r}")


def _policy_format(policy: str) -> str:
    """The format whose pass-1 rows a policy uses (T-0084c criterion 8): ``draft_only`` cc,
    ``draft_only:native`` native, a cell policy its cell's."""
    _kind, cell = _policy_parts(policy)
    if cell is None:
        return "native" if policy == DRAFT_ONLY_NATIVE else "cc"
    return cell_format(cell)


def derive(rows: Iterable[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    """Per-case rows of ``policy`` from pass-1 and pass-2 artifact rows, offline (no request).

    - ``draft_only``: pass 1's draft parsed and scored; no pass 2; e2e wall = pass 1's. ``draft_only:native``:
      the same over the native pass-1 rows (T-0084c).
    - ``cd_always:<cell>`` (or the bare cell): the cell's pass-2 result on every routed case; e2e = pass 1 +
      pass 2.
    - ``cd_on_invalid:<cell>``: the draft if schema-valid, else the cell's pass-2 result (latency adds pass 2
      only on invalid drafts).

    A case without a call (route not ``toolcall``) has ``called`` False, ``schema_valid`` None, ``exact`` False.
    A routed case whose pass-2 row is missing (a stopped run) has ``missing`` True and is not schema-valid.

    T-0084c: a policy reads only its own format's pass-1 rows (``row_format``; rows without ``format`` are cc), so
    a cell is paired with its own format's draft. A native ``wrong_tag`` draft is an invalid draft (T-0084d criterion
    7, amending ruling C8): ``draft_only:native`` keeps it called, invalid and not exact; the cell policies take its
    pass-2 row, and without one it is ``missing`` (ruling D8).
    """
    kind, cell = _policy_parts(policy)
    fmt = _policy_format(policy)
    rows = list(rows)
    p1 = [r for r in rows if r.get("pass") == 1 and row_format(r) == fmt]
    p2 = {(r["case"], r["cell"]): r for r in rows if r.get("pass") == 2}
    out: list[dict[str, Any]] = []
    for r in p1:
        route = r.get("route")
        rkind = route.get("kind") if isinstance(route, Mapping) else route
        called = rkind == "toolcall"
        tool_case = r.get("kind") == "tool"
        wall1 = float(r.get("wall_ms") or 0.0)
        row: dict[str, Any] = {
            "policy": policy,
            "case": r["case"],
            "tool": r.get("tool"),
            "tool_case": tool_case,
            "called": called,
            "schema_valid": None,
            "exact": False,
            "parsed": False,
            "call": None,
            "cs": None,
            "pass2_wall_ms": None,
            "e2e_wall_ms": wall1,
            "steps": 0.0,
            "pass2_sent": False,
            "missing": False,
        }
        if called:
            draft_valid = bool(r.get("draft_schema_valid"))
            use_pass2 = kind == "cd_always" or (kind == "cd_on_invalid" and not draft_valid)
            if not use_pass2:
                row.update(
                    schema_valid=draft_valid,
                    exact=bool(r.get("draft_exact")) and tool_case,
                    parsed=r.get("draft_parse") == "ok",
                    call=r.get("draft_call"),
                )
            else:
                q = p2.get((r["case"], cell))
                if q is None:
                    row.update(schema_valid=False, missing=True, cs=False)
                else:
                    wall2 = float(q.get("wall_ms") or 0.0)
                    row.update(
                        schema_valid=bool(q.get("schema_valid")),
                        exact=bool(q.get("exact")) and tool_case,
                        parsed=q.get("parse") == "ok",
                        call=q.get("call"),
                        cs=bool(q.get("cs")),
                        pass2_wall_ms=wall2,
                        e2e_wall_ms=wall1 + wall2,
                        steps=float(sum(int(s) for s in (q.get("steps") or []))),
                        pass2_sent=True,
                        error=q.get("error"),
                        prefix_hit_tokens=q.get("prefix_hit_tokens"),
                        prefix_query_tokens=q.get("prefix_query_tokens"),
                        close_101_pos=q.get("close_101_pos"),
                        requests=q.get("requests"),
                    )
        out.append(row)
    return out


# --------------------------------------------------------------------------- the pre-registered choice


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs))


def _p90(xs: Sequence[float]) -> float:
    """Nearest rank: the ceil(0.9 n)-th smallest value."""
    s = sorted(xs)
    return float(s[max(0, math.ceil(0.9 * len(s)) - 1)])


def _group(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r["policy"]), []).append(r)
    return out


def _schema_rate_one(rows: Sequence[Mapping[str, Any]]) -> bool:
    called = [r for r in rows if r.get("called")]
    return bool(called) and all(r.get("schema_valid") is True for r in called)


def _exact_rate(rows: Sequence[Mapping[str, Any]]) -> Fraction:
    tool = [r for r in rows if r.get("tool_case")]
    if not tool:
        return Fraction(0)
    return Fraction(sum(1 for r in tool if r.get("exact")), len(tool))


def _mean_steps(rows: Sequence[Mapping[str, Any]]) -> Fraction:
    if not rows:
        return Fraction(0)
    return sum((Fraction(float(r.get("steps") or 0.0)) for r in rows), Fraction(0)) / len(rows)


_THOUGHT_ORDER = {"T4": 0, "T2": 1, "T1": 2}
_KIND_ORDER = {"draft_only": 0, "cd_on_invalid": 1, "cd_always": 2}
_ARM_ORDER = {"end_projection": 0, "perstep_mar": 1, "perstep_mar_qargmax": 2}
_STAGE1 = re.compile(r"\A(T1|T2|T4)/named/perstep_mar\Z")


def _accuracy_filter(
    eligible: Mapping[str, Sequence[Mapping[str, Any]]], tol: Fraction
) -> dict[str, Sequence[Mapping[str, Any]]]:
    acc = {p: _exact_rate(rs) for p, rs in eligible.items()}
    best = max(acc.values())
    return {p: rs for p, rs in eligible.items() if acc[p] >= best - tol}


def select(rows: Iterable[Mapping[str, Any]], rule: str) -> str:
    """The pre-registered selection of T-0080c on policy rows (``derive`` output); returns the policy name.

    ``stage1``: candidates are the cells ``<T1|T2|T4>/named/perstep_mar`` (K-D2; ``union`` and ``draft_only``
    are reported, never chosen). Eligible = schema_valid 1 over the called rows. Keep exact (over tool cases)
    >= best eligible - 1/10. Lowest median pass-2 wall; among cells whose median <= 1.05 x that minimum, fewer
    mean steps (over rows with a pass 2); then T4, T2, T1.

    ``stage2``: candidates ``draft_only``, ``cd_always:<...>``, ``cd_on_invalid:<...>``. Eligible = schema_valid
    1 over the called rows and every constrained row (``cs`` not None) has ``cs`` True. Keep exact >= best
    eligible - 1/20. Lowest p50 end-to-end wall over tool cases; among policies whose p50 <= 1.05 x that
    minimum, the lower p90 (nearest rank), then fewer mean steps (over tool cases), then ``draft_only`` <
    ``cd_on_invalid`` < ``cd_always`` and ``end_projection`` < ``perstep_mar`` < ``perstep_mar_qargmax``.

    All comparisons are exact (``Fraction``). ``ValueError``: unknown rule, or no eligible policy.
    """
    groups = _group(rows)
    if rule == "stage1":
        cand = {p: rs for p, rs in groups.items() if _STAGE1.match(p)}
        eligible = {p: rs for p, rs in cand.items() if _schema_rate_one(rs)}
        if not eligible:
            raise ValueError(f"stage1: no eligible cell among {sorted(groups)}")
        kept = _accuracy_filter(eligible, STAGE1_TOLERANCE)
        med: dict[str, Fraction] = {}
        steps: dict[str, Fraction] = {}
        for p, rs in kept.items():
            sent = [r for r in rs if r.get("pass2_wall_ms") is not None]
            if not sent:
                continue
            med[p] = Fraction(_median([float(r["pass2_wall_ms"]) for r in sent]))
            steps[p] = _mean_steps(sent)
        if not med:
            raise ValueError("stage1: no kept cell sent a pass 2")
        lo = min(med.values())
        within = [p for p in med if med[p] <= WALL_TOLERANCE * lo]
        return min(within, key=lambda p: (steps[p], _THOUGHT_ORDER[p.split("/")[0]]))
    if rule == "stage2":
        cand = {
            p: rs
            for p, rs in groups.items()
            if p == "draft_only" or p.startswith(("cd_always:", "cd_on_invalid:"))
        }
        eligible = {
            p: rs
            for p, rs in cand.items()
            if _schema_rate_one(rs) and all(r.get("cs") is not False for r in rs)
        }
        if not eligible:
            raise ValueError(f"stage2: no eligible policy among {sorted(groups)}")
        kept = _accuracy_filter(eligible, STAGE2_TOLERANCE)
        p50: dict[str, Fraction] = {}
        p90: dict[str, Fraction] = {}
        st: dict[str, Fraction] = {}
        for p, rs in kept.items():
            tool = [r for r in rs if r.get("tool_case")]
            if not tool:
                continue
            walls = [float(r["e2e_wall_ms"]) for r in tool]
            p50[p], p90[p], st[p] = (
                Fraction(_median(walls)),
                Fraction(_p90(walls)),
                _mean_steps(tool),
            )
        if not p50:
            raise ValueError("stage2: no kept policy has tool cases")
        lo = min(p50.values())
        within = [p for p in p50 if p50[p] <= WALL_TOLERANCE * lo]

        def order(p: str) -> tuple[int, int]:
            kind, cell = _policy_parts(p)
            arm = cell.split("/")[-1] if cell else ""
            return _KIND_ORDER[kind], _ARM_ORDER.get(arm, len(_ARM_ORDER))

        return min(within, key=lambda p: (p90[p], st[p], order(p), p))
    raise ValueError(f"unknown rule {rule!r}; one of stage1, stage2")


# --------------------------------------------------------------------------- the format comparison (T-0084d)

#: T-0084d criterion 1 (approved at H-T84-0): step 2's accuracy tolerance per stage, exact
FORMAT_TOLERANCE = {"dev": Fraction(1, 10), "test": Fraction(1, 20)}
#: the last tie-break of step 2: format cc before native (QN2, "final-level ties go to cc")
_FORMAT_ORDER = {"cc": 0, "native": 1}
#: the cc thought of the comparison (T-0080c's stage-1 choice; QN1)
FORMAT_CC_THOUGHT = "T1"
FORMAT_RULE = "T-0084d"


@dataclasses.dataclass(frozen=True)
class FormatChoice:
    """``select_format``'s result: native thought θ_n, the chosen policy, its format and the two steps' tables."""

    native_thought: str
    choice: str
    winner: str
    tables: dict[str, Any]


def format_family(native_thought: str) -> list[str]:
    """Step 2's family: ``draft_only``, ``draft_only:native`` and ``cd_always:``/``cd_on_invalid:`` over cc
    ``T1/named/{4 arms}`` and ``native/θ_n/named/{4 arms}``."""
    if native_thought not in THOUGHTS:
        raise ValueError(f"native_thought must be one of {THOUGHTS}, not {native_thought!r}")
    cells = [f"{FORMAT_CC_THOUGHT}/named/{a}" for a in CELL_ARMS]
    cells += [f"{NATIVE_PREFIX}{native_thought}/named/{a}" for a in CELL_ARMS]
    return [
        "draft_only",
        DRAFT_ONLY_NATIVE,
        *[f"{k}:{c}" for c in cells for k in ("cd_always", "cd_on_invalid")],
    ]


def _bare(policy: str) -> str:
    """A native policy name in ``select``'s cc grammar (``draft_only:native`` -> ``draft_only``, ``native/`` dropped
    from the cell). Names are unique within one format, so the map back is a dict."""
    if policy == DRAFT_ONLY_NATIVE:
        return "draft_only"
    return policy.replace(NATIVE_PREFIX, "", 1)


def _select_in_format(rows: Sequence[Mapping[str, Any]], rule: str, fmt: str) -> str:
    """``select(rows, rule)`` on one format's policy rows, unchanged; native names are renamed for it and back."""
    if fmt == "cc":
        return select(rows, rule)
    back: dict[str, str] = {}
    renamed: list[dict[str, Any]] = []
    for r in rows:
        name = str(r["policy"])
        back[_bare(name)] = name
        renamed.append({**r, "policy": _bare(name)})
    return back[select(renamed, rule)]


def _rule_line(rs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One policy's line in the step tables (the figures the rule reads; floats for display only)."""
    tool = [r for r in rs if r.get("tool_case")]
    called = [r for r in rs if r.get("called")]
    sent = [r for r in rs if r.get("pass2_wall_ms") is not None]
    walls = [float(r["e2e_wall_ms"]) for r in tool]
    pass2 = [float(r["pass2_wall_ms"]) for r in sent]
    n_valid = sum(1 for r in called if r.get("schema_valid") is True)
    return {
        "rows": len(rs),
        "exact": f"{sum(1 for r in tool if r.get('exact'))}/{len(tool)}",
        "schema_valid": f"{n_valid}/{len(called)}",
        "cs_false": sum(1 for r in rs if r.get("cs") is False),
        "pass2_wall_ms_median": _median(pass2) if pass2 else None,
        "steps_mean_pass2": float(_mean_steps(sent)) if sent else None,
        "e2e_wall_ms_tool_p50": _median(walls) if walls else None,
        "e2e_wall_ms_tool_p90": _p90(walls) if walls else None,
        "steps_mean_tool": float(_mean_steps(tool)) if tool else None,
    }


def _format_step2(
    groups: Mapping[str, Sequence[Mapping[str, Any]]], tol: Fraction
) -> tuple[str, dict[str, Any]]:
    """Step 2 of ``select_format`` over ``groups`` (policy -> rows): ``(choice, table)``. ``ValueError`` when no
    policy is eligible."""
    eligible = {
        p: rs
        for p, rs in groups.items()
        if _schema_rate_one(rs) and all(r.get("cs") is not False for r in rs)
    }
    if not eligible:
        raise ValueError(f"step 2: no eligible policy among {sorted(groups)}")
    best = max(_exact_rate(rs) for rs in eligible.values())
    kept = _accuracy_filter(eligible, tol)
    p50: dict[str, Fraction] = {}
    p90: dict[str, Fraction] = {}
    st: dict[str, Fraction] = {}
    for p, rs in kept.items():
        tool = [r for r in rs if r.get("tool_case")]
        if not tool:
            continue
        walls = [float(r["e2e_wall_ms"]) for r in tool]
        p50[p], p90[p], st[p] = (
            Fraction(_median(walls)),
            Fraction(_p90(walls)),
            _mean_steps(tool),
        )
    if not p50:
        raise ValueError("step 2: no kept policy has tool cases")
    lo = min(p50.values())
    within = [p for p in p50 if p50[p] <= WALL_TOLERANCE * lo]

    def order(p: str) -> tuple[int, int, int]:
        kind, cell = _policy_parts(p)
        arm = cell.split("/")[-1] if cell else ""
        return (
            _KIND_ORDER[kind],
            _ARM_ORDER.get(arm, len(_ARM_ORDER)),  # the seed arm last
            _FORMAT_ORDER[_policy_format(p)],
        )

    choice = min(within, key=lambda p: (p90[p], st[p], order(p), p))
    table = {
        "tolerance": str(tol),
        "best_eligible_exact": str(best),
        "bar": str(best - tol),
        "policies": {
            p: {
                "format": _policy_format(p),
                "eligible": p in eligible,
                "kept": p in kept,
                "within_5pct": p in within,
                **_rule_line(rs),
            }
            for p, rs in groups.items()
        },
    }
    return choice, table


def select_format(
    rows: Iterable[Mapping[str, Any]], stage: str, native_thought: str | None = None
) -> FormatChoice:
    """T-0084d criterion 1 (the rule approved at H-T84-0), on policy rows (``derive`` output, full policy names).

    - **Step 1 (dev only; T-0080c criterion 1, unchanged).** Over ``cd_always:native/<T1|T2|T4>/named/perstep_mar``:
      eligible = schema_valid 1.000; keep exact >= best - 1/10; the lowest median pass-2 wall; within 5 %, fewer
      mean steps; then T4, T2, T1 (``select(..., "stage1")`` itself, on the renamed rows). The result is θ_n.
    - **Step 2.** Family ``format_family(θ_n)``, every member required (``ValueError`` naming the missing ones, e.g.
      θ_n's seed cell). Eligible = schema 1.000 and CS 1.000 on every constrained row. Keep exact >= best eligible -
      tol (1/10 dev, 1/20 test). The lowest e2e p50 over tool cases; within 5 %, the lower p90; then fewer mean steps;
      then kind order; then arm order (seed arm last); then format cc before native.

    ``winner`` is the chosen policy's format. ``test`` takes θ_n as ``native_thought`` (from the dev selection file)
    and runs step 2 only. Every comparison is exact (``Fraction``). ``ValueError``: an unknown stage, ``test`` without
    a native thought (or ``dev`` with one), no eligible step-1 cell or step-2 policy, an incomplete family."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {stage!r}")
    groups = _group(rows)
    tables: dict[str, Any] = {"rule": FORMAT_RULE, "stage": stage}
    if stage == "dev":
        if native_thought is not None:
            raise ValueError(
                "the dev stage chooses θ_n itself (step 1); native_thought is for test"
            )
        step1 = [f"cd_always:{NATIVE_PREFIX}{t}/named/perstep_mar" for t in THOUGHTS]
        step1_rows = [
            {**r, "policy": _bare(p).removeprefix("cd_always:")}
            for p in step1
            for r in groups.get(p, [])
        ]
        theta = select(step1_rows, "stage1").split("/")[0]
        tables["step1"] = {
            "tolerance": str(STAGE1_TOLERANCE),
            "native_thought": theta,
            "policies": {p: _rule_line(groups.get(p, [])) for p in step1},
        }
    else:
        if native_thought is None:
            raise ValueError("the test stage takes θ_n from the dev selection (native_thought)")
        if native_thought not in THOUGHTS:
            raise ValueError(f"native_thought must be one of {THOUGHTS}, not {native_thought!r}")
        theta = native_thought
        tables["step1"] = None
    family = format_family(theta)
    missing = [p for p in family if not groups.get(p)]
    if missing:
        raise ValueError(f"step 2: the family of θ_n {theta} is incomplete; no rows for {missing}")
    choice, tables["step2"] = _format_step2({p: groups[p] for p in family}, FORMAT_TOLERANCE[stage])
    return FormatChoice(theta, choice, _policy_format(choice), tables)


# --------------------------------------------------------------------------- the report (artifacts only)


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = _stats.wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "wilson95": [lo, hi]}


def _dist(xs: Sequence[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0, "p50": None, "p90": None, "mean": None}
    return {"n": len(xs), "p50": _median(xs), "p90": _p90(xs), "mean": sum(xs) / len(xs)}


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tool = [r for r in rows if r.get("tool_case")]
    called = [r for r in rows if r.get("called")]
    constrained = [r for r in rows if r.get("cs") is not None]
    sent = [r for r in rows if r.get("pass2_sent")]
    hits = [r for r in sent if r.get("prefix_hit_tokens") is not None]
    no_tool = [r for r in rows if not r.get("tool_case")]
    return {
        "exact": _rate(sum(1 for r in tool if r.get("exact")), len(tool)),
        "schema_valid": _rate(sum(1 for r in called if r.get("schema_valid") is True), len(called)),
        "cs": _rate(sum(1 for r in constrained if r.get("cs") is True), len(constrained)),
        "pass2_wall_ms": _dist([float(r["pass2_wall_ms"]) for r in sent]),
        "e2e_wall_ms_tool": _dist([float(r["e2e_wall_ms"]) for r in tool]),
        "e2e_wall_ms_no_tool": _dist([float(r["e2e_wall_ms"]) for r in no_tool]),
        "steps": _dist([float(r.get("steps") or 0.0) for r in sent]),
        "pass2_sent": len(sent),
        "prefix_hit_fraction": (
            sum(int(r["prefix_hit_tokens"]) for r in hits)
            / max(1, sum(int(r.get("prefix_query_tokens") or 0) for r in hits))
            if hits
            else None
        ),
        "prefix_hit_fraction_median": (
            _median(
                [
                    int(r["prefix_hit_tokens"]) / int(r["prefix_query_tokens"])
                    for r in hits
                    if r.get("prefix_query_tokens")
                ]
            )
            if any(r.get("prefix_query_tokens") for r in hits)
            else None
        ),
    }


def _mcnemar(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]], family: int) -> Any:
    """Exact McNemar on ``exact`` over tool cases, paired by case; the both-parsed restriction beside it."""
    ra = {r["case"]: r for r in a if r.get("tool_case")}
    rb = {r["case"]: r for r in b if r.get("tool_case")}
    ids = sorted(set(ra) & set(rb))

    def table(keys: Sequence[str]) -> dict[str, Any]:
        bb = sum(1 for i in keys if ra[i].get("exact") and not rb[i].get("exact"))
        cc = sum(1 for i in keys if rb[i].get("exact") and not ra[i].get("exact"))
        p = _stats.mcnemar_exact(bb, cc)
        return {
            "n": len(keys),
            "b": bb,
            "c": cc,
            "p": p,
            "p_bonferroni": _stats.bonferroni(p, family),
        }

    both = [i for i in ids if ra[i].get("parsed") and rb[i].get("parsed")]
    return {"all": table(ids), "both_parsed": table(both), "unpaired": len(set(ra) ^ set(rb))}


def is_copy(committed: Sequence[int], seed: Sequence[int]) -> bool:
    """The commit copies the seed (T-0081 ``is_copy``): equal up to and including the seed's first stop id (all
    of a seed without one)."""
    stops = _replay_stop_ids()
    head: list[int] = []
    for t in seed:
        head.append(int(t))
        if int(t) in stops:
            break
    return [int(t) for t in committed[: len(head)]] == head


def _seed_report(
    cell: str, p1: Sequence[Mapping[str, Any]], q2: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Criterion 9 for one seed cell: coverage over the routed cases (picker cases included; a routed case without
    a row of this cell is ``missing``) and the copy rate of the seeded requests' first canvas commit, split by the
    pass-1 draft's schema validity."""
    rows = {r["case"]: r for r in q2 if r.get("cell") == cell}
    routed = seeded = 0
    no_seed: Counter[str] = Counter()
    copy = {True: [0, 0], False: [0, 0]}
    trimmed = 0
    for r in p1:
        route = r.get("route")
        if (route.get("kind") if isinstance(route, Mapping) else route) != "toolcall":
            continue
        routed += 1
        q = rows.get(r["case"])
        if q is None:
            no_seed["missing"] += 1
            continue
        ids = q.get("seed_ids")
        if ids is None:
            no_seed[str(q.get("seed_reason") or "unknown")] += 1
            continue
        seeded += 1
        trimmed += int(int(q.get("seed_trimmed") or 0) > 0)
        committed = (q.get("committed_ids") or [None])[0]
        bucket = copy[bool(r.get("draft_schema_valid"))]
        bucket[1] += 1
        bucket[0] += int(committed is not None and is_copy(committed, ids))
    return {
        "routed": routed,
        "seeded": seeded,
        "no_seed": dict(sorted(no_seed.items())),
        "trimmed": _rate(trimmed, seeded),
        "copy": {
            "valid_draft": _rate(*copy[True]),
            "invalid_draft": _rate(*copy[False]),
        },
    }


def _stripped_close(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """T-0082a criterion 9 for one artifact: pass-1 drafts with a stripped close (non-empty ``draft_close``) over
    the routed (``toolcall``) pass-1 rows."""
    routed = [
        r
        for r in rows
        if r.get("pass") == 1
        and (r["route"].get("kind") if isinstance(r.get("route"), Mapping) else r.get("route"))
        == "toolcall"
    ]
    return _rate(sum(1 for r in routed if r.get("draft_close")), len(routed))


def _routed(r: Mapping[str, Any]) -> bool:
    route = r.get("route")
    return (route.get("kind") if isinstance(route, Mapping) else route) == "toolcall"


def _route_tag(r: Mapping[str, Any]) -> Any:
    route = r.get("route")
    return route.get("tag") if isinstance(route, Mapping) else None


def _format_sections(
    fmt: str,
    *,
    own_tag: bool,
    rows: Sequence[Mapping[str, Any]],
    artifact_rows: Sequence[Sequence[Mapping[str, Any]]],
    cells: Sequence[str],
    policies: Sequence[str],
    derived: Mapping[str, Sequence[Mapping[str, Any]]],
    stages: Sequence[str],
) -> dict[str, Any]:
    """The format-dependent sections of ``report`` over one format's rows and cells (T-0084d criterion 3a). On a
    cc-only artifact (``own_tag`` False) this is ``main``'s computation exactly; on a mixed one each format gets its
    own ``p1_by_case``, seed report, ``n_cases``, route recall (by its own tag, ``own_tag``), false tags, pass-1 wall,
    stripped close, McNemar against its own draft policy and stage line."""
    p1 = [r for r in rows if r.get("pass") == 1 and row_format(r) == fmt]
    q2_all = [r for r in rows if r.get("pass") == 2 and row_format(r) == fmt]
    cells_f = [c for c in cells if cell_format(c) == fmt]
    draft = "draft_only" if fmt == "cc" else DRAFT_ONLY_NATIVE
    mine = [p for p in policies if _policy_format(p) == fmt]
    tool_cases = [r for r in p1 if r.get("kind") == "tool"]
    no_tool = [r for r in p1 if r.get("kind") != "tool"]

    def recalled(r: Mapping[str, Any]) -> bool:
        return _routed(r) and (not own_tag or _route_tag(r) == fmt)

    tools = sorted({str(r.get("tool")) for r in tool_cases})
    cd = [p for p in mine if p != draft]
    family = len(cd) + len(cells_f) * (len(cells_f) - 1) // 2
    mcnemar = {f"{p} vs {draft}": _mcnemar(derived[p], derived[draft], family) for p in cd}
    for i, a in enumerate(cells_f):
        for b in cells_f[i + 1 :]:
            mcnemar[f"cd_always:{a} vs cd_always:{b}"] = _mcnemar(
                derived[f"cd_always:{a}"], derived[f"cd_always:{b}"], family
            )
    close_hist: dict[str, dict[str, int]] = {}
    agreement: dict[str, Any] = {}
    p1_by_case = {r["case"]: r for r in p1}
    for c in cells_f:
        q2 = [r for r in q2_all if r.get("cell") == c]
        if c.removeprefix(NATIVE_PREFIX).startswith("T1/"):
            h = Counter(
                "none" if r.get("close_101_pos") is None else str(r["close_101_pos"]) for r in q2
            )
            close_hist[c] = dict(sorted(h.items()))
        same = valid_drafts = repaired = invalid = 0
        for r in q2:
            d = p1_by_case.get(r["case"], {})
            if d.get("draft_schema_valid"):
                valid_drafts += 1
                same += int(r.get("call") is not None and r.get("call") == d.get("draft_call"))
            else:
                invalid += 1
                repaired += int(bool(r.get("schema_valid")))
        agreement[c] = {
            "pass2_equals_valid_draft": _rate(same, valid_drafts),
            "repair_of_invalid_draft": _rate(repaired, invalid),
            "name_picker_rows": sum(1 for r in q2 if int(r.get("requests") or 1) == 2),
        }
    # T-0082b R10: pass-2 walls by position within a case (artifact order): the first cell after pass 1 (live) or
    # the warm-up (replay) against the later cells -- a check that the replay's cache state matches live
    first_seen: set[Any] = set()
    by_position: dict[str, list[float]] = {"first": [], "later": []}
    for r in q2_all:
        key = r["case"]
        by_position["later" if key in first_seen else "first"].append(
            float(r.get("wall_ms") or 0.0)
        )
        first_seen.add(key)
    seed = {c: _seed_report(c, p1, q2_all) for c in cells_f if parse_cell(c).arm == SEED_ARM}
    selection: dict[str, Any] = {}
    if list(stages) == ["dev"]:
        stage1_rows = [
            *derived[draft],
            *[{**r, "policy": c} for c in cells_f for r in derived[f"cd_always:{c}"]],
        ]
        try:
            selection = {"rule": "stage1", "choice": _select_in_format(stage1_rows, "stage1", fmt)}
        except ValueError as exc:
            selection = {"rule": "stage1", "choice": None, "reason": str(exc)}
    elif list(stages) == ["test"]:
        try:
            choice = _select_in_format([r for p in mine for r in derived[p]], "stage2", fmt)
            selection = {"rule": "stage2", "choice": choice}
        except ValueError as exc:
            selection = {"rule": "stage2", "choice": None, "reason": str(exc)}
    return {
        "n_cases": len(p1),
        "route_recall": _rate(sum(1 for r in tool_cases if recalled(r)), len(tool_cases)),
        "false_tag_rate": _rate(sum(1 for r in no_tool if _routed(r)), len(no_tool)),
        "route_recall_per_tool": {
            t: _rate(
                sum(1 for r in tool_cases if r.get("tool") == t and recalled(r)),
                sum(1 for r in tool_cases if r.get("tool") == t),
            )
            for t in tools
        },
        "pass1_wall_ms": _dist([float(r.get("wall_ms") or 0.0) for r in p1]),
        "pass2_wall_ms_by_position": {k: _dist(v) for k, v in by_position.items()},
        "mcnemar_exact": mcnemar,
        "mcnemar_family": family,
        "t1_close_101_position": close_hist,
        "draft_agreement": agreement,
        "seed": seed,
        "stripped_close": [
            _stripped_close([r for r in prow if row_format(r) == fmt]) for prow in artifact_rows
        ],
        "selection": selection,
    }


def _format_additions(fmt: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """T-0084d criterion 3, per format, from its pass-1 rows (definitions of ruling D3): route recall over any
    ``toolcall`` tag beside the own-tag one, the count of routes with another format's tag (``wrong_tag``), the
    ``open_thought_call`` count, the draft parse-failure histogram (``wrong_tag`` its own reason), the spoken-prefix
    share over the tool cases routed by the format's own tag (non-empty after strip) and the median length of the
    non-empty prefixes (characters as spoken, unstripped), the ``text_after_close`` count (non-blank text after the
    close over the rows with a close), pass-1 prompt tokens and the median pass-1 prefix-hit fraction."""
    p1 = [r for r in rows if r.get("pass") == 1 and row_format(r) == fmt]
    tool_cases = [r for r in p1 if r.get("kind") == "tool"]
    routed = [r for r in p1 if _routed(r)]
    own = [r for r in tool_cases if _routed(r) and _route_tag(r) == fmt]
    prefixes: list[str] = []
    for r in own:
        route = r.get("route")
        prefixes.append(
            str((route.get("spoken_prefix") if isinstance(route, Mapping) else "") or "")
        )
    spoken = [s for s in prefixes if s.strip()]
    flagged = [r for r in p1 if "open_thought_call" in r]
    closes = [r for r in p1 if r.get("text_after_close") is not None]
    hits = [
        int(r["prefix_hit_tokens"]) / int(r["prefix_query_tokens"])
        for r in p1
        if r.get("prefix_hit_tokens") is not None and r.get("prefix_query_tokens")
    ]
    failures = Counter(
        str(r.get("draft_parse")) for r in routed if r.get("draft_parse") not in (None, "ok")
    )
    return {
        "route_tag": fmt,
        "route_recall_any_tag": _rate(sum(1 for r in tool_cases if _routed(r)), len(tool_cases)),
        "wrong_tag": sum(1 for r in routed if _route_tag(r) != fmt),
        "open_thought_call": _rate(
            sum(1 for r in flagged if r.get("open_thought_call") is True), len(flagged)
        ),
        "draft_parse_failures": dict(sorted(failures.items())),
        "spoken_prefix_share": _rate(len(spoken), len(prefixes)),
        "spoken_prefix_median_chars": _median([float(len(s)) for s in spoken]) if spoken else None,
        "text_after_close": _rate(
            sum(1 for r in closes if str(r["text_after_close"]).strip()), len(closes)
        ),
        "pass1_prompt_tokens": _dist(
            [float(r["prompt_len"]) for r in p1 if r.get("prompt_len") is not None]
        ),
        "pass1_prefix_hit_fraction_median": _median(hits) if hits else None,
    }


def _format_comparison(
    policies: Sequence[str],
    derived: Mapping[str, Sequence[Mapping[str, Any]]],
    cells: Sequence[str],
    stages: Sequence[str],
    by_format: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(format_selection, format_mcnemar)`` of a mixed report (T-0084d criteria 1 and 3). θ_n is step 1's on dev
    and, on test, the one native thought of the artifact's cells (ruling D4). The cross-format McNemar pairs each
    format's best eligible policy under step 2 restricted to that format (ruling D3), by case over tool cases;
    ``b`` counts cases the cc policy gets exact and the native one does not, ``c`` the reverse; Bonferroni is over
    every McNemar comparison of the report (each format's family plus this one)."""
    rows = [r for p in policies for r in derived[p]]
    try:
        if list(stages) == ["dev"]:
            fc = select_format(rows, "dev")
        elif list(stages) == ["test"]:
            thoughts = sorted({parse_cell(c).thought for c in cells if cell_format(c) == "native"})
            if len(thoughts) != 1:
                raise ValueError(
                    f"the test stage needs one native thought among the cells, not {thoughts}"
                )
            fc = select_format(rows, "test", native_thought=thoughts[0])
        else:
            raise ValueError(f"one stage per report, not {list(stages)}")
    except ValueError as exc:
        reason = str(exc)
        failed: dict[str, Any] = {
            "rule": FORMAT_RULE,
            "native_thought": None,
            "choice": None,
            "winner": None,
            "tables": {},
            "reason": reason,
        }
        return failed, {"reason": reason}
    fs: dict[str, Any] = {
        "rule": FORMAT_RULE,
        "native_thought": fc.native_thought,
        "choice": fc.choice,
        "winner": fc.winner,
        "tables": fc.tables,
    }
    groups = _group(rows)
    best: dict[str, str] = {}
    try:
        for f in FORMATS:
            sub = {p: groups[p] for p in format_family(fc.native_thought) if _policy_format(p) == f}
            best[f], _table = _format_step2(sub, FORMAT_TOLERANCE[stages[0]])
    except ValueError as exc:
        return fs, {"reason": f"no best eligible policy in one format: {exc}"}
    family = sum(int(s["mcnemar_family"]) for s in by_format.values()) + 1
    fmc = {
        "cc_policy": best["cc"],
        "native_policy": best["native"],
        "orientation": "b: cc exact and native not; c: native exact and cc not",
        "family": family,
        **_mcnemar(derived[best["cc"]], derived[best["native"]], family),
    }
    return fs, fmc


_SECTION_KEYS_A = (
    "n_cases",
    "route_recall",
    "false_tag_rate",
    "route_recall_per_tool",
    "pass1_wall_ms",
    "pass2_wall_ms_by_position",
)
_SECTION_KEYS_B = (
    "mcnemar_exact",
    "mcnemar_family",
    "t1_close_101_position",
    "draft_agreement",
    "seed",
    "stripped_close",
    "selection",
)


def report(paths: Sequence[Path | str]) -> dict[str, Any]:
    """Every figure of criterion 5 recomputed from the given artifacts (one stage per call is expected).

    Per policy (each cell as ``cd_always``, ``draft_only``, ``cd_on_invalid:<cell>``), overall and per tool:
    exact and schema_valid with Wilson 95 % CIs, CS, pass-2 and end-to-end wall p50/p90 (end to end over tool
    cases; no-tool cases separately), executed steps mean/p90, the prefix-cache hit fraction; route recall and
    the false-tag rate; McNemar against ``draft_only`` and between the CD cells (both-parsed restriction
    beside); T1's ``101`` position histogram; the agreement of pass 2 with a valid draft and the repair rate
    on invalid drafts; per seed cell (T-0081b) the seed coverage and copy rate (``seed``); and the stage's
    pre-registered selection (``select``).

    T-0084d: when a native cell or row is present, the format-dependent sections move under ``by_format[fmt]``
    (computed per format, criterion 3a, plus the criterion-3 additions), ``draft_only:native`` joins the policies,
    and ``format_selection`` (``select_format``) and ``format_mcnemar`` are added; no top-level key keeps a pooled
    value. A cc-only report is ``main``'s, byte for byte (ruling D2)."""
    regimes: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    stopped: list[Any] = []
    artifact_rows: list[list[dict[str, Any]]] = []
    for p in paths:
        regime, prow, other = read_artifact(p)
        regimes.append(regime)
        rows.extend(prow)
        artifact_rows.append(prow)
        stopped.extend(x.get("stopped") for x in [regime, *other] if x.get("stopped"))
    cells = list(dict.fromkeys(c for r in regimes for c in r.get("cells", [])))
    present = {cell_format(c) for c in cells} | {row_format(r) for r in rows}
    mixed = "native" in present
    formats = [f for f in FORMATS if f in present] if mixed else ["cc"]
    drafts = ["draft_only" if f == "cc" else DRAFT_ONLY_NATIVE for f in formats]
    policies = [*drafts, *[f"cd_always:{c}" for c in cells]]
    policies += [f"cd_on_invalid:{c}" for c in cells]
    derived = {p: derive(rows, p) for p in policies}
    tool_cases = [r for r in rows if r.get("pass") == 1 and r.get("kind") == "tool"]
    tools = sorted({str(r.get("tool")) for r in tool_cases})
    table: dict[str, Any] = {}
    for p, drows in derived.items():
        entry = {"all": _summary(drows)}
        for t in tools:
            entry[t] = _summary([r for r in drows if r.get("tool") == t])
        table[p] = entry
    stages = sorted({str(r.get("stage")) for r in regimes})
    head: dict[str, Any] = {
        "artifacts": [str(p) for p in paths],
        "regime": [{k: v for k, v in r.items() if k not in ("system_prompt",)} for r in regimes],
        "stopped": stopped,
        "cells": cells,
    }
    if not mixed:
        s = _format_sections(
            "cc",
            own_tag=False,
            rows=rows,
            artifact_rows=artifact_rows,
            cells=cells,
            policies=policies,
            derived=derived,
            stages=stages,
        )
        return {
            **head,
            **{k: s[k] for k in _SECTION_KEYS_A},
            "policies": table,
            **{k: s[k] for k in _SECTION_KEYS_B},
        }
    by_format: dict[str, dict[str, Any]] = {}
    for f in formats:
        by_format[f] = {
            **_format_sections(
                f,
                own_tag=True,
                rows=rows,
                artifact_rows=artifact_rows,
                cells=cells,
                policies=policies,
                derived=derived,
                stages=stages,
            ),
            **_format_additions(f, rows),
        }
    fs, fmc = _format_comparison(policies, derived, cells, stages, by_format)
    if stopped:
        # T-0085 criterion 7 and K6/K7: a stopped run decides nothing and names no "best" policy (the CLI's exit and
        # selection file are unchanged). The stage lines' reason names no cell, so no policy is printed there.
        reason = f"stopped: {'; '.join(str(x) for x in stopped)}; no decision"
        fs = {
            "rule": FORMAT_RULE,
            "native_thought": None,
            "choice": None,
            "winner": None,
            "tables": {},
            "reason": reason,
        }
        fmc = {"reason": reason}
        for sec in by_format.values():
            rule = sec["selection"].get("rule")
            sec["selection"] = {
                "rule": rule,
                "choice": None,
                "reason": "stopped: run stopped; no decision",
            }
    return {
        **head,
        "formats": formats,
        "policies": table,
        "by_format": by_format,
        "format_selection": fs,
        "format_mcnemar": fmc,
    }


def _fmt(x: Any, nd: int = 1) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _fmt_rate(r: Mapping[str, Any]) -> str:
    if not r["n"]:
        return "n/a"
    lo, hi = r["wilson95"]
    return f"{r['k']}/{r['n']} = {r['rate']:.3f} [{lo:.3f}, {hi:.3f}]"


def _regime_lines(rep: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for r in rep["regime"]:
        lines.append(
            f"- stage {r.get('stage')}, commit {r.get('commit')}, cells {r.get('cells')}, n {r.get('n_cases')}, "
            f"system prompt sha256 {r.get('system_prompt_sha256')}, tools sha256 {r.get('tools_sha256')}, "
            f"cases sha256 {r.get('cases_sha256')}, calibration {r.get('calibration')}, thinking on, "
            f"pass-1 max_tokens {r.get('pass1_max_tokens')}, pass-2 {r.get('pass2_max_tokens')}, "
            f"concurrency 1, no seed, versions "
            f"{ {k: v for k, v in (r.get('versions') or {}).items() if k != 'versions_lock'} }"
        )
        if r.get("pass1_from"):
            src = r.get("pass1_source") or {}
            lines.append(
                f"- pass 1 replayed (T-0082b) from {r['pass1_from']} (sha256 {r.get('pass1_sha256')}, "
                f"commit {src.get('commit')}, run {src.get('run_id')}): no pass-1 request was sent; the pass-1 "
                "rows are that artifact's, re-scored, and the pass-1 walls (and e2e walls) come from that "
                "artifact's boot"
            )
    return lines


def _position_lines(pos: Mapping[str, Any] | None) -> list[str]:
    return [
        f"- pass-2 wall ms by position within a case (T-0082b R10; first cell vs later cells): "
        f"first p50/p90 {_fmt(p['first']['p50'])}/{_fmt(p['first']['p90'])} (n {p['first']['n']}); "
        f"later p50/p90 {_fmt(p['later']['p50'])}/{_fmt(p['later']['p90'])} (n {p['later']['n']})"
        for p in [pos]
        if p
    ]


def _policy_table_lines(policies: Mapping[str, Any]) -> list[str]:
    lines = [
        "| policy | exact | schema_valid | CS | pass-2 wall p50/p90 ms | e2e p50/p90 ms (tool) "
        "| e2e p50 ms (no-tool) | steps mean/p90 | prefix hit (pooled; median) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p, entry in policies.items():
        s = entry["all"]
        lines.append(
            f"| {p} | {_fmt_rate(s['exact'])} | {_fmt_rate(s['schema_valid'])} | {_fmt_rate(s['cs'])} "
            f"| {_fmt(s['pass2_wall_ms']['p50'])}/{_fmt(s['pass2_wall_ms']['p90'])} "
            f"| {_fmt(s['e2e_wall_ms_tool']['p50'])}/{_fmt(s['e2e_wall_ms_tool']['p90'])} "
            f"| {_fmt(s['e2e_wall_ms_no_tool']['p50'])} "
            f"| {_fmt(s['steps']['mean'])}/{_fmt(s['steps']['p90'])} "
            f"| {_fmt(s['prefix_hit_fraction'], 3)}; {_fmt(s['prefix_hit_fraction_median'], 3)} |"
        )
    lines += ["", "| policy | tool | exact | schema_valid | e2e p50 ms |", "|---|---|---|---|---|"]
    for p, entry in policies.items():
        for t, s in entry.items():
            if t == "all":
                continue
            lines.append(
                f"| {p} | {t} | {_fmt_rate(s['exact'])} | {_fmt_rate(s['schema_valid'])} "
                f"| {_fmt(s['e2e_wall_ms_tool']['p50'])} |"
            )
    return lines


def _mcnemar_lines(mcnemar: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for k, v in mcnemar.items():
        a, bp = v["all"], v["both_parsed"]
        lines.append(
            f"- {k}: n {a['n']}, b {a['b']}, c {a['c']}, p {a['p']:.4g} (adj {a['p_bonferroni']:.4g}); "
            f"both parsed n {bp['n']}, b {bp['b']}, c {bp['c']}, p {bp['p']:.4g}"
        )
    return lines


def _agreement_seed_lines(agreement: Mapping[str, Any], seed: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for c, a in agreement.items():
        lines.append(
            f"- {c}: pass 2 equals the valid draft {_fmt_rate(a['pass2_equals_valid_draft'])}; "
            f"invalid drafts repaired {_fmt_rate(a['repair_of_invalid_draft'])}; "
            f"name-picker rows {a['name_picker_rows']}"
        )
    for c, sd in seed.items():
        lines.append(
            f"- seed {c}: seeded {sd['seeded']}/{sd['routed']} routed cases; no seed {sd['no_seed'] or 'none'}"
        )
        lines.append(
            f"- seed copy {c} (first canvas equals the seed through its first stop): valid drafts "
            f"{_fmt_rate(sd['copy']['valid_draft'])}; invalid drafts {_fmt_rate(sd['copy']['invalid_draft'])}"
        )
        lines.append(
            f"- seed trim {c} (seeded rows whose draft lost trailing whitespace, R2): "
            f"{_fmt_rate(sd['trimmed'])}"
        )
    return lines


def render_markdown(rep: Mapping[str, Any]) -> str:
    """A markdown rendering of ``report``'s dict (no figure that is not in it)."""
    if "by_format" in rep:
        return _render_mixed(rep)
    lines = ["# Two-pass tool-call report (T-0080b driver)", ""]
    lines += _regime_lines(rep)
    lines += [
        f"- stopped: {rep['stopped'] or 'no'}",
        f"- route recall (tool cases): {_fmt_rate(rep['route_recall'])}; "
        f"false-tag rate (no-tool cases): {_fmt_rate(rep['false_tag_rate'])}",
        *(
            f"- drafts with a stripped `</cc>` close (T-0082a), {a}: {s['k']}/{s['n']} routed"
            for a, s in zip(rep["artifacts"], rep.get("stripped_close") or [], strict=False)
        ),
        f"- selection: {rep['selection']}",
        *_position_lines(rep.get("pass2_wall_ms_by_position")),
        "",
    ]
    lines += _policy_table_lines(rep["policies"])
    lines += ["", f"McNemar (exact, tool cases; Bonferroni over {rep['mcnemar_family']}):", ""]
    lines += _mcnemar_lines(rep["mcnemar_exact"])
    lines += ["", f"T1 `101` position: {rep['t1_close_101_position']}", ""]
    lines += _agreement_seed_lines(rep["draft_agreement"], rep.get("seed") or {})
    return "\n".join(lines) + "\n"


def _render_mixed(rep: Mapping[str, Any]) -> str:
    """T-0084d: the mixed (cc + native) report: per-format sections, the step tables, the format winner line and
    the cross-format McNemar; a dev artifact is marked a pilot."""
    stages = {str(r.get("stage")) for r in rep["regime"]}
    fs = rep["format_selection"]
    lines = ["# Two-pass tool-call report (T-0080b driver; T-0084d format comparison)", ""]
    if "dev" in stages:
        lines += ["**dev pilot; not a held-out result**", ""]
    lines += _regime_lines(rep)
    lines += [
        f"- stopped: {rep['stopped'] or 'no'}",
        f"- formats: {', '.join(rep['formats'])}, paired by case in one boot. The formats differ jointly in the "
        "prompt instructions, where the declarations live, the call syntax and the grammar (T-0084).",
    ]
    if rep["stopped"]:
        lines.append("- format winner: none (run stopped; no decision)")
    elif fs.get("choice") is not None:
        lines.append(
            f"- format winner: {fs['winner']} (policy {fs['choice']}; native thought θ_n "
            f"{fs['native_thought']}; rule {fs['rule']} criterion 1, step-2 tolerance "
            f"{fs['tables']['step2']['tolerance']})"
        )
    else:
        lines.append(f"- format winner: none ({fs.get('reason')})")
    lines += [""]
    lines += _policy_table_lines(rep["policies"])
    for fmt, sec in rep["by_format"].items():
        p1w, ptok = sec["pass1_wall_ms"], sec["pass1_prompt_tokens"]
        lines += [
            "",
            f"## Format {fmt}",
            "",
            f"- cases {sec['n_cases']}; route recall by the `{sec['route_tag']}` tag (tool cases): "
            f"{_fmt_rate(sec['route_recall'])}; by any tool-call tag: {_fmt_rate(sec['route_recall_any_tag'])}; "
            f"false-tag rate (no-tool cases): {_fmt_rate(sec['false_tag_rate'])}",
            f"- routes with another format's tag (wrong_tag): {sec['wrong_tag']}; open_thought_call: "
            f"{_fmt_rate(sec['open_thought_call'])}",
            f"- draft parse failures: {sec['draft_parse_failures'] or 'none'}",
            f"- spoken prefix non-empty (routed tool cases): {_fmt_rate(sec['spoken_prefix_share'])}; median "
            f"chars {_fmt(sec['spoken_prefix_median_chars'])}; text after the close: "
            f"{_fmt_rate(sec['text_after_close'])}",
            f"- pass-1 wall ms p50/p90 {_fmt(p1w['p50'])}/{_fmt(p1w['p90'])} (n {p1w['n']}); pass-1 prompt "
            f"tokens p50 {_fmt(ptok['p50'])} (n {ptok['n']}); pass-1 prefix hit median "
            f"{_fmt(sec['pass1_prefix_hit_fraction_median'], 3)}",
            *(
                f"- drafts with a stripped close, {a}: {s['k']}/{s['n']} routed"
                for a, s in zip(rep["artifacts"], sec.get("stripped_close") or [], strict=False)
            ),
            f"- {fmt} stage line: {sec['selection']}",
            *_position_lines(sec.get("pass2_wall_ms_by_position")),
            "",
            f"McNemar (exact, tool cases; Bonferroni over {sec['mcnemar_family']}):",
            "",
            *_mcnemar_lines(sec["mcnemar_exact"]),
            "",
            f"T1 `101` position: {sec['t1_close_101_position']}",
            "",
            *_agreement_seed_lines(sec["draft_agreement"], sec.get("seed") or {}),
        ]
    lines += ["", "## Format comparison (T-0084d criterion 1)", ""]
    tables = fs.get("tables") or {}
    step1 = tables.get("step1")
    if step1:
        lines += [
            f"Step 1 (θ_n = {step1['native_thought']}; tolerance {step1['tolerance']}):",
            "",
            "| step-1 cell | exact | schema_valid | pass-2 median ms | steps mean |",
            "|---|---|---|---|---|",
            *(
                f"| {p} | {x['exact']} | {x['schema_valid']} | {_fmt(x['pass2_wall_ms_median'])} "
                f"| {_fmt(x['steps_mean_pass2'])} |"
                for p, x in step1["policies"].items()
            ),
            "",
        ]
    step2 = tables.get("step2")
    if step2:
        lines += [
            f"Step 2 (tolerance {step2['tolerance']}; best eligible exact {step2['best_eligible_exact']}, "
            f"bar {step2['bar']}):",
            "",
            "| policy | format | eligible | kept | within 5 % | exact | schema_valid | CS false "
            "| e2e p50/p90 ms (tool) | steps mean |",
            "|---|---|---|---|---|---|---|---|---|---|",
            *(
                f"| {p} | {x['format']} | {x['eligible']} | {x['kept']} | {x['within_5pct']} "
                f"| {x['exact']} | {x['schema_valid']} | {x['cs_false']} "
                f"| {_fmt(x['e2e_wall_ms_tool_p50'])}/{_fmt(x['e2e_wall_ms_tool_p90'])} "
                f"| {_fmt(x['steps_mean_tool'])} |"
                for p, x in step2["policies"].items()
            ),
            "",
        ]
    fmc = rep["format_mcnemar"]
    if "all" in fmc:
        a, bp = fmc["all"], fmc["both_parsed"]
        lines.append(
            f"- cross-format McNemar (reported only), {fmc['cc_policy']} vs {fmc['native_policy']} "
            f"({fmc['orientation']}): n {a['n']}, b {a['b']}, c {a['c']}, p {a['p']:.4g} (Bonferroni "
            f"over {fmc['family']}: {a['p_bonferroni']:.4g}); both parsed n {bp['n']}, b {bp['b']}, "
            f"c {bp['c']}"
        )
    else:
        lines.append(f"- cross-format McNemar: not computed ({fmc.get('reason')})")
    return "\n".join(lines) + "\n"
