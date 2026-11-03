"""The T-0069 tier-3 smoke checks (criteria 4-9) over saved evidence: pure, torch-free.

The driver (``scripts/gpu/cd_smoke.py``) collects the evidence from a booted engine -- responses, tap
frames, the engine-log slice -- and decides each criterion with these functions; the independent re-check
(``tests/integration/test_cd_engine_smoke.py``) re-derives the same verdicts without importing this module.
Nothing here talks to the engine, compiles a grammar or imports torch: the host replay arrives as an
``accepts`` callable from the caller.

Conventions (the API ``tests/cd/unit/test_cd_smoke_checks.py`` fixes):

- every ``*_problems`` function returns ``list[str]``, one human-readable problem per finding; ``[]`` is a
  pass;
- records are ``CdCanvasRecord`` JSON dicts (as parsed from ``ASTRA_CD <json>`` lines); frames are
  ``astra.canvas.step`` wire dicts; release rows are ``b16_tap.parse_tap_release_lines`` dicts.

Join keys (T-0069 implementation notes): a tap frame's ``request_id`` is the response ``id`` plus
``-<8 lowercase hex>``; CD records join on ``record_tag`` (the caller selects them).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from local_astra.bench.b16_tap import boot_config_lines, last_boot
from local_astra.contracts.protocol import AstraCanvasStep, AstraMessageAdapter

__all__ = [
    "CANVAS_LENGTH",
    "EOS",
    "HEADER_ID",
    "LOUD_CODES",
    "STOP_IDS",
    "belongs",
    "boot_problems",
    "commit_count_problems",
    "commit_frame_problems",
    "frame_problems",
    "frames_for",
    "header_report",
    "loud_problems",
    "multicanvas_problems",
    "preempted_count",
    "quiet_window_problems",
    "record_problems",
    "release_problems",
    "replay_problems",
    "serving_problems",
    "token_ids_problems",
    "transport_problems",
    "vllm_sh_problems",
]

Record = Mapping[str, Any]
Frame = Mapping[str, Any]

#: The engine's canvas length (``--diffusion-config``, T-0069 Regime).
CANVAS_LENGTH = 256
EOS = 1
#: The pinned ``generation_config.json`` ``eos_token_id`` (= ``replay.STOP_IDS``; repeated here because
#: ``replay`` imports the grammar compiler on use and this module stays import-light).
STOP_IDS = frozenset({1, 50, 106})
#: ``<|channel>``: the first id of a header-language prefix (``replay.CHANNEL_OPEN_ID``).
HEADER_ID = 100
#: The codes a loud (malformed-payload) request may carry in criterion 5.
LOUD_CODES = ("bad_encoding", "structure")
#: Criterion 6's literal length window: literal + stop > 256 (bucket 512 by construction).
LITERAL_MIN, LITERAL_MAX = 260, 290
MULTICANVAS_BUCKET = 512
#: Criterion 7's sizes relative to the cap.
NEAR_CAP_MARGIN = 512
#: Criterion 9's header sample: 10 Sudoku + 10 BFCL dev records.
HEADER_SAMPLE = {"sudoku": 10, "bfcl": 10}
#: T-0070 arm name -> the mode its records name (ruling W5: the SC arm's records say ``perstep_mar``).
_ARM_MODE = {
    "end_projection": "end_projection",
    "perstep_mf": "perstep_mf",
    "perstep_mar": "perstep_mar",
    "perstep_mar_sc": "perstep_mar",
    "perstep_mar_qargmax": "perstep_mar_qargmax",
    "perstep_mar_qargmax_sc": "perstep_mar_qargmax",
}
_ENGINE_SUFFIX = re.compile(r"-[0-9a-f]{8}")
_CD_MARKS = ("ASTRA_CD ", "astra_cd ")


# --------------------------------------------------------------------------- join


def belongs(engine_request_id: str, response_id: str) -> bool:
    """True iff ``engine_request_id`` is ``response_id`` + ``-`` + 8 lowercase hex characters."""
    if not engine_request_id.startswith(response_id):
        return False
    return _ENGINE_SUFFIX.fullmatch(engine_request_id[len(response_id) :]) is not None


def frames_for(frames: Sequence[Frame], response_id: str) -> list[dict[str, Any]]:
    """The frames of ``response_id`` (by ``belongs``), in input order."""
    return [dict(f) for f in frames if belongs(str(f.get("request_id", "")), response_id)]


def _by_idx(records: Sequence[Record]) -> list[Record]:
    return sorted(records, key=lambda r: int(r["canvas_idx"]))


def _ids(records: Sequence[Record]) -> list[int]:
    return [int(t) for r in _by_idx(records) for t in r["committed_ids"]]


def _commits(frames: Sequence[Frame]) -> list[Frame]:
    return [f for f in frames if f.get("phase") == "commit"]


# --------------------------------------------------------------------------- frames (criterion 4)


def frame_problems(frames: Sequence[Frame]) -> list[str]:
    """R10 shape per phase, R15/R14 step ranges (both ``0..max_steps``; a commit may carry step 0, ruling
    O6), ``accepted``/``entropy`` None, on every frame."""
    bad: list[str] = []
    for f in frames:
        where = f"{f.get('request_id')}#{f.get('canvas_idx')} step {f.get('step')}"
        try:
            m = AstraMessageAdapter.validate_python(dict(f))
        except ValidationError as exc:
            bad.append(f"invalid frame {where}: {exc.errors()[0].get('msg', exc)}")
            continue
        if not isinstance(m, AstraCanvasStep):
            bad.append(f"not a canvas step: {f.get('type')!r}")
            continue
        if m.accepted is not None or m.entropy is not None:
            bad.append(f"accepted/entropy filled on {where}")
        if m.phase == "denoise" and not 0 <= m.step <= m.max_steps:
            bad.append(f"denoise step {m.step} outside 0..{m.max_steps} on {where} (R15)")
        if m.phase == "commit" and not 0 <= m.step <= m.max_steps:
            bad.append(f"commit step {m.step} outside 0..{m.max_steps} on {where} (R14, ruling O6)")
    return bad


# --------------------------------------------------------------------------- records (criterion 4)


def record_problems(records: Sequence[Record], *, record_tag: str, arm: str) -> list[str]:
    """4(a)(b) and "error is null": tag echoed, arm's mode named, ``canvas_idx`` 0..n-1, the final
    record carries ``stop_pos`` or ``error``, no record has an ``error``."""
    if not records:
        return [f"no ASTRA_CD record for tag {record_tag}: xargs did not reach extra_args"]
    out: list[str] = []
    mode = _ARM_MODE.get(arm)
    if mode is None:
        out.append(f"unknown arm {arm!r}")
    tags = sorted({str(r.get("record_tag")) for r in records} - {record_tag})
    if tags:
        out.append(f"records carry tag(s) {tags}, not {record_tag}")
    named = sorted({str(r.get("arm")) for r in records})
    if mode is not None and named != [mode]:
        out.append(f"records name arm(s) {named}; {arm} records name {mode}")
    idx = [int(r["canvas_idx"]) for r in _by_idx(records)]
    if idx != list(range(len(records))):
        out.append(f"canvas_idx {idx} is not 0..{len(records) - 1} without gap or duplicate")
    last = _by_idx(records)[-1]
    if last.get("stop_pos") is None and last.get("error") is None:
        out.append(f"the final record (canvas {last['canvas_idx']}) has neither stop_pos nor error")
    for r in _by_idx(records):
        if r.get("error") is not None:
            out.append(f"canvas {r['canvas_idx']} has error {r['error']}")
    return out


def replay_problems(
    records: Sequence[Record], accepts: Callable[[Sequence[int]], bool]
) -> list[str]:
    """4(c): host replay of the committed ids (``canvas_idx`` order) accepts, and agrees with the final
    record's ``in_final``."""
    if not records:
        return ["no records to replay"]
    ok = bool(accepts(_ids(records)))
    out: list[str] = []
    if not ok:
        out.append("host replay rejects the committed ids")
    in_final = bool(_by_idx(records)[-1].get("in_final"))
    if ok != in_final:
        out.append(f"host replay says {ok}, the plugin's in_final says {in_final}")
    return out


def release_problems(release_rows: Sequence[Mapping[str, Any]], response_id: str) -> list[str]:
    """4(d): at least one tap release line of this request, every one with ``commit_overrun == 0``."""
    mine = [r for r in release_rows if belongs(str(r.get("req_id", "")), response_id)]
    if not mine:
        return [f"no tap release line for {response_id}"]
    return [
        f"release line {r['req_id']} has commit_overrun {r['commit_overrun']}"
        for r in mine
        if r.get("commit_overrun") != 0
    ]


def commit_frame_problems(frames: Sequence[Frame], records: Sequence[Record]) -> list[str]:
    """4(e): one commit frame per record, matched on ``canvas_idx``, ``committed_tokens ==
    committed_ids``."""
    commits = _commits(frames)
    out: list[str] = []
    if len(commits) != len(records):
        out.append(f"{len(commits)} commit frames for {len(records)} records")
    by_frame: dict[int, list[Frame]] = {}
    for f in commits:
        by_frame.setdefault(int(f["canvas_idx"]), []).append(f)
    for idx, dup in sorted(by_frame.items()):
        if len(dup) > 1:
            out.append(f"{len(dup)} commit frames for canvas {idx}")
    for r in _by_idx(records):
        idx = int(r["canvas_idx"])
        fs = by_frame.get(idx)
        if not fs:
            out.append(f"no commit frame for canvas {idx}")
            continue
        want = [int(t) for t in r["committed_ids"]]
        got = [int(t) for t in fs[0].get("committed_tokens") or ()]
        if got != want:
            diff = next(
                (i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b),
                min(len(got), len(want)),
            )
            out.append(
                f"canvas {idx}: committed_tokens differ from the record's committed_ids at "
                f"position {diff} (frame {len(got)} ids, record {len(want)} ids)"
            )
    extra = sorted(set(by_frame) - {int(r["canvas_idx"]) for r in records})
    if extra:
        out.append(f"commit frames for canvases {extra} that have no record")
    return out


def commit_count_problems(frames: Sequence[Frame], completion_tokens: int) -> list[str]:
    """Criterion 8: the number of commit frames is ``ceil(completion_tokens / 256)``."""
    n = len(_commits(frames))
    want = math.ceil(completion_tokens / CANVAS_LENGTH)
    if n != want:
        return [f"{n} commit frames, want ceil({completion_tokens}/{CANVAS_LENGTH}) = {want}"]
    return []


def preempted_count(records: Sequence[Record]) -> int:
    """T-0067 S10, reported: the records whose ``error`` is ``preempted``."""
    return sum(r.get("error") == "preempted" for r in records)


# --------------------------------------------------------------------------- criterion 5


def loud_problems(content: str | None, records: Sequence[Record], *, predicted: str) -> list[str]:
    """X: empty content; one record, canvas 0, first committed id EOS, ``error == predicted``, and the
    host predicts ``bad_encoding`` or ``structure``."""
    out: list[str] = []
    if predicted not in LOUD_CODES:
        out.append(f"the host predicts {predicted!r}, not one of {LOUD_CODES}: X is not this row")
    if content:
        out.append(f"X returned content {content[:40]!r}")
    if len(records) != 1:
        out.append(f"X has {len(records)} records, want exactly 1")
    if not records:
        return out
    r = records[0]
    if int(r["canvas_idx"]) != 0:
        out.append(f"X's record is canvas {r['canvas_idx']}, want 0")
    ids = list(r.get("committed_ids") or ())
    if ids[:1] != [EOS]:
        out.append(f"X's commit does not start with EOS ({EOS}): {ids[:4]}")
    if r.get("error") != predicted:
        out.append(f"X's record says {r.get('error')!r}, the host predicts {predicted!r}")
    return out


def serving_problems(*, next_status: int, health_status: int) -> list[str]:
    """The next unconstrained request and ``/health`` both returned 200."""
    out: list[str] = []
    if next_status != 200:
        out.append(f"the next unconstrained request returned HTTP {next_status}")
    if health_status != 200:
        out.append(f"/health returned {health_status}")
    return out


# --------------------------------------------------------------------------- criterion 6


def multicanvas_problems(
    records: Sequence[Record],
    *,
    literal_ids: Sequence[int],
    completion_tokens: int,
    max_tokens: int = 300,
) -> list[str]:
    """A 260-290-token literal (bucket 512) closes exactly canvases 0 and 1, the committed ids are the
    literal followed by a stop id, ``completion_tokens <= max_tokens``, and no record has an error
    (``internal`` = an S9-caught OOM included)."""
    out: list[str] = []
    lit = [int(t) for t in literal_ids]
    if not LITERAL_MIN <= len(lit) <= LITERAL_MAX:
        out.append(f"the literal has {len(lit)} ids, want {LITERAL_MIN}-{LITERAL_MAX}")
    idx = [int(r["canvas_idx"]) for r in _by_idx(records)]
    if idx != [0, 1]:
        out.append(f"records at canvases {idx}, want exactly [0, 1]")
    buckets = sorted({r.get("bucket") for r in records}, key=str)
    if buckets != [MULTICANVAS_BUCKET]:
        out.append(f"buckets {buckets}, want [{MULTICANVAS_BUCKET}]")
    for r in _by_idx(records):
        if r.get("error") is not None:
            out.append(f"canvas {r['canvas_idx']} has error {r['error']} (loud; never retried)")
    ids = _ids(records)
    if ids[: len(lit)] != lit:
        out.append("the committed ids do not reproduce the literal")
    elif len(ids) <= len(lit) or ids[len(lit)] not in STOP_IDS:
        out.append("no stop id right after the literal")
    if completion_tokens > max_tokens:
        out.append(f"completion_tokens {completion_tokens} > max_tokens {max_tokens}")
    return out


# --------------------------------------------------------------------------- criterion 7


def transport_problems(near: Mapping[str, Any], over: Mapping[str, Any], *, cap: int) -> list[str]:
    """Decision 4: the near-cap payload arrives intact (``bad_encoding`` is reachable only after the sha
    matched), the over-cap one is refused by size (``payload_too_large``), both with HTTP 200 and the
    host-predicted code. A 4xx/413 goes to the human (ruling O3)."""
    out: list[str] = []
    for name, case, length, want in (
        ("near", near, cap - NEAR_CAP_MARGIN, "bad_encoding"),
        ("over", over, cap + 1, "payload_too_large"),
    ):
        if case.get("length") != length:
            out.append(f"{name}: payload length {case.get('length')}, want {length}")
        status = case.get("status")
        if status != 200:
            out.append(
                f"{name}: HTTP {status}: the API server refused the body; inline base64 at the cap is "
                "not viable -- the human picks the transport (ruling O3), never a silent retry"
            )
        recs = list(case.get("records") or ())
        if len(recs) != 1:
            out.append(f"{name}: {len(recs)} records, want 1")
        predicted = case.get("predicted")
        if predicted != want:
            out.append(f"{name}: the host predicts {predicted!r}, criterion 7 wants {want!r}")
        if recs:
            code = recs[0].get("error")
            if code != predicted:
                out.append(f"{name}: record says {code!r}, the host predicts {predicted!r}")
            if code != want:
                hint = (
                    " (sha_mismatch: the payload did not arrive intact; a fail, not proof)"
                    if (code == "sha_mismatch")
                    else ""
                )
                out.append(f"{name}: record says {code!r}, want {want!r}{hint}")
    return out


# --------------------------------------------------------------------------- criterion 8


def boot_problems(log: str, reference: str) -> list[str]:
    """R16: ``boot_config_lines(last_boot(log))`` equals ``reference`` minus its ``#`` and blank lines."""
    try:
        boot = last_boot(log)
    except ValueError as exc:
        return [f"no boot in the engine log: {exc}"]
    got = set(boot_config_lines(boot))
    ref = {ln for ln in reference.splitlines() if ln.strip() and not ln.startswith("#")}
    return [f"only in this boot: {x}" for x in sorted(got - ref)] + [
        f"only in the reference: {x}" for x in sorted(ref - got)
    ]


def vllm_sh_problems(diff: str) -> list[str]:
    """``git diff 806da12..HEAD -- services/vllm.sh`` must be empty; anything else is a refusal."""
    if diff.strip():
        n = sum(ln.startswith(("+", "-")) for ln in diff.splitlines())
        return [f"services/vllm.sh changed since 806da12 ({n} diff lines): refusing"]
    return []


def quiet_window_problems(window: str) -> list[str]:
    """An unconstrained request's log window holds no ``ASTRA_CD `` and no ``astra_cd `` line."""
    return [
        f"CD line in the unconstrained window: {ln.strip()[:160]}"
        for ln in window.splitlines()
        if any(m in ln for m in _CD_MARKS)
    ]


# --------------------------------------------------------------------------- criterion 9


def header_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reported, not gated: ``n``, ``first8``, ``header_prefix`` (rows opening with ``<|channel>``),
    ``rate`` and ``problems`` (the sample, never the rate)."""
    problems: list[str] = []
    n = len(rows)
    want_n = sum(HEADER_SAMPLE.values())
    if n != want_n:
        problems.append(f"{n} header rows, want {want_n}")
    counts = {t: sum(r.get("task") == t for r in rows) for t in HEADER_SAMPLE}
    if counts != HEADER_SAMPLE or sum(counts.values()) != n:
        problems.append(f"header rows by task {counts}, want {HEADER_SAMPLE}")
    first8: list[list[int] | None] = []
    header = 0
    for r in rows:
        ids = r.get("token_ids")
        if not isinstance(ids, list) or not ids:
            problems.append(f"{r.get('dataset_id')}: token_ids absent (T-0070 E5)")
            first8.append(None)
            continue
        first8.append([int(t) for t in ids[:8]])
        header += int(ids[0]) == HEADER_ID
    return {
        "n": n,
        "first8": first8,
        "header_prefix": header,
        "rate": header / n if n else 0.0,
        "problems": problems,
    }


def token_ids_problems(
    token_ids: Sequence[int] | None, records: Sequence[Record]
) -> tuple[list[str], bool | None]:
    """``token_ids`` equals the concatenated ``committed_ids`` up to the first stop id, without or with
    it; the second value says which (``None`` when ``token_ids`` is absent)."""
    if token_ids is None:
        return ["choices[0].token_ids absent (T-0070 E5)"], None
    got = [int(t) for t in token_ids]
    committed = _ids(records)
    cut = next((i for i, t in enumerate(committed) if t in STOP_IDS), len(committed))
    if got == committed[:cut]:
        return [], False
    if cut < len(committed) and got == committed[: cut + 1]:
        return [], True
    return [
        f"token_ids ({len(got)} ids, first {got[:8]}) are not the committed ids up to the first stop "
        f"({cut} ids, first {committed[:8]})"
    ], None
