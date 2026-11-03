"""T-0087 -- what the live stream carries for a native tool call. `/launch` ONLY.

    uv run --all-extras --group cd python scripts/gpu/t0080d_native_stream_probe.py \
        --base-url http://127.0.0.1:8001/v1 --repo .

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one (engine only, no STT). Importing this file opens no socket. Refusals (exit 2), in this order and before any
socket: a ``--base-url`` whose host is not an IP literal in 127.0.0.0/8 or ``::1`` (ruling P15; ``localhost`` is a
name and is refused, nothing is resolved), a dirty ``--repo`` (the
dirty paths are named), an existing non-empty ``--out-dir``, a ``cases.jsonl`` that is not the pinned one.

What it sends (docs/tasks/T-0087.md), concurrency 1, in this order:

1. q8 (prefix cache, ``/metrics`` deltas via ``twopass._metrics``), before any arm so the prompts are cold: on
   ``Q8_TOOLS_CASES`` a 1-token prefill **with** ``tools`` then the generate request (arm (d)'s body); then the
   control on ``Q8_CONTROL_CASES`` (other screens, so their image/user blocks are cold), a 1-token prefill
   **without** ``tools`` then the same kind of generate request. The control is on other cases than the
   with-tools pairs because a second generate of one prompt would hit whatever the first one cached.
2. The five arms (``arm_body``) on each of the 12 ``CASE_IDS``, arms in the order a..e per case.
3. q9: a non-stream T4 continuation (``spoken_prefix + "<|tool_call>"``, string, then the same as one text part)
   on the first ``Q9_N`` tool cases whose arm-(d) host decode routes ``native``.
4. q10: the engine's ``/tokenize`` (``add_special_tokens: false``) of each tool case's arm-(d) draft call text
   against ``vocab.encode(draft.rstrip())``.

q3/q4 reference ids (ruling P7): arms (b)-(e) carry ``return_token_ids`` and each is judged against its **own** ids
(the response ids for (d), the concatenated per-chunk ids for a stream arm). Arm (a), today's ``llm_client``
baseline, carries none: its q3/q4 are ``None`` and its comparison with arm (d)'s decode of the same case is the
diagnostic ``vs_d = {"q3", "q4", "label": "noise-confounded"}`` (the sampler is unseeded), which decides nothing.
Arm (e) is the same request as (b), a replicate: ``b_ids_equal_e_ids`` per case is the noise level and decides
nothing.

Output (``--out-dir``, default ``<repo>/logs/t0087-probe-<sha7>/``): ``raw.jsonl`` (every request, redacted with
``redact``, and its raw SSE lines or JSON payload), ``report.json`` and ``report.md``. Field names are vLLM's,
verbatim (ruling P6). Exit codes: 0 every request answered; 1 no engine, or a request failed at the transport
level; 2 refused.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import hashlib
import http.client
import ipaddress
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from local_astra.decoding import vocab
from local_astra.eval.cd import replay, twopass
from local_astra.toolcall import router
from local_astra.toolcall import tools as _tools

REPO = Path(__file__).resolve().parents[2]
SET_REL = Path("tests") / "eval" / "toolcall"
#: sha256 of ``tests/eval/toolcall/cases.jsonl`` (criterion 1)
CASES_SHA256 = "460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85"

#: Criterion 1: 12 dev cases, 2 per tool and 2 no-tool (one text-only, one with a screen).
CASE_IDS = (
    "search-01",
    "search-02",
    "click-01",
    "click-02",
    "type-01",
    "type-02",
    "scroll-01",
    "scroll-02",
    "key-01",
    "key-02",
    "chat-01",
    "chat-06",
)
ARMS = ("a", "b", "c", "d", "e")
#: The arms reading A may choose, in preference order (criterion 6).
READING_ARMS = ("b", "c")
#: The stream arms that ask for per-chunk ids (ruling P13's ``streamed_ids``).
STREAM_ID_ARMS = ("b", "c", "e")
#: The decisions a run with a transport error leaves to a re-run (ruling P11).
DECISIONS = ("reading", "stop_id", "tools_in_prefill", "continuation_content", "seed_tokenizer")
#: Ruling P17: also stated as re-runs under ``complete: false`` (``arm`` becomes ``None``).
UNDECIDED_EXTRA = ("arm", "fallback", "streamed_ids")
#: q8: three tool cases with tools in the prefill (screens d01, d02, d08) and three control cases on other screens
#: (d03, d05, d06), so no control prompt shares an image block with an earlier request.
Q8_TOOLS_CASES = ("click-01", "type-02", "key-01")
Q8_CONTROL_CASES = ("click-05", "scroll-01", "type-05")
Q8_MIN = Fraction(9, 10)
Q9_N = 3
Q10_N = 10
N_CASES = 12
N_TOOL_CASES = 10
STOP_INTS = frozenset({1, 106, 50})
TOOL_RESPONSE = "<|tool_response>"
TOOL_RESPONSE_ID = 50
TOOL_CALL_OPEN_ID = twopass.TOOL_CALL_OPEN_ID
THOUGHT_HEADER = "<|channel>thought\n"
CHANNEL_CLOSE = router.CHANNEL_CLOSE
REASONING_FIELDS = ("reasoning", "reasoning_content")
TOKENIZE_PATHS = ("/tokenize", "/v1/tokenize")
EXIT_OK, EXIT_STOPPED, EXIT_REFUSED = 0, 1, 2

_DATA_URL = re.compile(r"data:[^;,\"\s]*;base64,([A-Za-z0-9+/=]+)")


def _say(msg: str) -> None:
    print(f"t0087_probe: {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- bodies


def arm_body(body: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Arm ``a``..``e`` applied to a ``twopass.native_pass1_body`` (a deep copy; ``body`` is not touched).

    (a) ``stream``, no ids (today's ``llm_client``); (b) (a) + ``skip_special_tokens: false`` +
    ``return_token_ids``; (c) (b) + ``spaces_between_special_tokens: false``; (d) (b) non-stream; (e) the same
    request as (b), a replicate (ruling P7, reversing P2 for (b) and (c))."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, not {arm!r}")
    out: dict[str, Any] = copy.deepcopy(dict(body))
    out.pop("return_token_ids", None)
    out.pop("skip_special_tokens", None)
    out.pop("spaces_between_special_tokens", None)
    out["stream"] = arm != "d"
    if arm != "a":
        out["skip_special_tokens"] = False
        out["return_token_ids"] = True
    if arm == "c":
        out["spaces_between_special_tokens"] = False
    return out


def _digest(url: str) -> dict[str, Any] | str:
    m = _DATA_URL.fullmatch(url)
    if m is None:
        return url
    try:
        raw = base64.b64decode(m.group(1), validate=True)
    except (binascii.Error, ValueError):
        return {
            "bytes": None,
            "sha256": hashlib.sha256(url.encode()).hexdigest(),
            "undecodable": True,
        }
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def redact(body: Any) -> Any:
    """A copy of ``body`` with each image data URL replaced by ``{"bytes", "sha256"}`` of the decoded image. A data
    URL embedded in a longer string (a 400 echoing the request, ruling P14) is replaced in place by that digest."""
    if isinstance(body, Mapping):
        return {k: redact(v) for k, v in body.items()}
    if isinstance(body, list | tuple):
        return [redact(v) for v in body]
    if isinstance(body, str) and body.startswith("data:") and _DATA_URL.fullmatch(body):
        return _digest(body)
    if isinstance(body, str) and "data:" in body:
        return _scrub(body)
    return body


def _scrub(text: str) -> str:
    """Raw response text with any data URL replaced by its digest (a response never carries one; a safety net)."""
    return _DATA_URL.sub(lambda m: json.dumps(_digest(m.group(0)), sort_keys=True), text)


# --------------------------------------------------------------------------- reassembly and the comparisons


def _dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _reasoning_of(obj: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """``(field, text)``: the first of ``REASONING_FIELDS`` carrying a string, verbatim; ``(None, None)``."""
    for name in REASONING_FIELDS:
        value = obj.get(name)
        if isinstance(value, str):
            return name, value
    return None, None


def reassemble(lines: Iterable[str]) -> dict[str, Any]:
    """Raw SSE lines to the joined ``reasoning``/``content``, the per-chunk fields (``None`` where the delta or the
    choice did not carry one), the final ``finish_reason``/``stop_reason``, whether ``[DONE]`` arrived, and the
    reasoning field name seen (ruling P6; ``None`` when no reasoning delta arrived)."""
    chunks: list[dict[str, Any]] = []
    reasoning: list[str] = []
    content: list[str] = []
    finish_reason: Any = None
    stop_reason: Any = None
    done = False
    field_seen: str | None = None
    fields_seen: list[str] = []
    delta_keys: set[str] = set()
    choice_keys: set[str] = set()
    top_keys: set[str] = set()
    prompt_ids: list[int] | None = None
    bad: list[str] = []
    error: Any = None
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            done = True
            continue
        try:
            obj = json.loads(data)
        except ValueError:
            bad.append(line)
            continue
        if not isinstance(obj, dict):
            bad.append(line)
            continue
        top_keys |= set(obj)
        if "error" in obj and error is None:
            error = obj["error"]
        if isinstance(obj.get("prompt_token_ids"), list) and prompt_ids is None:
            prompt_ids = [int(t) for t in obj["prompt_token_ids"]]
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue
        choice = choices[0]
        choice_keys |= set(choice)
        delta = _dict(choice.get("delta"))
        delta_keys |= set(delta)
        field, r = _reasoning_of(delta)
        if field is not None:
            if field_seen is None:
                field_seen = field
            if field not in fields_seen:
                fields_seen.append(field)
        c = delta.get("content") if isinstance(delta.get("content"), str) else None
        ids = choice.get("token_ids")
        chunks.append(
            {
                "reasoning": r,
                "content": c,
                "token_ids": [int(t) for t in ids] if isinstance(ids, list) else None,
            }
        )
        if r is not None:
            reasoning.append(r)
        if c is not None:
            content.append(c)
        if choice.get("finish_reason") is not None:
            finish_reason = choice.get("finish_reason")
            stop_reason = choice.get("stop_reason")
    return {
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "chunks": chunks,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "done": done,
        "reasoning_field": field_seen,
        "reasoning_fields": fields_seen,
        "delta_keys": sorted(delta_keys),
        "choice_keys": sorted(choice_keys),
        "top_keys": sorted(top_keys),
        "prompt_token_ids": prompt_ids,
        "unparsed_lines": len(bad),
        "error": error,
    }


def _until_stop(ids: Sequence[int]) -> list[int]:
    out: list[int] = []
    for t in ids:
        if int(t) in replay.STOP_IDS:
            break
        out.append(int(t))
    return out


def host_decode(ids: Sequence[int]) -> str:
    """``replay.detokenize`` of the ids before the first stop id (1, 106, 50)."""
    return replay.detokenize(_until_stop(ids))


def regions(text: str) -> tuple[str | None, str | None]:
    """``(thought, answer)`` of a host decode: the text between ``<|channel>thought\\n`` and the next ``<channel|>``,
    and the text after that ``<channel|>``. No header: no thought, the whole text is the answer. A header that
    is never closed: neither region exists."""
    i = text.find(THOUGHT_HEADER)
    if i < 0:
        return None, text
    j = text.find(CHANNEL_CLOSE, i + len(THOUGHT_HEADER))
    if j < 0:
        return None, None
    return text[i + len(THOUGHT_HEADER) : j], text[j + len(CHANNEL_CLOSE) :]


def reasoning_separated(reasoning: str | None, content: str | None, ids: Sequence[int]) -> bool:
    """q3: ``reasoning`` equals the thought region of the host decode of ``ids``, and the thought is not in
    ``content``."""
    thought, _ = regions(host_decode(ids))
    if reasoning is None or thought is None:
        return False
    return reasoning == thought and (thought == "" or thought not in (content or ""))


def content_identical(content: str | None, ids: Sequence[int]) -> bool:
    """q4: ``content`` is byte-identical to the answer region of the host decode of ``ids``."""
    _, answer = regions(host_decode(ids))
    return answer is not None and (content or "") == answer


def special_pieces() -> tuple[str, ...]:
    """The pinned tokenizer's added tokens with ``special``, longest first."""
    added = vocab.load_tokenizer().get_added_tokens_decoder().values()
    return tuple(sorted({str(a.content) for a in added if a.special}, key=lambda s: (-len(s), s)))


def split_special_pieces(texts: Sequence[str]) -> int:
    """q6: how many special-piece occurrences in the joined ``texts`` straddle a boundary between two texts."""
    joined = "".join(texts)
    cuts: set[int] = set()
    pos = 0
    for t in texts[:-1]:
        pos += len(t)
        cuts.add(pos)
    pattern = re.compile("|".join(re.escape(p) for p in special_pieces()))
    return sum(1 for m in pattern.finditer(joined) if any(m.start() < c < m.end() for c in cuts))


# --------------------------------------------------------------------------- the reading (criterion 6)


def _arm_qualifies(rows: Mapping[str, Mapping[str, Any]]) -> tuple[bool, list[str]]:
    why: list[str] = []
    tool = [cid for cid, r in rows.items() if r.get("kind") == "tool"]
    if len(rows) != N_CASES or len(tool) != N_TOOL_CASES:
        why.append(f"{len(rows)} cases ({len(tool)} tool), not {N_CASES} ({N_TOOL_CASES})")
    for cid, r in rows.items():
        if r.get("status") != 200:
            why.append(f"{cid}: status {r.get('status')}")
        # ruling P11: a truncated or errored response never qualifies (any non-null finish_reason is finished)
        if r.get("done") is not True:
            why.append(f"{cid}: not done")
        if r.get("finish_reason") is None:
            why.append(f"{cid}: no finish_reason")
        if r.get("error") is not None:
            why.append(f"{cid}: in-band error")
        if r.get("q3") is not True:
            why.append(f"{cid}: q3")
        if r.get("kind") == "tool" and r.get("q4") is not True:
            why.append(f"{cid}: q4")
        if r.get("split_specials") != 0:
            why.append(f"{cid}: {r.get('split_specials')} split special pieces")
    return not why, why


def _is_stop_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x in STOP_INTS


def _q8_ok(row: Mapping[str, Any]) -> bool:
    hits, queries = row.get("hits"), row.get("queries")
    if not isinstance(hits, int) or not isinstance(queries, int) or queries <= 0:
        return False
    return Fraction(hits, queries) >= Q8_MIN


def reading(results: Mapping[str, Any]) -> dict[str, Any]:
    """Criterion 6 with rulings P3-P5, on the probe's results (shape: ``tests/cd/unit/test_t0087_probe_offline``).

    Reading A: (b) or (c) is 200, q3 on all 12, q4 on the 10 tool cases and no split piece; (b) preferred. The
    stop id is ``stop_reason`` iff the chosen arm's every tool case gives an int in {1, 106, 50} (under reading B no
    arm is chosen, so T-0091 is replanned). Tools in the prefill iff every q8 case is at or above 0.9.
    Continuation content ``string`` iff all ``Q9_N`` string continuations are 200 with the prompt ending in 48
    (else replan, never a silent switch to list). Seed tokenizer ``tokenize`` iff q10 is equal on 10/10.
    Ruling P18: a q8 row with unreadable ``/metrics`` makes ``tools_in_prefill`` a re-run. Ruling P17: with a
    transport error (``complete: false``) every decision, ``fallback`` and ``streamed_ids`` read re-run, ``arm`` is
    ``None``, and the would-be values are kept under ``why.undecided``."""
    arms = results.get("arms") or {}
    why: dict[str, Any] = {}
    chosen: str | None = None
    for arm in READING_ARMS:
        ok, reasons = _arm_qualifies(arms.get(arm) or {})
        why[f"arm_{arm}"] = reasons
        if ok and chosen is None:
            chosen = arm
    if chosen is None:
        stop_ok = False
        why["stop_id"] = "no arm chosen (reading B)"
    else:
        bad = {
            cid: r.get("stop_reason")
            for cid, r in arms[chosen].items()
            if r.get("kind") == "tool" and not _is_stop_int(r.get("stop_reason"))
        }
        stop_ok = not bad
        why["stop_id"] = {k: repr(v) for k, v in bad.items()}
    q8 = list(results.get("q8") or [])
    tools_ok = len(q8) == len(Q8_TOOLS_CASES) and all(_q8_ok(r) for r in q8)
    # ruling P18: an unreadable /metrics on any q8 row is a re-run, not a measured False
    q8_unreadable = any(r.get("hits") is None or r.get("queries") is None for r in q8)
    tools_decision: Any = "re-run (human): q8 metrics unreadable" if q8_unreadable else tools_ok
    q9s = list((results.get("q9") or {}).get("string") or [])
    string_ok = len(q9s) == Q9_N and all(
        r.get("status") == 200 and r.get("prompt_ends_48") is True for r in q9s
    )
    q10 = list(results.get("q10") or [])
    tok_ok = len(q10) == Q10_N and all(x is True for x in q10)
    # ruling P13: B is a real fallback only if some stream arm carried per-chunk ids on all 12 cases
    streamed = [
        arm
        for arm in STREAM_ID_ARMS
        if len(arms.get(arm) or {}) == N_CASES
        and all(r.get("q7") is True for r in (arms.get(arm) or {}).values())
    ]
    why["streamed_ids_arms"] = streamed
    if chosen is not None:
        fallback = None
    elif streamed:
        fallback = "replan T-0088 on streamed token ids with host detokenisation (H10)"
    else:
        fallback = "escalate to human: reading B with no streamed ids, so H10's replan has nothing to build on"
    out: dict[str, Any] = {
        "reading": "A" if chosen is not None else "B",
        "arm": chosen,
        "stop_id": "stop_reason" if stop_ok else "replan T-0091",
        "tools_in_prefill": tools_decision,
        "continuation_content": "string" if string_ok else "replan T-0091 (ruling P5)",
        "seed_tokenizer": "tokenize"
        if tok_ok
        else "H9 alternative (tokenizers + pinned tokenizer.json)",
        "streamed_ids": bool(streamed),
        "fallback": fallback,
        "complete": True,
    }
    # ruling P11: a run with a transport error decides nothing
    errors = results.get("transport_errors") or 0
    if errors:
        why["undecided"] = {k: out[k] for k in (*DECISIONS, *UNDECIDED_EXTRA)}
        why["transport_errors"] = errors
        for k in (*DECISIONS, "fallback", "streamed_ids"):
            out[k] = f"re-run (human): {errors} transport error(s)"
        out["arm"] = None
        out["complete"] = False
    out["why"] = why
    return out


# --------------------------------------------------------------------------- I/O


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Ruling P15: a 3xx is recorded as the status (urllib raises it as an ``HTTPError``), never followed."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
#: Transport failures: the socket's, and http.client's (a response cut short is not always an ``OSError``).
_TRANSPORT = (OSError, ValueError, http.client.HTTPException)


def check_loopback(base_url: str) -> None:
    """Ruling P15: ``http(s)`` to an IP literal in 127.0.0.0/8 or ``::1``; any name (``localhost`` too) is refused
    without being resolved. ``ValueError`` otherwise."""
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"base_url must be http(s): {base_url!r}")
    host = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"base_url host {host!r} is not a loopback IP literal") from None
    ok = (
        addr in ipaddress.ip_network("127.0.0.0/8")
        if addr.version == 4
        else addr == ipaddress.ip_address("::1")
    )
    if not ok:
        raise ValueError(f"base_url host {host!r} is not in 127.0.0.0/8 or ::1")


def _root(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def _post_json(url: str, body: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
    """``{"status", "payload", "text", "error", "wall_ms"}`` for a non-stream POST (an HTTP error keeps its body)."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    status: int | None = None
    text = ""
    error: str | None = None
    try:
        with _OPENER.open(req, timeout=timeout_s) as resp:
            status = int(resp.status)
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        text = exc.read().decode("utf-8", errors="replace")
    except _TRANSPORT as exc:
        error = f"{type(exc).__name__}: {exc}"
    payload: Any = None
    if text:
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
    return {
        "status": status,
        "payload": payload,
        "text": _scrub(text),
        "error": error,
        "wall_ms": (time.perf_counter() - t0) * 1000.0,
    }


def _post_stream(url: str, body: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
    """``{"status", "lines", "error", "wall_ms"}``: every raw line of the response, as received."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.perf_counter()
    status: int | None = None
    lines: list[str] = []
    error: str | None = None
    try:
        with _OPENER.open(req, timeout=timeout_s) as resp:
            status = int(resp.status)
            for raw in resp:
                lines.append(_scrub(raw.decode("utf-8", errors="replace").rstrip("\r\n")))
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        lines = _scrub(exc.read().decode("utf-8", errors="replace")).splitlines()
    except _TRANSPORT as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "status": status,
        "lines": lines,
        "error": error,
        "wall_ms": (time.perf_counter() - t0) * 1000.0,
    }


def _get_json(url: str, timeout_s: float = 10.0) -> Any:
    try:
        with _OPENER.open(url, timeout=timeout_s) as resp:
            return json.loads(resp.read())
    except (OSError, ValueError):
        return None


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


# --------------------------------------------------------------------------- the probe


class _Probe:
    def __init__(
        self,
        base_url: str,
        model: str,
        set_dir: Path,
        cases: Mapping[str, Any],
        raw: Any,
        timeout_s: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.set_dir = set_dir
        self.cases = cases
        self.raw = raw
        self.timeout_s = timeout_s
        self.tools = _tools.load()
        self.transport_errors = 0

    def _log(self, record: Mapping[str, Any]) -> None:
        self.raw.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.raw.flush()

    def base(self, cid: str) -> dict[str, Any]:
        return twopass.native_pass1_body(self.model, self.cases[cid], self.set_dir, self.tools)

    def chat(self, what: str, cid: str, arm: str | None, body: Mapping[str, Any]) -> dict[str, Any]:
        url = self.base_url + "/chat/completions"
        if body.get("stream"):
            res = _post_stream(url, body, self.timeout_s)
            rec = {"lines": res["lines"]}
        else:
            res = _post_json(url, body, self.timeout_s)
            rec = (
                {"payload": redact(res["payload"])}
                if res["payload"] is not None
                else {"text": res["text"]}
            )
        if res["error"] is not None:
            self.transport_errors += 1
        self._log(
            {
                "what": what,
                "case": cid,
                "arm": arm,
                "path": "/chat/completions",
                "request": redact(body),
                "status": res["status"],
                "error": res["error"],
                "wall_ms": round(res["wall_ms"], 1),
                **rec,
            }
        )
        return res

    # q8 ----------------------------------------------------------------------------------------------------
    def q8_pair(self, cid: str, *, with_tools: bool) -> dict[str, Any]:
        gen = arm_body(self.base(cid), "d")
        pre = copy.deepcopy(gen)
        pre["max_tokens"] = 1
        if not with_tools:
            pre.pop("tools", None)
            pre.pop("tool_choice", None)
        m0 = twopass._metrics(self.base_url)
        p = self.chat("q8_prefill" if with_tools else "q8_control_prefill", cid, None, pre)
        m1 = twopass._metrics(self.base_url)
        g = self.chat("q8_generate" if with_tools else "q8_control_generate", cid, None, gen)
        m2 = twopass._metrics(self.base_url)
        hits, queries = twopass._delta(m1, m2)
        return {
            "case": cid,
            "with_tools": with_tools,
            "prefill_status": p["status"],
            "generate_status": g["status"],
            "prefill": dict(zip(("hits", "queries"), twopass._delta(m0, m1), strict=True)),
            "hits": hits,
            "queries": queries,
            "rate": None if hits is None or not queries else round(hits / queries, 4),
            "metrics": [m0, m1, m2],
        }

    # arms --------------------------------------------------------------------------------------------------
    def arm(self, cid: str, arm: str) -> dict[str, Any]:
        body = arm_body(self.base(cid), arm)
        res = self.chat("arm", cid, arm, body)
        row: dict[str, Any] = {"status": res["status"], "error": res["error"]}
        if arm == "d":
            payload = _dict(res["payload"])
            choices = payload.get("choices")
            choice = _dict(choices[0] if isinstance(choices, list) and choices else None)
            msg = _dict(choice.get("message"))
            field, r = _reasoning_of(msg)
            ids = choice.get("token_ids")
            row.update(
                {
                    "done": res["status"] == 200 and bool(choice),
                    "reasoning_field": field,
                    "message_keys": sorted(msg),
                    "choice_keys": sorted(choice),
                    "top_keys": sorted(payload),
                    "reasoning": r,
                    "content": msg.get("content") if isinstance(msg.get("content"), str) else None,
                    "finish_reason": choice.get("finish_reason"),
                    "stop_reason": choice.get("stop_reason"),
                    "error": redact(
                        payload if payload.get("object") == "error" else payload.get("error")
                    ),
                    "ids": [int(t) for t in ids] if isinstance(ids, list) else None,
                    "content_chunks": None,
                }
            )
        else:
            got = reassemble(res["lines"])
            flat = [t for c in got["chunks"] for t in (c["token_ids"] or [])]
            texts = [c["content"] for c in got["chunks"] if c["content"]]
            row.update(
                {
                    "done": got["done"],
                    "reasoning_field": got["reasoning_field"],
                    "reasoning_fields": got["reasoning_fields"],
                    "delta_keys": got["delta_keys"],
                    "choice_keys": got["choice_keys"],
                    "top_keys": got["top_keys"],
                    "reasoning": got["reasoning"] if got["reasoning_field"] is not None else None,
                    "content": got["content"],
                    "finish_reason": got["finish_reason"],
                    "stop_reason": got["stop_reason"],
                    "ids": flat
                    if arm != "a" and any(c["token_ids"] is not None for c in got["chunks"])
                    else None,
                    "chunks": len(got["chunks"]),
                    "chunks_with_token_ids": sum(c["token_ids"] is not None for c in got["chunks"]),
                    "content_chunks": len(texts),
                    "split_specials": split_special_pieces(texts),
                    "reasoning_split_specials": split_special_pieces(
                        [c["reasoning"] for c in got["chunks"] if c["reasoning"]]
                    ),
                    "unparsed_lines": got["unparsed_lines"],
                    "error": redact(got["error"]),
                    "chunk_contents": [c["content"] or "" for c in got["chunks"]],
                }
            )
        return row


def _finish(rows: dict[str, dict[str, dict[str, Any]]], cases: Mapping[str, Any]) -> None:
    """q3-q7 per arm and case (ruling P7): (b)-(e) against their own ids; (a) has no ids, so its q3/q4 are ``None``
    and its comparison with arm (d)'s decode is the ``vs_d`` diagnostic, labelled noise-confounded."""
    for cid in CASE_IDS:
        d_ids = rows["d"][cid].get("ids")
        b_ids = rows["b"][cid].get("ids")
        e_ids = rows["e"][cid].get("ids")
        for arm in ARMS:
            r = rows[arm][cid]
            r["kind"] = "tool" if cases[cid].get("tool") is not None else "none"
            ref = r.get("ids")
            if arm == "a":
                r["q3"] = r["q4"] = r["q3_stripped"] = None
                r["ref_canvases"] = math.ceil(len(d_ids) / twopass.CANVAS_LENGTH) if d_ids else None
                r["vs_d"] = {
                    "q3": bool(d_ids)
                    and reasoning_separated(r.get("reasoning"), r.get("content"), d_ids or []),
                    "q4": bool(d_ids) and content_identical(r.get("content"), d_ids or []),
                    "label": "noise-confounded",
                }
            elif ref:
                r["q3"] = reasoning_separated(r.get("reasoning"), r.get("content"), ref)
                r["q4"] = content_identical(r.get("content"), ref)
                thought, _ = regions(host_decode(ref))
                r["q3_stripped"] = (
                    r.get("reasoning") is not None
                    and thought is not None
                    and (r.get("reasoning") or "").strip() == thought.strip()
                )
                r["ref_canvases"] = math.ceil(len(ref) / twopass.CANVAS_LENGTH)
            else:
                r["q3"] = r["q4"] = r["q3_stripped"] = False
                r["ref_canvases"] = None
            r["q5_stop_id_in_ids"] = None if r.get("ids") is None else TOOL_RESPONSE_ID in r["ids"]
            r["q5_stop_text_in_content"] = TOOL_RESPONSE in (r.get("content") or "")
            if arm == "d":
                r["split_specials"] = None
            if r.get("content_chunks") is not None and r["ref_canvases"]:
                r["chunks_per_canvas"] = round(r["content_chunks"] / r["ref_canvases"], 2)
            if arm in ("b", "c", "e"):
                r["q7"] = bool(r.get("chunks_with_token_ids"))
            if arm == "e":
                r["b_ids_equal_e_ids"] = b_ids is not None and e_ids is not None and b_ids == e_ids


def _summary_md(report: Mapping[str, Any]) -> str:
    rows = report["arms"]
    out = [f"# T-0087 native stream probe at {report['regime']['commit']}", ""]
    out.append("## Regime")
    for k, v in report["regime"].items():
        out.append(f"- {k}: `{v}`")
    out += [
        "",
        "## Run",
        f"- transport_errors: {report['transport_errors']}",
        f"- complete: {report['reading'].get('complete')}",
    ]
    out += ["", "## Arms (q1-q7)", ""]
    out.append(
        "| arm | q1 200 | q2 done | q3 | q4 tool | q4 none | q5 stop_reason | q6 splits | q7 ids | field |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        rs = rows[arm]
        tool = [r for r in rs.values() if r["kind"] == "tool"]
        none = [r for r in rs.values() if r["kind"] != "tool"]
        stops = sorted({repr(r.get("stop_reason")) for r in rs.values()})
        splits = sum(r.get("split_specials") or 0 for r in rs.values())
        fields = sorted({str(r.get("reasoning_field")) for r in rs.values()})
        q7 = sum(bool(r.get("q7")) for r in rs.values()) if arm in ("b", "c", "e") else "-"
        out.append(
            f"| {arm} | {sum(r['status'] == 200 for r in rs.values())}/{len(rs)} "
            f"| {sum(bool(r.get('done')) for r in rs.values())}/{len(rs)} "
            f"| {sum(bool(r.get('q3')) for r in rs.values())}/{len(rs)} "
            f"| {sum(bool(r.get('q4')) for r in tool)}/{len(tool)} "
            f"| {sum(bool(r.get('q4')) for r in none)}/{len(none)} "
            f"| {', '.join(stops)} | {splits if arm != 'd' else '-'} | {q7} | {', '.join(fields)} |"
        )
    det = [cid for cid, r in rows["e"].items() if r.get("b_ids_equal_e_ids")]
    vs = [r.get("vs_d") or {} for r in rows["a"].values()]
    out += [
        "",
        "Arm (a) has no ids; its q3/q4 are not measured (ruling P7).",
        f"Arm (a) vs arm (d)'s decode (noise-confounded, decides nothing): q3 "
        f"{sum(bool(v.get('q3')) for v in vs)}/{len(vs)}, q4 {sum(bool(v.get('q4')) for v in vs)}/{len(vs)}.",
        f"Replicate (arm (b) ids == arm (e) ids, noise level, decides nothing): "
        f"{len(det)}/{len(rows['e'])} cases.",
    ]
    out += ["", "## q8 prefix cache", ""]
    for r in report["q8"] + report["q8_control"]:
        tag = "with tools" if r["with_tools"] else "control (no tools in prefill)"
        out.append(f"- {r['case']} {tag}: hits {r['hits']} / queries {r['queries']} = {r['rate']}")
    out += ["", "## q9 T4 continuation", ""]
    for form in ("string", "list"):
        for r in report["q9"][form]:
            out.append(
                f"- {form} {r['case']}: status {r['status']}, prompt ends in 48: {r['prompt_ends_48']}"
            )
    out += ["", "## q10 /tokenize parity", ""]
    out.append(f"- path {report['q10_path']}, body keys {report['q10_body_keys']}")
    for r in report["q10_rows"]:
        out.append(f"- {r['case']}: equal {r['equal']} ({r['n_engine']} vs {r['n_host']} ids)")
    out += ["", "## Reading (criterion 6)", ""]
    for k, v in report["reading"].items():
        if k != "why":
            out.append(f"- {k}: `{v}`")
    if report["confounds"]:
        out += ["", "## Confounds", ""] + [f"- {c}" for c in report["confounds"]]
    return "\n".join(out) + "\n"


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--request-timeout-s", type=float, default=600.0)
    return ap


def _cases(set_dir: Path) -> dict[str, Any]:
    raw = (set_dir / "cases.jsonl").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != CASES_SHA256:
        raise ValueError(
            f"{set_dir / 'cases.jsonl'} has sha256 {sha}, not the pinned {CASES_SHA256}"
        )
    rows = {r["id"]: r for r in (json.loads(x) for x in raw.decode().splitlines() if x.strip())}
    for cid in (*CASE_IDS, *Q8_TOOLS_CASES, *Q8_CONTROL_CASES):
        if cid not in rows or rows[cid].get("split") != "dev":
            raise ValueError(f"case {cid} is not in the dev split")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    try:
        check_loopback(args.base_url)
    except ValueError as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    repo = Path(args.repo).resolve()
    try:
        dirty = _git(repo, "status", "--porcelain").rstrip("\n")
        sha = _git(repo, "rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _say(f"refusing: {repo} is not a readable git tree ({exc})")
        return EXIT_REFUSED
    if dirty:
        _say(f"refusing: the tree at {repo} is dirty:\n{dirty}")
        return EXIT_REFUSED
    out: Path = args.out_dir or repo / "logs" / f"t0087-probe-{sha[:7]}"
    if out.exists() and any(out.iterdir()):
        _say(f"refusing: {out} exists and is not empty; move it aside first")
        return EXIT_REFUSED
    set_dir = repo / SET_REL
    try:
        cases = _cases(set_dir)
    except (OSError, ValueError) as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED

    models = _get_json(args.base_url.rstrip("/") + "/models")
    try:
        model = str(models["data"][0]["id"])
    except (KeyError, IndexError, TypeError):
        _say(f"no engine answering GET {args.base_url}/models; the locked script boots it")
        return EXIT_STOPPED
    out.mkdir(parents=True, exist_ok=True)
    regime = {
        "commit": sha,
        "base_url": args.base_url,
        "served_model": model,
        "engine_version": _get_json(_root(args.base_url) + "/version"),
        "versions_lock_sha256": _sha256(repo / "services" / "versions.lock"),
        "model_revision": vocab.MODEL_REVISION,
        "tokenizer_sha256": vocab.TOKENIZER_SHA256,
        "cases_sha256": CASES_SHA256,
        "tools_v1_sha256": _tools.load().version,
        "case_ids": list(CASE_IDS),
        "arms": {arm: {k: v for k, v in arm_body({}, arm).items()} for arm in ARMS},
        "pass1_max_tokens": twopass.PASS1_MAX_TOKENS,
        "enable_thinking": True,
        "n_per_arm_case": 1,
        "seed": None,
        "concurrency": 1,
        "q8_tools_cases": list(Q8_TOOLS_CASES),
        "q8_control_cases": list(Q8_CONTROL_CASES),
        "order": "q8 (with tools, then control), arms a..e per case, q9, q10",
    }
    with (out / "raw.jsonl").open("w", encoding="utf-8") as raw:
        probe = _Probe(args.base_url, model, set_dir, cases, raw, args.request_timeout_s)
        q8 = [probe.q8_pair(cid, with_tools=True) for cid in Q8_TOOLS_CASES]
        q8c = [probe.q8_pair(cid, with_tools=False) for cid in Q8_CONTROL_CASES]

        rows: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
        for cid in CASE_IDS:
            for arm in ARMS:
                rows[arm][cid] = probe.arm(cid, arm)
        _finish(rows, cases)

        routes = {
            cid: router.route(host_decode(rows["d"][cid]["ids"] or []))
            for cid in CASE_IDS
            if rows["d"][cid]["kind"] == "tool"
        }
        q9: dict[str, list[dict[str, Any]]] = {"string": [], "list": []}
        q9_cases = [cid for cid, rt in routes.items() if rt.tag == "native"][:Q9_N]
        for cid in q9_cases:
            continuation = routes[cid].spoken_prefix + twopass.TOOL_CALL_OPEN
            for form in ("string", "list"):
                body = arm_body(probe.base(cid), "d")
                body["max_tokens"] = twopass.PASS2_MAX_TOKENS["T4"]
                content: Any = (
                    continuation if form == "string" else [{"type": "text", "text": continuation}]
                )
                body["messages"].append({"role": "assistant", "content": content})
                body["continue_final_message"] = True
                body["add_generation_prompt"] = False
                res = probe.chat(f"q9_{form}", cid, None, body)
                prompt = (
                    twopass._prompt_ids(res["payload"])
                    if isinstance(res["payload"], dict)
                    else None
                )
                q9[form].append(
                    {
                        "case": cid,
                        "status": res["status"],
                        "prompt_ends_48": prompt is not None
                        and len(prompt) > 0
                        and prompt[-1] == TOOL_CALL_OPEN_ID,
                        "prompt_tail": prompt[-8:] if prompt else None,
                        "continuation_guard": twopass.continuation_guard(
                            continuation, twopass.TOOL_CALL_OPEN
                        ),
                    }
                )

        q10_rows: list[dict[str, Any]] = []
        q10_path: str | None = None
        q10_body_keys: list[str] = []
        for cid, rt in routes.items():
            draft = rt.draft_call_text.rstrip() if rt.tag == "native" else ""
            if not draft:
                q10_rows.append(
                    {
                        "case": cid,
                        "equal": None,
                        "n_engine": None,
                        "n_host": None,
                        "why": "no native draft",
                    }
                )
                continue
            tbody = {"model": model, "prompt": draft, "add_special_tokens": False}
            q10_body_keys = sorted(tbody)
            tres: dict[str, Any] = {}
            for path in [q10_path] if q10_path else list(TOKENIZE_PATHS):
                tres = _post_json(_root(args.base_url) + path, tbody, args.request_timeout_s)
                probe._log(
                    {
                        "what": "q10",
                        "case": cid,
                        "path": path,
                        "request": tbody,
                        "status": tres["status"],
                        "error": tres["error"],
                        "payload": redact(tres["payload"]),
                    }
                )
                if tres["status"] == 200:
                    q10_path = path
                    break
            tokens = (
                tres.get("payload", {}).get("tokens")
                if isinstance(tres.get("payload"), dict)
                else None
            )
            host = vocab.encode(draft)
            q10_rows.append(
                {
                    "case": cid,
                    "status": tres.get("status"),
                    "equal": isinstance(tokens, list) and [int(t) for t in tokens] == host,
                    "n_engine": len(tokens) if isinstance(tokens, list) else None,
                    "n_host": len(host),
                }
            )

    results = {
        "arms": {
            arm: {
                cid: {
                    "kind": r["kind"],
                    "status": r["status"],
                    "q3": r["q3"],
                    "q4": r["q4"],
                    "split_specials": r.get("split_specials"),
                    "stop_reason": r.get("stop_reason"),
                    "done": r.get("done") is True,
                    "finish_reason": r.get("finish_reason"),
                    "error": r.get("error"),
                    "q7": r.get("q7") is True,
                }
                for cid, r in rows[arm].items()
            }
            for arm in ARMS
        },
        "q8": [{"hits": r["hits"], "queries": r["queries"]} for r in q8],
        "q9": {
            form: [{"status": r["status"], "prompt_ends_48": r["prompt_ends_48"]} for r in q9[form]]
            for form in q9
        },
        "q10": [r["equal"] is True for r in q10_rows if r["equal"] is not None],
        "transport_errors": probe.transport_errors,
    }
    verdict = reading(results)
    confounds: list[str] = []
    for arm in ("b", "c", "d", "e"):
        missing = [cid for cid in CASE_IDS if not rows[arm][cid].get("ids")]
        if missing:
            confounds.append(
                f"arm ({arm}) carried no ids of its own on {len(missing)}/{len(CASE_IDS)} cases "
                f"({', '.join(missing)}): q3/q4 there are False for want of ids, not on the served text"
            )
    report = {
        "regime": regime,
        "arms": {
            arm: {
                cid: {k: v for k, v in r.items() if k not in ("ids",)}
                for cid, r in rows[arm].items()
            }
            for arm in ARMS
        },
        "ids": {
            arm: {cid: rows[arm][cid].get("ids") for cid in CASE_IDS}
            for arm in ("b", "c", "d", "e")
        },
        "q8": q8,
        "q8_control": q8c,
        "q9": q9,
        "q10_path": q10_path,
        "q10_body_keys": q10_body_keys,
        "q10_rows": q10_rows,
        "results": results,
        "reading": verdict,
        "confounds": confounds,
        "transport_errors": probe.transport_errors,
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "report.md").write_text(_summary_md(report), encoding="utf-8")
    _say(f"reading {verdict['reading']} (arm {verdict['arm']}); written to {out}")
    return EXIT_STOPPED if probe.transport_errors else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
