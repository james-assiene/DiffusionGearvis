"""The live tool-call judge and the H-T84-2 repair builder (T-0091 criteria 1-6). Pure apart from two file reads.

- ``load_payloads(path=None)`` reads the precompiled native T4 payloads (``toolcall/payloads/native_T4.json``,
  written by ``scripts/build_toolcall_payloads.py`` only) and checks them: the file's tools sha is
  ``tools.load().version``, every v1 tool has exactly one payload whose ``automaton_sha256`` is the sha256 of its
  ``automaton``, and every bucket is at most ``LIVE_MAX_BUCKET``. Any failure raises ``PayloadStale`` (and only
  that: T-0092 turns repair off on it).
- ``load_calibration(path=None)`` reads the frozen ``perstep_mar`` calibration ``docs/cd/calibration-bfcl.json``,
  whose file sha256 must be ``CALIBRATION_SHA256``; a missing, unreadable or other file raises ``PayloadStale`` (F10).
- ``classify(tool_text)`` judges a pass-1 ``tool`` commit: ``valid`` iff ``native.parse_call`` succeeds and
  ``score.schema_valid`` holds; ``repairable`` iff not valid and ``native.draft_name`` is a v1 name; ``refused``
  otherwise, with the parse reason, or ``"no tool name"`` when the call parses but names no v1 tool.
  ``query_problem`` is ``score.query_problem`` of a parsed ``google_search`` query, log-only (H7).
  T-0100 narrows ``repairable`` (H-T99d, H-T99e): a not-valid ``press_key`` draft is ``refused`` with
  ``NO_COMBO`` when no combo can be read and ``COMBO_NOT_IN_ENUM`` when the combo is not a string whose
  normalised form (whitespace squashed, casefolded, each ``+`` part mapped through ``COMBO_ALIASES``) is a
  squashed list entry; a not-valid ``click_at``/``type_at`` draft whose read-back ``x`` or ``y`` is not an
  integer within the schema's bounds (0-999) is ``refused`` with ``COORDINATE_OUT_OF_RANGE`` (a missing or
  unreadable coordinate leaves today's verdict). The values are the parsed arguments, or, for a draft that does
  not parse, a string-aware scan of its top-level keys (``_scan``); such a refusal carries ``arguments=None``.
  The list entries and the bounds are read from ``tools.load()``; ``scroll``/``google_search`` are untouched.
  T-0102 narrows it again (H-T101c), after T-0100's rules: a not-valid draft of any v1 tool whose call has an end
  E (``_call_end``: a string-aware depth walk from ``call:NAME{``; the closing brace, plus a following
  ``<tool_call|>``, or a call tag) and whose remainder after E is not a second call and holds an alphanumeric
  character once every ``SPECIAL_TOKENS`` literal is replaced by a space is ``refused`` with
  ``MODEL_TEXT_AFTER_CALL``, ``arguments=None``. A clean cut (no E) and a benign remainder stay ``repairable``.
  T-0196: when that verdict is not ``valid`` and the body is a call whose first ``<tool_call|>`` outside strings is
  followed only by more ``<tool_call|>`` literals and whitespace (``_single_close``), the body cut after that first
  close is judged too, and its verdict is adopted only if it is ``valid`` (``repair_verdict`` stays strict).
  T-0215 (R-1): with a ``tool_set``, a not-valid draft of a v1 tool the set does not declare is ``refused`` with
  ``NOT_DECLARED`` before every other rule (a set that declares every v1 name, and ``None``, are unchanged).
- ``repair_prompt(pass1, spoken_prefix)`` continues pass 1 with ``assistant(spoken_prefix + "<|tool_call>")``, or
  refuses with the T-0085 continuation guard (``template`` before ``marker``).
- ``repair_request(...)`` is the seeded ``perstep_mar`` ``CdRequest`` the eval sent for
  ``native/T4/named/perstep_mar+seed``: ``seed = (draft_ids + [stop_id])[:256]``. The ids come in from the
  session (``LlmBackend.tokenize``); nothing here tokenises.
- ``repair_verdict(text)`` judges the repair's output; a ``repairable`` result becomes ``refused`` (the T-0100
  rules apply there too: they live in ``_judge``).

Nothing here imports ``local_astra.decoding``, the eval, numpy, torch, outlines or tokenizers: the parity with
``eval.cd.twopass`` is proven by ``tests/cd/unit/test_toolcall_live_parity.py``, not by import.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

from local_astra.contracts.cd import CdRequest
from local_astra.contracts.llm import LlmOptions, Message, Prompt, TextPart
from local_astra.toolcall import native, router, score, tools
from local_astra.toolcall.special_tokens import SPECIAL_IDS, SPECIAL_TOKENS, STOP_IDS

__all__ = [
    "CALIBRATION_SHA256",
    "CANVAS_LENGTH",
    "COMBO_ALIASES",
    "COMBO_NOT_IN_ENUM",
    "COORDINATE_OUT_OF_RANGE",
    "LIVE_MAX_BUCKET",
    "MODEL_TEXT_AFTER_CALL",
    "NOT_DECLARED",
    "NO_COMBO",
    "REPAIR_OPTIONS",
    "Calibration",
    "Payload",
    "PayloadStale",
    "Payloads",
    "Refusal",
    "Verdict",
    "classify",
    "load_calibration",
    "load_payloads",
    "repair_prompt",
    "repair_request",
    "repair_verdict",
]

_REPO: Final = Path(__file__).resolve().parents[2]
DEFAULT_PAYLOADS: Final = Path(__file__).resolve().parent / "payloads" / "native_T4.json"
DEFAULT_CALIBRATION: Final = _REPO / "docs" / "cd" / "calibration-bfcl.json"
#: The frozen ``perstep_mar`` BFCL calibration (T-0079 re-freeze; entropy bound 1.0, confidence threshold 0.005).
CALIBRATION_SHA256: Final = "4084026c580a3148b4351699af64832e22b8e756fb288b551a17e642585ba74a"
#: The live gate on |S| (T-0084a measured 29-80 states for T4).
LIVE_MAX_BUCKET: Final = 128
#: The engine-global canvas length: the seed is cut to it (``twopass.CANVAS_LENGTH``).
CANVAS_LENGTH: Final = 256
#: The repair's mode and its draft knobs (``twopass.SEED_BASE_ARM``, ``SEED_DRAFT_NOISE``,
#: ``SEED_DRAFT_START_STEP``: T-0081 tier 2, cell ``mar_n0_s24``).
REPAIR_MODE: Final = "perstep_mar"
DRAFT_NOISE: Final = 0.0
DRAFT_START_STEP: Final = 24
#: ``twopass.PASS2_MAX_TOKENS["T4"]``.
REPAIR_OPTIONS: Final = LlmOptions(max_tokens=256)

#: T-0100 refusal reasons (H-T99d: ``press_key``; H-T99e: ``click_at``/``type_at``).
COMBO_NOT_IN_ENUM: Final = "combo not in enum"
NO_COMBO: Final = "no combo"
COORDINATE_OUT_OF_RANGE: Final = "coordinate out of range"
#: T-0102 refusal reason (H-T101c): a not-valid draft with more model text after its call.
MODEL_TEXT_AFTER_CALL: Final = "model text after call"
#: T-0215 (R-1): a v1 name the judged set leaves out (``google_search`` with search off) is refused, never repaired.
NOT_DECLARED: Final = "not a declared tool"

_OPEN: Final = native.TOOL_CALL_OPEN
_STOP_TEXTS: Final = tuple(SPECIAL_IDS[i] for i in sorted(STOP_IDS))

Status = Literal["valid", "repairable", "refused"]


class PayloadStale(Exception):
    """The payload file does not match the tools, a payload does not match its sha, a v1 tool has none, a bucket
    is above the live gate, or the calibration file is missing, unreadable or not the pinned one (ruling F10)."""


@dataclass(frozen=True)
class Payload:
    automaton: str
    automaton_sha256: str
    n_states: int
    bucket: int


@dataclass(frozen=True)
class Payloads:
    """The checked payload file: one ``Payload`` per v1 tool, and the manifest it was built under."""

    tools: Mapping[str, Payload]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class Calibration:
    entropy_bound: float
    confidence_threshold: float
    sha256: str


@dataclass(frozen=True)
class Verdict:
    status: Status
    name: str | None
    arguments: dict[str, Any] | None
    reason: str | None
    query_problem: str | None


@dataclass(frozen=True)
class Refusal:
    """The continuation guard refused the repair prompt: ``template`` or ``marker``."""

    reason: Literal["template", "marker"]


# --------------------------------------------------------------------------- criterion 1: the payloads


def load_payloads(path: str | Path | None = None) -> Payloads:
    """Read and check the payload file; ``PayloadStale`` on any mismatch (module docstring)."""
    p = Path(path) if path is not None else DEFAULT_PAYLOADS
    try:
        doc = json.loads(p.read_bytes())
    except (OSError, ValueError) as exc:
        raise PayloadStale(f"{p}: unreadable payload file ({exc})") from exc
    if not isinstance(doc, dict):
        raise PayloadStale(f"{p}: not a JSON object")
    manifest, entries = doc.get("manifest"), doc.get("tools")
    if not isinstance(manifest, dict) or not isinstance(entries, dict):
        raise PayloadStale(f"{p}: no manifest or no tools")
    version = tools.load().version
    if manifest.get("tools_sha256") != version:
        raise PayloadStale(
            f"{p}: built for tools {manifest.get('tools_sha256')!r}, the tools are {version}"
        )
    v1 = tools.load().names
    if sorted(entries) != sorted(v1):
        raise PayloadStale(f"{p}: payloads for {sorted(entries)}, the v1 tools are {sorted(v1)}")
    out: dict[str, Payload] = {}
    for name in v1:
        e = entries[name]
        if not isinstance(e, dict):
            raise PayloadStale(f"{p}: {name}: not an object")
        a, sha = e.get("automaton"), e.get("automaton_sha256")
        n, bucket = e.get("n_states"), e.get("bucket")
        if not isinstance(a, str) or not a or not a.isascii() or not isinstance(sha, str):
            raise PayloadStale(f"{p}: {name}: no ASCII automaton or no sha")
        if hashlib.sha256(a.encode("ascii")).hexdigest() != sha:
            raise PayloadStale(f"{p}: {name}: the automaton is not its automaton_sha256")
        if (
            not isinstance(n, int)
            or not isinstance(bucket, int)
            or isinstance(n, bool)
            or isinstance(bucket, bool)
        ):
            raise PayloadStale(f"{p}: {name}: bad n_states/bucket ({n!r}/{bucket!r})")
        if not 0 < n <= bucket:
            raise PayloadStale(f"{p}: {name}: bad n_states/bucket ({n!r}/{bucket!r})")
        if bucket > LIVE_MAX_BUCKET:
            raise PayloadStale(
                f"{p}: {name}: bucket {bucket} is above the live gate {LIVE_MAX_BUCKET}"
            )
        out[name] = Payload(automaton=a, automaton_sha256=sha, n_states=n, bucket=bucket)
    return Payloads(tools=MappingProxyType(out), manifest=MappingProxyType(dict(manifest)))


def load_calibration(path: str | Path | None = None) -> Calibration:
    """The frozen ``perstep_mar`` knobs. ``PayloadStale`` (ruling F10) when the file is missing or unreadable or its
    sha256 is not ``CALIBRATION_SHA256``, so the session turns repair off instead of crashing."""
    p = Path(path) if path is not None else DEFAULT_CALIBRATION
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise PayloadStale(f"{p}: unreadable calibration file ({exc})") from exc
    sha = hashlib.sha256(raw).hexdigest()
    if sha != CALIBRATION_SHA256:
        raise PayloadStale(
            f"{p}: sha256 {sha} is not the frozen perstep_mar calibration {CALIBRATION_SHA256}"
        )
    data = json.loads(raw)
    return Calibration(
        entropy_bound=float(data["entropy_bound"]),
        confidence_threshold=float(data["confidence_threshold"]),
        sha256=sha,
    )


# --------------------------------------------------------------------------- criteria 2, 3 and 6: the verdicts


def _drop_tag(text: str) -> str:
    """A leading ``<|tool_call>`` dropped (after leading whitespace), as ``twopass.call_text_of``."""
    s = text.lstrip()
    return s[len(_OPEN) :] if s.startswith(_OPEN) else text


def _drop_stop(text: str) -> str:
    """A trailing stop text dropped, if any (a no-op on the live path: the client stops at the first stop id)."""
    for stop in _STOP_TEXTS:
        if text.endswith(stop):
            return text[: -len(stop)]
    return text


@dataclass(frozen=True)
class _RawText:
    """A bare token that is not a JSON literal (ruling V3): raw text, never a string value."""

    text: str


_DRAFT_HEAD: Final = re.compile(r"\A\s*call:([A-Za-z_][A-Za-z0-9_]*)\s*\{")
_KEY: Final = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*")
#: A bare token is ended by ``,``, ``}``, whitespace or ``<`` (T-0100 Definitions, "Value scan"), and by ``{``, ``[``,
#: ``]`` so that a nested value is depth-tracked, never swallowed as a token (ruling V4).
_BARE: Final = re.compile(r"[^,{}\[\]\s<]*")
_DELIM: Final = native.STRING_DELIM
_CLOSE: Final = native.TOOL_CALL_CLOSE


def _bare_value(tok: str) -> Any:
    """A non-empty bare token: a JSON literal (``true``/``false``/``null``/a number, read as ``native.parse_call``
    reads one) or ``_RawText`` (V3)."""
    ok, value = native._literal(tok)
    return value if ok else _RawText(tok)


def _scan(body: str) -> dict[str, Any]:
    """The value scan of a draft that does not parse (T-0100 Definitions): the first value of each top-level key.

    From after ``call:NAME{``: closed strings are values, never searched for keys; an unterminated string, a call tag
    outside a string, or the call's own closing brace ends the scan; a key is an identifier in key position (after
    ``{`` or ``,`` at depth 1, ruling V4) followed by ``\\s*:\\s*``; a value is a closed string or a bare token; a
    bare token running into the end of the body (a length cut) or an empty one (V5) gives no value. Never raises.
    """
    head = _DRAFT_HEAD.match(body)
    if head is None:
        return {}
    out: dict[str, Any] = {}
    n, i, depth, key_position = len(body), head.end(), 1, True
    while i < n:
        if body.startswith(_DELIM, i):
            end = body.find(_DELIM, i + len(_DELIM))
            if end < 0:
                break
            i, key_position = end + len(_DELIM), False
            continue
        if body.startswith(_CLOSE, i) or body.startswith(_OPEN, i):
            break
        ch = body[i]
        if ch in "{[":
            i, depth, key_position = i + 1, depth + 1, False
            continue
        if ch in "}]":
            i, depth, key_position = i + 1, depth - 1, False
            if depth <= 0:
                break
            continue
        if ch == ",":
            i, key_position = i + 1, True
            continue
        if ch.isspace():
            i += 1
            continue
        k = _KEY.match(body, i) if key_position and depth == 1 else None
        if k is None:
            i, key_position = i + 1, False
            continue
        key, j = k.group(1), k.end()
        if body.startswith(_DELIM, j):
            end = body.find(_DELIM, j + len(_DELIM))
            if end < 0:
                break  # unterminated: no value, and the scan ends
            out.setdefault(key, body[j + len(_DELIM) : end])
            i, key_position = end + len(_DELIM), False
            continue
        tok = _BARE.match(body, j)
        assert tok is not None  # the pattern matches the empty string
        if tok.end() >= n:
            break  # a length cut: the last token is unread
        if tok.group():
            out.setdefault(key, _bare_value(tok.group()))
        i, key_position = max(tok.end(), j), False
    return out


#: Q1: whole parts of a squashed combo mapped before the list lookup; ``cmd`` is not mapped.
COMBO_ALIASES: Final = MappingProxyType(
    {
        "esc": "escape",
        "return": "enter",
        "control": "ctrl",
        "del": "delete",
        "up": "arrowup",
        "down": "arrowdown",
        "left": "arrowleft",
        "right": "arrowright",
    }
)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _normalised(combo: str) -> str:
    """Q1: squash, split on ``+``, map each whole part through ``COMBO_ALIASES``, rejoin."""
    return "+".join(COMBO_ALIASES.get(p, p) for p in _squash(combo).split("+"))


def _combos_of(ts: tools.ToolSet) -> frozenset[str]:
    combo = ts.parameters("press_key")["properties"]["combo"]
    return frozenset(_squash(str(e)) for e in combo["enum"])


def _bounds_of(ts: tools.ToolSet, name: str, key: str) -> tuple[int, int]:
    prop = ts.parameters(name)["properties"][key]
    return int(prop["minimum"]), int(prop["maximum"])


@functools.cache
def _v1_combos() -> frozenset[str]:
    return _combos_of(tools.load())


@functools.cache
def _v1_bounds(name: str, key: str) -> tuple[int, int]:
    return _bounds_of(tools.load(), name, key)


def _squashed_combos(tool_set: tools.ToolSet | None = None) -> frozenset[str]:
    """The ``press_key`` list entries from ``tool_set`` (``None``: ``tools.load()``, cached), squashed."""
    return _v1_combos() if tool_set is None else _combos_of(tool_set)


def _coordinate_bounds(
    name: str, key: str, tool_set: tools.ToolSet | None = None
) -> tuple[int, int]:
    """``minimum``/``maximum`` of ``name``'s ``key`` from ``tool_set`` (``None``: ``tools.load()``, cached)."""
    return _v1_bounds(name, key) if tool_set is None else _bounds_of(tool_set, name, key)


def _in_range(name: str, key: str, v: Any, tool_set: tools.ToolSet | None = None) -> bool:
    """An integer as jsonschema counts (an ``int`` that is not a ``bool``, or an integral ``float``) within bounds."""
    integral = (isinstance(v, int) and not isinstance(v, bool)) or (
        isinstance(v, float) and v.is_integer()
    )
    lo, hi = _coordinate_bounds(name, key, tool_set)
    return integral and lo <= v <= hi


def _h99_refusal(
    name: str, values: Mapping[str, Any], tool_set: tools.ToolSet | None = None
) -> str | None:
    """H-T99d/H-T99e on a not-valid draft's read-back values: the refusal reason, or None (today's verdict)."""
    if name == "press_key":
        if "combo" not in values:
            return NO_COMBO
        combo = values["combo"]
        if not isinstance(combo, str) or _normalised(combo) not in _squashed_combos(tool_set):
            return COMBO_NOT_IN_ENUM
        return None
    if name in ("click_at", "type_at") and any(
        k in values and not _in_range(name, k, values[k], tool_set) for k in ("x", "y")
    ):
        return COORDINATE_OUT_OF_RANGE
    return None


def _call_end(body: str) -> int | None:
    """The call's end E (T-0102 Definitions): a string-aware depth walk from after ``call:NAME{``, or None for a
    clean cut (the body ends inside the call or inside a string). Linear; never raises."""
    head = _DRAFT_HEAD.match(body)
    if head is None:
        return None
    n, i, depth = len(body), head.end(), 1
    while i < n:
        if body.startswith(_DELIM, i):
            end = body.find(_DELIM, i + len(_DELIM))
            if end < 0:
                return None  # a cut inside a string is clean
            i = end + len(_DELIM)
            continue
        if body.startswith(_CLOSE, i):
            return i + len(_CLOSE)
        if body.startswith(_OPEN, i):
            return i
        ch = body[i]
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                j = i + 1
                while j < n and body[j].isspace():
                    j += 1
                return j + len(_CLOSE) if body.startswith(_CLOSE, j) else i + 1
        i += 1
    return None


def _model_text_after_call(body: str) -> bool:
    """H-T101c: the remainder after the call's end holds more model text -- not a second call, and with every
    special literal replaced by a space, at least one alphanumeric character."""
    end = _call_end(body)
    if end is None:
        return False
    rest = body[end:]
    if rest.lstrip().startswith(_OPEN):
        return False  # a second call keeps today's verdict (Q2)
    for lit in SPECIAL_TOKENS:
        rest = rest.replace(lit, " ")
    return any(ch.isalnum() for ch in rest)


def _judge(body: str, tool_set: tools.ToolSet | None = None) -> Verdict:
    """The verdict on the text after ``<|tool_call>``, against ``tool_set`` (``None``: ``tools.load()``).

    ``repairable`` still needs ``native.draft_name`` (v1 names only), so a not-valid draft of a tool outside v1
    (T-0140's ``page_overview``, which has no payload) is never ``repairable``."""
    parsed = native.parse_call(body)
    if isinstance(parsed, score.ParseFailure):
        name = native.draft_name(body)
        if name is not None:
            if tool_set is not None and name not in tool_set.names:
                return Verdict("refused", name, None, NOT_DECLARED, None)  # T-0215 (R-1)
            refusal = _h99_refusal(name, _scan(body), tool_set)
            if refusal is not None:  # V2: the scanned values decide, they are not reported
                return Verdict("refused", name, None, refusal, None)
            if _model_text_after_call(body):  # H-T101c (T-0102): after T-0100's rules
                return Verdict("refused", name, None, MODEL_TEXT_AFTER_CALL, None)
            return Verdict("repairable", name, None, parsed.reason, None)
        return Verdict("refused", None, None, parsed.reason, None)
    name, arguments = parsed["name"], parsed["arguments"]
    problem: str | None = None
    if name == "google_search" and isinstance(arguments.get("query"), str):
        problem = score.query_problem(arguments["query"])
    if score.schema_valid(parsed, tools.load() if tool_set is None else tool_set):
        return Verdict("valid", name, arguments, None, problem)
    if native.draft_name(body) is not None:
        if tool_set is not None and name not in tool_set.names:
            return Verdict("refused", name, arguments, NOT_DECLARED, problem)  # T-0215 (R-1)
        refusal = _h99_refusal(name, arguments, tool_set)
        if refusal is not None:
            return Verdict("refused", name, arguments, refusal, problem)
        return Verdict("repairable", name, arguments, "schema invalid", problem)
    return Verdict("refused", None, None, "no tool name", None)


def _single_close(body: str) -> str | None:
    """T-0196: ``body`` up to and including its first ``<tool_call|>`` outside every ``<|"|>`` string, if what
    follows is non-empty, holds at least one ``<tool_call|>`` and only ``<tool_call|>`` literals and whitespace;
    otherwise ``None``. One left-to-right pass (linear, total on any ``str``, no regex)."""
    i, n = 0, len(body)
    while i < n:
        if body.startswith(_DELIM, i):
            end = body.find(_DELIM, i + len(_DELIM))
            if end < 0:
                return None  # an unterminated string: no close outside strings
            i = end + len(_DELIM)
        elif body.startswith(_CLOSE, i):
            break
        else:
            i += 1
    else:
        return None
    cut = i + len(_CLOSE)
    j, closes = cut, 0
    while j < n:
        if body.startswith(_CLOSE, j):
            j, closes = j + len(_CLOSE), closes + 1
        elif body[j].isspace():
            j += 1
        else:
            return None
    return body[:cut] if closes else None


def classify(tool_text: str, tool_set: tools.ToolSet | None = None) -> Verdict:
    """Judge a pass-1 ``tool`` commit (module docstring) against ``tool_set`` (T-0140; ``None``: ``tools.load()``)."""
    body = _drop_stop(_drop_tag(tool_text))
    verdict = _judge(body, tool_set)
    if verdict.status != "valid":
        single = _single_close(body)
        if single is not None:
            retry = _judge(single, tool_set)
            if retry.status == "valid":  # T-0196: adopted only when valid
                return retry
    return verdict


def repair_verdict(text: str, tool_set: tools.ToolSet | None = None) -> Verdict:
    """Judge the repair's ``tool`` text: closed thoughts removed (an unclosed one leaves nothing), the leading tag
    dropped; ``repairable`` becomes ``refused`` (never a second repair). ``tool_set`` as in ``classify``."""
    scanned = router._strip_closed_thoughts(text)
    v = _judge(_drop_stop(_drop_tag(scanned if scanned is not None else "")), tool_set)
    if v.status == "repairable":
        return Verdict("refused", v.name, v.arguments, v.reason, v.query_problem)
    return v


# --------------------------------------------------------------------------- criterion 4: the repair prompt


def repair_prompt(pass1: Prompt, spoken_prefix: str) -> Prompt | Refusal:
    """Pass 1 continued with ``assistant(spoken_prefix + "<|tool_call>")``, or the continuation guard's refusal."""
    continuation = spoken_prefix + _OPEN
    if router.template_content(continuation) != continuation.strip():
        return Refusal("template")
    if any(lit in spoken_prefix for lit in SPECIAL_TOKENS):
        return Refusal("marker")
    return Prompt(
        messages=(
            *pass1.messages,
            Message(role="assistant", content=(TextPart(text=continuation),)),
        ),
        prefix_len=pass1.prefix_len,
        thinking=pass1.thinking,
        assists=pass1.assists,
        tools=pass1.tools,
        continue_final_message=True,
    )


# --------------------------------------------------------------------------- criterion 5: the repair request


def repair_request(
    name: str,
    draft_ids: Sequence[int],
    stop_id: int | None,
    record_tag: str,
    payloads: Payloads,
    calibration: Calibration,
) -> CdRequest:
    """The seeded ``perstep_mar`` request for ``name``. ``draft_ids`` are used verbatim (the session tokenises
    ``draft.rstrip()``); ``stop_id`` is pass 1's first stop id (1, 106 or 50) or ``None`` for a length cut.
    ``ValueError`` for another stop id, ``KeyError`` for a name without a payload."""
    if stop_id is not None and (type(stop_id) is not int or stop_id not in STOP_IDS):
        raise ValueError(f"stop_id must be one of {sorted(STOP_IDS)} or None, not {stop_id!r}")
    payload = payloads.tools.get(name)
    if payload is None:
        raise KeyError(f"no payload for tool {name!r}")
    seed = [*draft_ids, *([stop_id] if stop_id is not None else [])][:CANVAS_LENGTH]
    draft = base64.b64encode(struct.pack(f"<{len(seed)}i", *seed)).decode("ascii")
    return CdRequest(
        mode=REPAIR_MODE,
        self_conditioning="model",
        automaton=payload.automaton,
        automaton_sha256=payload.automaton_sha256,
        record_tag=record_tag,
        entropy_bound=calibration.entropy_bound,
        confidence_threshold=calibration.confidence_threshold,
        draft=draft,
        draft_noise=DRAFT_NOISE,
        draft_start_step=DRAFT_START_STEP,
    )
