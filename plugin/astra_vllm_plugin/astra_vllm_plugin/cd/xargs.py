"""Parse the ``astra_cd_*`` keys of ``SamplingParams.extra_args`` (T-0064), stdlib + numpy only.

The request side is ``local_astra.contracts.cd.CdRequest.to_xargs()``; the keys are ``"astra_cd_" +
field``. ``parse_cd_xargs`` returns ``None`` when no ``astra_cd_*`` key is present (the stock,
unconstrained path), a ``CdSpec`` for a valid request, and raises ``CdRequestError`` (with a stable
``reason``) for anything else. A misspelt ``astra_cd_*`` key is refused, never ignored: silently decoding a
constrained request as unconstrained is the failure this module exists to prevent. Keys outside the
prefix (``astra_tap``, ...) are not ours and are ignored.

Check order (ruling W1): unknown keys, then the missing mode, then missing keys and bad values (the T-0081
draft keys among them: ``draft_noise`` / ``draft_start_step`` without ``draft`` is a missing key, then the
context-free draft checks), then the 1 MiB cap (before any decoding), then the sha (before any cache lookup), then the decode, then the
per-request bucket limit (re-checked on every request, cache hit or not).

Runs in ``add_request`` on the engine thread; T-0067 turns ``CdRequestError`` into the non-fatal loud path.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import math
import re
import threading
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np

from .automaton import (
    BUCKET_LADDER,
    MAX_PAYLOAD_CHARS,
    R_BUCKET,
    R_DRAFT_ENCODING,
    R_DRAFT_LENGTH,
    R_DRAFT_NOISE,
    R_DRAFT_START,
    R_DRAFT_VOCAB,
    R_MISSING_KEY,
    R_NO_MODE,
    R_SHA,
    R_SIZE,
    R_UNKNOWN_KEY,
    R_VALUE,
    REASONS,
    VOCAB_SIZE,
    CdRequestError,
    WireAutomaton,
    decode_payload,
)

__all__ = [
    "CD_PREFIX",
    "KEYS",
    "REASONS",
    "CdRequestError",
    "CdSpec",
    "WireAutomaton",
    "clear_cache",
    "parse_cd_xargs",
]

CD_PREFIX = "astra_cd_"
MODES = ("end_projection", "perstep_mf", "perstep_mar", "perstep_mar_qargmax")
SELF_CONDITIONING = ("model", "constrained")
REQUIRED_FIELDS = ("mode", "automaton", "automaton_sha256", "record_tag")
OPTIONAL_FIELDS = (
    "self_conditioning",
    "entropy_bound",
    "confidence_threshold",
    "max_bucket",
    "draft",
    "draft_noise",
    "draft_start_step",
)
KEYS = {f: CD_PREFIX + f for f in REQUIRED_FIELDS + OPTIONAL_FIELDS}
_KNOWN_KEYS = frozenset(KEYS.values())
_RECORD_TAG = re.compile(r"[A-Za-z0-9_.:-]{1,64}")

#: Decoded automata by sha256 (content address of the payload string). Bounded LRU.
CACHE_SIZE = 64
_cache: OrderedDict[str, WireAutomaton] = OrderedDict()
_cache_lock = threading.Lock()


@dataclasses.dataclass(frozen=True, eq=False)
class CdSpec:
    """A validated constrained request."""

    mode: str
    self_conditioning: str
    record_tag: str
    entropy_bound: float | None
    confidence_threshold: float | None
    max_bucket: int | None
    automaton_sha256: str
    automaton: WireAutomaton
    #: T-0081: the draft ids (read-only int64, 1..n; ``None``: not seeded), the noised fraction and the start step.
    #: Bounds that need CL or ``max_denoising_steps`` are admission's (``cd/seed.py``).
    draft: np.ndarray | None = None
    draft_noise: float = 0.0
    draft_start_step: int = 0


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _positive(x: Mapping[str, Any], field: str) -> float | None:
    key = KEYS[field]
    if key not in x:
        return None
    v = x[key]
    if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
        raise CdRequestError(R_VALUE, f"{key}={v!r} must be a finite number > 0")
    return float(v)


def _choice(x: Mapping[str, Any], field: str, options: tuple[str, ...]) -> str:
    v = x[KEYS[field]]
    if type(v) is not str or v not in options:
        raise CdRequestError(R_VALUE, f"{KEYS[field]}={v!r} is not one of {options}")
    return v


def _draft(x: Mapping[str, Any]) -> tuple[np.ndarray | None, float, int]:
    """The context-free checks of the draft keys (T-0081 ruling SD3); the caller checked noise/start need a draft."""
    if KEYS["draft"] not in x:
        return None, 0.0, 0
    raw_s = x[KEYS["draft"]]
    if type(raw_s) is not str:
        raise CdRequestError(R_DRAFT_ENCODING, f"draft is {type(raw_s).__name__}, not str")
    try:
        raw = base64.b64decode(raw_s.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError):
        raise CdRequestError(R_DRAFT_ENCODING, "draft is not standard base64") from None
    if base64.b64encode(raw).decode("ascii") != raw_s:
        raise CdRequestError(R_DRAFT_ENCODING, "draft is not canonical standard base64")
    if len(raw) % 4:
        raise CdRequestError(
            R_DRAFT_ENCODING, f"draft is {len(raw)} bytes, not a whole number of int32 ids"
        )
    if not raw:
        raise CdRequestError(R_DRAFT_LENGTH, "draft holds no token id")
    ids = np.frombuffer(raw, dtype="<i4").astype(np.int64)
    if bool((ids < 0).any()) or bool((ids >= VOCAB_SIZE).any()):
        raise CdRequestError(R_DRAFT_VOCAB, f"draft holds an id outside [0, {VOCAB_SIZE})")
    ids.setflags(write=False)
    noise: Any = x.get(KEYS["draft_noise"], 0.0)
    if type(noise) not in (int, float) or not math.isfinite(noise) or not 0.0 <= noise < 1.0:
        raise CdRequestError(R_DRAFT_NOISE, f"draft_noise={noise!r} must be a number in [0, 1)")
    start: Any = x.get(KEYS["draft_start_step"], 0)
    if type(start) is not int or start < 0:
        raise CdRequestError(R_DRAFT_START, f"draft_start_step={start!r} must be an int >= 0")
    return ids, float(noise), start


def _decoded(payload: str, digest: str) -> WireAutomaton:
    with _cache_lock:
        hit = _cache.get(digest)
        if hit is not None:
            _cache.move_to_end(digest)
            return hit
    automaton = decode_payload(payload)
    with _cache_lock:
        _cache[digest] = automaton
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return automaton


def parse_cd_xargs(extra_args: Mapping[str, Any] | None) -> CdSpec | None:
    """``None`` (unconstrained), a ``CdSpec``, or ``CdRequestError``; see the module docstring."""
    if not extra_args:
        return None
    x = {k: v for k, v in extra_args.items() if isinstance(k, str) and k.startswith(CD_PREFIX)}
    if not x:
        return None
    unknown = sorted(set(x) - _KNOWN_KEYS)
    if unknown:
        raise CdRequestError(
            R_UNKNOWN_KEY, f"unknown astra_cd_* key(s) {unknown}; known: {sorted(_KNOWN_KEYS)}"
        )
    if KEYS["mode"] not in x:
        raise CdRequestError(R_NO_MODE, f"astra_cd_* keys {sorted(x)} without {KEYS['mode']}")
    missing = [KEYS[f] for f in REQUIRED_FIELDS if KEYS[f] not in x]
    if missing:
        raise CdRequestError(R_MISSING_KEY, f"missing {missing}")
    if KEYS["draft"] not in x and (KEYS["draft_noise"] in x or KEYS["draft_start_step"] in x):
        raise CdRequestError(
            R_MISSING_KEY, f"draft_noise / draft_start_step without {KEYS['draft']}"
        )

    mode = _choice(x, "mode", MODES)
    self_conditioning = (
        _choice(x, "self_conditioning", SELF_CONDITIONING)
        if KEYS["self_conditioning"] in x
        else "model"
    )
    if self_conditioning == "constrained" and not mode.startswith("perstep_"):
        raise CdRequestError(R_VALUE, "self_conditioning='constrained' needs a perstep_* mode")
    tag = x[KEYS["record_tag"]]
    if type(tag) is not str or _RECORD_TAG.fullmatch(tag) is None:
        raise CdRequestError(
            R_VALUE, f"record_tag {tag!r} must be 1-64 characters of [A-Za-z0-9_.:-]"
        )
    entropy_bound = _positive(x, "entropy_bound")
    confidence_threshold = _positive(x, "confidence_threshold")
    max_bucket = x.get(KEYS["max_bucket"])
    if max_bucket is not None and (type(max_bucket) is not int or max_bucket not in BUCKET_LADDER):
        raise CdRequestError(R_VALUE, f"max_bucket {max_bucket!r} is not one of {BUCKET_LADDER}")
    draft, draft_noise, draft_start_step = _draft(x)
    payload = x[KEYS["automaton"]]
    digest = x[KEYS["automaton_sha256"]]
    if type(payload) is not str:
        raise CdRequestError(R_VALUE, f"automaton is {type(payload).__name__}, not str")
    if type(digest) is not str:
        raise CdRequestError(R_VALUE, f"automaton_sha256 is {type(digest).__name__}, not str")

    if len(payload) > MAX_PAYLOAD_CHARS:
        raise CdRequestError(R_SIZE, f"automaton is {len(payload)} chars > {MAX_PAYLOAD_CHARS}")
    try:
        actual = hashlib.sha256(payload.encode("ascii")).hexdigest()
    except UnicodeEncodeError:
        raise CdRequestError(R_SHA, "automaton is not ASCII, so it cannot match its sha") from None
    if actual != digest:
        raise CdRequestError(
            R_SHA, f"sha256 of the automaton is {actual}, the request says {digest}"
        )

    automaton = _decoded(payload, digest)
    limit = BUCKET_LADDER[-1] if max_bucket is None else max_bucket
    if automaton.n_states_bucket > limit:
        raise CdRequestError(R_BUCKET, f"bucket {automaton.n_states_bucket} > max_bucket {limit}")
    return CdSpec(
        mode=mode,
        self_conditioning=self_conditioning,
        record_tag=tag,
        entropy_bound=entropy_bound,
        confidence_threshold=confidence_threshold,
        max_bucket=max_bucket,
        automaton_sha256=digest,
        automaton=automaton,
        draft=draft,
        draft_noise=draft_noise,
        draft_start_step=draft_start_step,
    )
