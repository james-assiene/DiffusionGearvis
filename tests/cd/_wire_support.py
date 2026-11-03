"""Shared helpers for the T-0064 rows (the automaton wire artifact and the ``astra_cd_*`` contract), and the
API ruling those rows fix.

Ruling T-0064 W1 (test-author, 2026-09-29; to be recorded in docs/tasks/T-0064.md by the owner of that file).

``local_astra.contracts.cd`` (pydantic only)
  ``CdRequest(mode, self_conditioning, automaton, automaton_sha256, record_tag, entropy_bound=None,
  confidence_threshold=None, max_bucket=None)``. ``to_xargs()`` maps each field ``f`` that is set to the key
  ``"astra_cd_" + f``; an unset optional is *omitted* (never ``None``: ``LlmOptions.extra_body`` refuses it).
  ``automaton`` is at most ``1 << 20`` characters (the encoded string is the capped thing, criterion 5).
  ``CdCanvasRecord``: exactly the 14 fields of criterion 6.
``local_astra.decoding.artifact``
  ``encode_automaton(compiled) -> str`` (ASCII; no size cap here -- the cap is enforced by ``CdRequest``
  and by the plugin, so the size report can see every automaton). ``unpack_payload(payload) ->
  (meta: dict, arrays: dict[str, np.ndarray])`` and ``pack_payload(meta, arrays) -> str`` are the raw
  container codec, mutual inverses, **no validation**; the rows use them to craft corrupt payloads without
  knowing the byte format. ``meta`` carries at least ``format_version``, ``vocab_size``,
  ``tokenizer_hash``, ``n_states`` and ``n_states_bucket``.
``automaton_sha256``
  lowercase hex sha256 of the payload string's ASCII bytes (content address of exactly what travels).
Plugin ``astra_vllm_plugin.cd.xargs`` (stdlib + numpy)
  ``parse_cd_xargs(extra_args: Mapping | None) -> CdSpec | None``; ``CdRequestError(Exception)`` with a
  ``reason: str`` from ``REASONS`` below. ``CdSpec`` has ``mode``, ``self_conditioning``, ``record_tag``,
  ``entropy_bound``, ``confidence_threshold``, ``max_bucket`` (``None`` when absent) and ``automaton``,
  whose attributes ``n_states``, ``n_states_bucket`` and every name in ``ARRAY_NAMES`` hold the decoded
  numpy arrays (flat: the plugin has no ``ClassTables``).
Plugin ``astra_vllm_plugin.cd.record_schema``
  ``RECORD_FIELDS``: a literal tuple (or list/frozenset) of the ``CdCanvasRecord`` field names.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
PLUGIN_SRC = Path(os.environ.get("ASTRA_CD_PLUGIN_SRC", str(REPO / "plugin" / "astra_vllm_plugin")))
CD_DIR = PLUGIN_SRC / "astra_vllm_plugin" / "cd"

MIB = 1 << 20
VOCAB_SIZE = 262_144
TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
STOP_IDS = (1, 50, 106)
LADDER = (16, 32, 64, 128, 256, 512)

#: Every array of a compiled automaton (vendored ``CompiledAutomaton`` + ``ClassTables``, 27c7d0d).
AUTOMATON_ARRAYS = ("edge_src", "edge_dst", "edge_class", "d", "is_final", "start_vector")
TABLE_ARRAYS = (
    "class_id",
    "class_size",
    "sum_is_neg",
    "sum_indptr",
    "sum_indices",
    "max_is_neg",
    "max_indptr",
    "max_indices",
)
ARRAY_NAMES = AUTOMATON_ARRAYS + TABLE_ARRAYS

#: The ``astra_cd_*`` keys (criterion 1: ``"astra_cd_" + field``).
KEYS = {
    "mode": "astra_cd_mode",
    "self_conditioning": "astra_cd_self_conditioning",
    "automaton": "astra_cd_automaton",
    "automaton_sha256": "astra_cd_automaton_sha256",
    "record_tag": "astra_cd_record_tag",
    "entropy_bound": "astra_cd_entropy_bound",
    "confidence_threshold": "astra_cd_confidence_threshold",
    "max_bucket": "astra_cd_max_bucket",
}

#: Stable reason codes (criterion 3/4/5). T-0067 writes them into ``CdCanvasRecord.error``.
R_SHA = "sha_mismatch"
R_ENCODING = "bad_encoding"
R_VERSION = "unknown_format_version"
R_VOCAB = "vocab_size_mismatch"
R_TOKENIZER = "tokenizer_hash_mismatch"
R_BUCKET = "bucket_too_large"
R_MISSING_ARRAY = "missing_array"
R_STRUCTURE = "structure"
R_SIZE = "payload_too_large"
R_UNKNOWN_KEY = "unknown_key"
R_NO_MODE = "missing_mode"
R_MISSING_KEY = "missing_key"
R_VALUE = "bad_value"
REASONS = frozenset(
    {
        R_SHA,
        R_ENCODING,
        R_VERSION,
        R_VOCAB,
        R_TOKENIZER,
        R_BUCKET,
        R_MISSING_ARRAY,
        R_STRUCTURE,
        R_SIZE,
        R_UNKNOWN_KEY,
        R_NO_MODE,
        R_MISSING_KEY,
        R_VALUE,
    }
)


# --------------------------------------------------------------------------- the code under test


def _need(modname: str, prefix: str) -> ModuleType:
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing.startswith(prefix) and (modname == missing or modname.startswith(missing + ".")):
            pytest.fail(f"T-0064 not implemented: `{modname}` does not exist ({exc})")
        raise


def contracts_cd() -> ModuleType:
    return _need("local_astra.contracts.cd", "local_astra.contracts.cd")


def artifact() -> ModuleType:
    return _need("local_astra.decoding.artifact", "local_astra.decoding.artifact")


def plugin(name: str) -> ModuleType:
    """``astra_vllm_plugin.cd.<name>`` from ``PLUGIN_SRC`` (the container has no ``local_astra``)."""
    if str(PLUGIN_SRC) not in sys.path:
        sys.path.insert(0, str(PLUGIN_SRC))
    return _need(f"astra_vllm_plugin.cd.{name}", f"astra_vllm_plugin.cd.{name}")


def parse(xargs: Mapping[str, Any] | None) -> Any:
    return plugin("xargs").parse_cd_xargs(xargs)


def expect_error(xargs: Mapping[str, Any] | None, reason: str) -> Exception:
    """The parse must raise ``CdRequestError`` (and nothing else) with ``reason``."""
    err_type = plugin("xargs").CdRequestError
    try:
        got = parse(xargs)
    except err_type as err:
        code = getattr(err, "reason", None)
        assert code == reason, f"CdRequestError.reason is {code!r}, expected {reason!r} ({err})"
        return err
    raise AssertionError(
        f"expected CdRequestError({reason!r}); parse accepted it and returned {type(got).__name__}"
    )


# --------------------------------------------------------------------------- payloads and requests


def sha(payload: str) -> str:
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def encode(compiled: Any) -> str:
    payload = artifact().encode_automaton(compiled)
    assert isinstance(payload, str), f"encode_automaton returned {type(payload).__name__}, not str"
    payload.encode("ascii")  # the wire value is an ASCII JSON string
    return payload


def raw_xargs(
    payload: str,
    *,
    digest: str | None = None,
    mode: str = "perstep_mar",
    self_conditioning: str = "model",
    record_tag: str = "t0064-row",
    **extra: Any,
) -> dict[str, Any]:
    """A hand-built ``astra_cd_*`` dict (bypasses ``CdRequest`` so invalid ones can be crafted)."""
    out: dict[str, Any] = {
        KEYS["mode"]: mode,
        KEYS["self_conditioning"]: self_conditioning,
        KEYS["automaton"]: payload,
        KEYS["automaton_sha256"]: sha(payload) if digest is None else digest,
        KEYS["record_tag"]: record_tag,
    }
    for k, v in extra.items():
        out[KEYS[k]] = v
    return out


def request_xargs(payload: str, **fields: Any) -> dict[str, Any]:
    """``CdRequest(...).to_xargs()`` pushed through ``json.dumps``/``loads`` (criterion 2's path)."""
    fields.setdefault("mode", "perstep_mar")
    fields.setdefault("self_conditioning", "model")
    fields.setdefault("record_tag", "t0064-rt")
    req = contracts_cd().CdRequest(automaton=payload, automaton_sha256=sha(payload), **fields)
    out: dict[str, Any] = json.loads(json.dumps(req.to_xargs()))
    return out


# --------------------------------------------------------------------------- automata


def compiled_arrays(compiled: Any) -> dict[str, Any]:
    out = {n: getattr(compiled, n) for n in AUTOMATON_ARRAYS}
    out.update({n: getattr(compiled.tables, n) for n in TABLE_ARRAYS})
    return out


def array_mismatches(compiled: Any, decoded: Any) -> list[str]:
    """Names whose decoded array differs from the compiled one in type, dtype, shape or any value."""
    import numpy as np

    bad = []
    for name, want in compiled_arrays(compiled).items():
        if not hasattr(decoded, name):
            bad.append(f"{name}: absent on {type(decoded).__name__}")
            continue
        got = getattr(decoded, name)
        want = np.asarray(want)
        if not isinstance(got, np.ndarray):
            bad.append(f"{name}: {type(got).__name__}, not ndarray")
        elif got.dtype != want.dtype:
            bad.append(f"{name}: dtype {got.dtype} != {want.dtype}")
        elif got.shape != want.shape:
            bad.append(f"{name}: shape {got.shape} != {want.shape}")
        elif not np.array_equal(got, want):
            bad.append(f"{name}: values differ")
    for attr in ("n_states", "n_states_bucket"):
        if getattr(decoded, attr, None) != getattr(compiled, attr):
            bad.append(f"{attr}: {getattr(decoded, attr, None)!r} != {getattr(compiled, attr)!r}")
    return bad


def _vendored_automaton() -> ModuleType:
    return importlib.import_module("local_astra.decoding._vendor.dgfa_compile.automaton")


def hand_built(
    n_states: int,
    transitions: list[tuple[int, int, int]],
    finals: set[int],
    *,
    name: str,
    tokenizer_hash: str = TOKENIZER_SHA256,
) -> Any:
    """Compile a hand-written (possibly nondeterministic) automaton through the vendored back half, with
    the pinned vocabulary size and stop set, so it is shaped exactly like T-0063's artifacts."""
    minimize = importlib.import_module("local_astra.decoding._vendor.dgfa_compile.minimize")
    dfa = minimize.Dfa(
        n_states=n_states,
        transitions=tuple(transitions),
        start=0,
        finals=frozenset(finals),
    )
    return _vendored_automaton().compile_automaton(
        dfa,
        name=name,
        end_tokens=STOP_IDS,
        vocab_size=VOCAB_SIZE,
        schema_hash=f"hand:{name}",
        tokenizer_hash=tokenizer_hash,
        compiler_version="t0064-hand-built",
    )


def nfa_overlapping() -> Any:
    """Small NFA: state 0 sends {6, 7} to both 1 and 2 (overlapping, not identical label sets)."""
    t = [(0, v, 1) for v in (5, 6, 7)]
    t += [(0, v, 2) for v in (6, 7, 8)]
    t += [(1, 5, 1), (1, 9, 3), (2, 9, 3), (2, 10, 3), (2, 9, 2)]
    a = hand_built(4, t, {3}, name="nfa_overlapping")
    assert not a.is_dfa, "fixture precondition: the hand-built automaton must be an NFA"
    return a


def nfa_wide_overlap() -> Any:
    """NFA with a label set wider than V/2 (stored as a complement) overlapping a positive one."""
    t = [(0, v, 1) for v in range(1000, VOCAB_SIZE)]
    t += [(0, v, 2) for v in range(1500, 1600)]
    t += [(1, 300, 3), (2, 300, 3), (2, 301, 1)]
    a = hand_built(4, t, {3}, name="nfa_wide_overlap")
    assert not a.is_dfa, "fixture precondition: the hand-built automaton must be an NFA"
    assert bool(a.tables.sum_is_neg.any()), "fixture precondition: a complement-stored class"
    return a


def chain(n_states: int) -> Any:
    """A DFA chain 0 -> 1 -> ... -> n-1 on token 1000 + i (for bucket-boundary rows)."""
    t = [(i, 1000 + i, i + 1) for i in range(n_states - 1)]
    return hand_built(n_states, t, {n_states - 1}, name=f"chain{n_states}")


def with_bucket(bucket: int) -> Any:
    """A hand-built chain whose vendored ``n_states_bucket`` is exactly ``bucket``."""
    # augmentation adds ACC; the bucket holds the states plus the dead state (vendored bucket_size).
    for k in (bucket - 2, bucket - 1):
        a = chain(k)
        if a.n_states_bucket == bucket:
            return a
    raise AssertionError(f"no chain length gives bucket {bucket}")


def replace_meta(compiled: Any, **changes: Any) -> Any:
    return dataclasses.replace(compiled, **changes)
