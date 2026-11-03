"""Decode and validate the automaton wire artifact (T-0064), stdlib + numpy only.

The encoder is ``local_astra/decoding/artifact.py`` (format version 1): standard base64 of a ZIP holding
``meta.json`` and one ``<name>.npy`` per array. The container has no ``local_astra``, so the format is
restated here; change both sides together and bump ``FORMAT_VERSION``.

``decode_payload`` either returns a fully validated ``WireAutomaton`` or raises ``CdRequestError`` with a
stable ``reason`` code; it never returns a partial object and never raises anything else for a bad
payload. The structural invariants are proved on the way in because a wrong automaton is internally
consistent downstream: a duplicated ``(src, dst)`` pair or an ``is_dfa`` flag that over-claims changes the
sampling *distribution*, which no acceptance check can see (vendored ``automaton.load``).

The per-request bucket limit (``astra_cd_max_bucket``) is checked by ``xargs.parse_cd_xargs``, not here:
a decoded automaton is cached by sha and reused across requests with different limits.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import io
import itertools
import json
import math
import zipfile
import zlib
from collections import defaultdict
from typing import Any

import numpy as np

FORMAT_VERSION = 1
META_ENTRY = "meta.json"
#: Gemma 4 vocabulary size.
VOCAB_SIZE = 262_144
#: sha256 of the pinned ``tokenizer.json`` (``local_astra.decoding.vocab.TOKENIZER_SHA256``).
TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
#: Cap on the encoded string (T-0062 decision 4, provisional).
MAX_PAYLOAD_CHARS = 1 << 20
#: Cap on the decompressed archive, so a crafted payload cannot inflate without bound.
MAX_UNCOMPRESSED_BYTES = 256 << 20
#: Buckets the plugin supports (T-0062 decision 5).
BUCKET_LADDER = (16, 32, 64, 128, 256, 512)
MAX_BUCKET = BUCKET_LADDER[-1]
#: ``d(s)`` where no accepting state is reachable (vendored ``INF_DISTANCE``).
INF_DISTANCE = 1 << 24

_I32 = np.dtype(np.int32)
_BOOL = np.dtype(np.bool_)
ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "edge_src": _I32,
    "edge_dst": _I32,
    "edge_class": _I32,
    "d": _I32,
    "is_final": _BOOL,
    "start_vector": _BOOL,
    "class_id": _I32,
    "class_size": _I32,
    "sum_is_neg": _BOOL,
    "sum_indptr": _I32,
    "sum_indices": _I32,
    "max_is_neg": _BOOL,
    "max_indptr": _I32,
    "max_indices": _I32,
}
ARRAY_NAMES = tuple(ARRAY_DTYPES)

# Stable reason codes (T-0064 ruling W1). T-0067 writes them into the record's ``error``.
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
# T-0081 (ruling SD3): the draft that seeds a constrained request's first canvas.
R_DRAFT_ENCODING = "draft_bad_encoding"
R_DRAFT_LENGTH = "draft_bad_length"
R_DRAFT_VOCAB = "draft_id_out_of_vocab"
R_DRAFT_NOISE = "draft_noise_out_of_range"
R_DRAFT_START = "draft_start_step_out_of_range"
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
        R_DRAFT_ENCODING,
        R_DRAFT_LENGTH,
        R_DRAFT_VOCAB,
        R_DRAFT_NOISE,
        R_DRAFT_START,
    }
)

# Everything the container codec can raise on hostile bytes (base64, zip, zlib, npy header, JSON).
_CODEC_ERRORS = (
    binascii.Error,
    zipfile.BadZipFile,
    zlib.error,
    ValueError,
    KeyError,
    OSError,
    EOFError,
    UnicodeError,
    NotImplementedError,
    RuntimeError,
    MemoryError,  # a hostile size must never escape into the engine thread (ruling W8)
)


class CdRequestError(Exception):
    """A constrained request that must not be decoded. ``reason`` is one of ``REASONS``."""

    def __init__(self, reason: str, detail: str = "") -> None:
        assert reason in REASONS, reason
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclasses.dataclass(frozen=True, eq=False)
class WireAutomaton:
    """A decoded, validated automaton. Arrays are flat, read-only numpy arrays (shapes as vendored
    ``CompiledAutomaton`` / ``ClassTables``); states ``>= n_states`` are the dead padding."""

    name: str
    n_states: int
    n_states_bucket: int
    n_edges: int
    n_classes: int
    vocab_size: int
    is_dfa: bool
    tokenizer_hash: str
    schema_hash: str
    compiler_version: str

    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_class: np.ndarray
    d: np.ndarray
    is_final: np.ndarray
    start_vector: np.ndarray
    class_id: np.ndarray
    class_size: np.ndarray
    sum_is_neg: np.ndarray
    sum_indptr: np.ndarray
    sum_indices: np.ndarray
    max_is_neg: np.ndarray
    max_indptr: np.ndarray
    max_indices: np.ndarray


def _fail(reason: str, detail: str = "") -> CdRequestError:
    return CdRequestError(reason, detail)


def _read_npy(name: str, data: bytes) -> np.ndarray:
    """One ``.npy`` entry. ``read_array`` allocates the header's declared shape before reading, so the
    declared byte count is checked against the bytes actually present first."""
    buf = io.BytesIO(data)
    version = np.lib.format.read_magic(buf)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(buf)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(buf)
    else:
        raise _fail(R_ENCODING, f"{name}: unsupported npy version {version}")
    declared = math.prod(int(n) for n in shape) * dtype.itemsize
    if declared > len(data) - buf.tell():
        raise _fail(
            R_ENCODING, f"{name} declares {declared} bytes but carries {len(data) - buf.tell()}"
        )
    buf.seek(0)
    return np.lib.format.read_array(buf, allow_pickle=False)


def _entry_bytes(raw: bytes, info: zipfile.ZipInfo) -> bytes:
    """One entry's bytes, which must end exactly at the declared size (ruling W9).

    ``ZipFile.open`` stops at the declared size and checks the CRC of what it read, so an entry whose
    CRC matches the declared prefix but whose deflate stream carries more hides the rest. So the
    compressed bytes are taken from the local header and inflated here, never past ``file_size + 1``,
    and the stream must end there with nothing left over.
    """
    off = info.header_offset
    if raw[off : off + 4] != b"PK\x03\x04":
        raise _fail(R_ENCODING, f"{info.filename}: no local file header")
    name_len = int.from_bytes(raw[off + 26 : off + 28], "little")
    extra_len = int.from_bytes(raw[off + 28 : off + 30], "little")
    start = off + 30 + name_len + extra_len
    comp = raw[start : start + info.compress_size]
    if len(comp) != info.compress_size:
        raise _fail(R_ENCODING, f"{info.filename}: compressed data is truncated")
    if info.compress_type == zipfile.ZIP_STORED:
        data = comp
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        inflater = zlib.decompressobj(-zlib.MAX_WBITS)
        data = inflater.decompress(comp, info.file_size + 1)
        if not inflater.eof or inflater.unconsumed_tail or inflater.unused_data:
            raise _fail(
                R_ENCODING, f"{info.filename}: the stream does not end at its declared size"
            )
    else:
        raise _fail(R_ENCODING, f"{info.filename}: unsupported compression {info.compress_type}")
    if len(data) != info.file_size:
        raise _fail(R_ENCODING, f"{info.filename}: {len(data)} bytes, declared {info.file_size}")
    if zlib.crc32(data) != info.CRC:
        raise _fail(R_ENCODING, f"{info.filename}: CRC mismatch")
    return data


def _unpack(payload: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The container, or ``bad_encoding``. No semantic checks."""
    try:
        raw = base64.b64decode(payload.encode("ascii"), validate=True)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            infos = zf.infolist()
            names = [i.filename for i in infos]
            if len(set(names)) != len(names):
                raise _fail(R_ENCODING, "duplicate archive entries")
            if sum(i.file_size for i in infos) > MAX_UNCOMPRESSED_BYTES:
                raise _fail(R_ENCODING, "archive inflates past the decompression cap")
            if META_ENTRY not in names:
                raise _fail(R_ENCODING, f"no {META_ENTRY} in the archive")
            meta: Any = None
            arrays: dict[str, np.ndarray] = {}
            for info in infos:
                if info.filename != META_ENTRY and not info.filename.endswith(".npy"):
                    raise _fail(R_ENCODING, f"unexpected archive entry {info.filename!r}")
                data = _entry_bytes(raw, info)
                if info.filename == META_ENTRY:
                    meta = json.loads(data.decode("utf-8"))
                else:
                    arrays[info.filename[: -len(".npy")]] = _read_npy(info.filename, data)
    except CdRequestError:
        raise
    except _CODEC_ERRORS as exc:
        raise _fail(R_ENCODING, f"{type(exc).__name__}: {exc}") from None
    if not isinstance(meta, dict):
        raise _fail(R_ENCODING, "meta is not a JSON object")
    return meta, arrays


def _meta_int(meta: dict[str, Any], key: str) -> int:
    v = meta.get(key)
    if type(v) is not int:
        raise _fail(R_STRUCTURE, f"meta {key!r} is {v!r}, not an int")
    return v


def _meta_str(meta: dict[str, Any], key: str) -> str:
    v = meta.get(key, "")
    if not isinstance(v, str):
        raise _fail(R_STRUCTURE, f"meta {key!r} is not a string")
    return v


def _row_sorted_unique(indptr: np.ndarray, indices: np.ndarray) -> bool:
    """Every CSR row strictly increasing (indptr already known monotone and in range)."""
    if indices.size < 2:
        return True
    bad = np.diff(indices.astype(np.int64)) <= 0
    starts = indptr[1:-1].astype(np.int64)
    starts = starts[(starts > 0) & (starts < indices.size)]
    bad[starts - 1] = False
    return not bool(bad.any())


def _check_csr(
    prefix: str, a: dict[str, np.ndarray], n_classes: int, class_size: np.ndarray
) -> None:
    is_neg, indptr, indices = a[f"{prefix}_is_neg"], a[f"{prefix}_indptr"], a[f"{prefix}_indices"]
    if is_neg.shape != (n_classes,) or indptr.shape != (n_classes + 1,):
        raise _fail(R_STRUCTURE, f"{prefix} table shapes disagree with {n_classes} classes")
    if int(indptr[0]) != 0 or bool((np.diff(indptr) < 0).any()) or int(indptr[-1]) != indices.size:
        raise _fail(R_STRUCTURE, f"{prefix}_indptr is not a CSR pointer over {prefix}_indices")
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= VOCAB_SIZE):
        raise _fail(R_STRUCTURE, f"{prefix}_indices holds a token id outside the vocabulary")
    if not _row_sorted_unique(indptr, indices):
        raise _fail(R_STRUCTURE, f"a {prefix} row is not strictly increasing")
    nnz = np.diff(indptr).astype(np.int64)
    true_size = np.where(is_neg, VOCAB_SIZE - nnz, nnz)
    if not np.array_equal(true_size, class_size.astype(np.int64)):
        raise _fail(R_STRUCTURE, f"class_size disagrees with the {prefix} table")


def _check_tables_agree(a: dict[str, np.ndarray], n_classes: int) -> None:
    """The SUM and MAX tables describe the same member set per class (their polarities may differ)."""
    s_neg, s_ptr, s_idx = a["sum_is_neg"], a["sum_indptr"], a["sum_indices"]
    m_neg, m_ptr, m_idx = a["max_is_neg"], a["max_indptr"], a["max_indices"]
    for c in range(n_classes):
        s = s_idx[int(s_ptr[c]) : int(s_ptr[c + 1])]
        m = m_idx[int(m_ptr[c]) : int(m_ptr[c + 1])]
        if bool(s_neg[c]) == bool(m_neg[c]):
            same = np.array_equal(s, m)
        else:  # one stores S_c, the other N_c = V \ S_c: disjoint (sizes already agree)
            same = not bool(np.isin(s, m, assume_unique=True).any())
        if not same:
            raise _fail(R_STRUCTURE, f"class {c}: the SUM and MAX tables hold different sets")


def _distance_to_final(n: int, src: np.ndarray, dst: np.ndarray, finals: np.ndarray) -> np.ndarray:
    """Reverse BFS, as the vendored ``distance_to_final``."""
    rev: list[list[int]] = [[] for _ in range(n)]
    for s, t in zip(src.tolist(), dst.tolist()):
        rev[t].append(s)
    d = np.full(n, INF_DISTANCE, dtype=np.int64)
    frontier = [int(f) for f in np.flatnonzero(finals)]
    d[frontier] = 0
    depth = 0
    while frontier:
        depth += 1
        nxt = []
        for q in frontier:
            for p in rev[q]:
                if d[p] == INF_DISTANCE:
                    d[p] = depth
                    nxt.append(p)
        frontier = nxt
    return d


def _check_dfa_claim(a: dict[str, np.ndarray]) -> None:
    """``is_dfa=True`` must hold on tokens: out of each state, label sets pairwise disjoint (vendored
    ``_assert_outgoing_labels_are_disjoint``; stored sides only, never over V)."""
    ptr, idx, neg = a["sum_indptr"], a["sum_indices"], a["sum_is_neg"]
    by_src: dict[int, list[int]] = defaultdict(list)
    for e, s in enumerate(a["edge_src"].tolist()):
        by_src[s].append(e)
    edge_class = a["edge_class"]

    def stored(e: int) -> tuple[np.ndarray, bool]:
        c = int(edge_class[e])
        return idx[int(ptr[c]) : int(ptr[c + 1])], bool(neg[c])

    for edges in by_src.values():
        if len(edges) < 2:
            continue
        pos: list[np.ndarray] = []
        negs: list[np.ndarray] = []
        for e in edges:
            ids, is_neg = stored(e)
            (negs if is_neg else pos).append(ids)
        if len(pos) > 1:
            stacked = np.concatenate(pos)
            if np.unique(stacked).size != stacked.size:
                raise _fail(R_STRUCTURE, "is_dfa=True but a state's positive labels overlap")
        for p in pos:
            for n_ids in negs:
                if bool((~np.isin(p, n_ids, assume_unique=True)).any()):
                    raise _fail(R_STRUCTURE, "is_dfa=True but a state's labels overlap")
        for n1, n2 in itertools.combinations(negs, 2):
            if VOCAB_SIZE - int(np.union1d(n1, n2).size) > 0:
                raise _fail(R_STRUCTURE, "is_dfa=True but a state's complement labels overlap")


def _validate(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> WireAutomaton:
    version = meta.get("format_version")
    if type(version) is not int or version != FORMAT_VERSION:
        raise _fail(R_VERSION, f"format_version {version!r}, this plugin reads {FORMAT_VERSION}")
    vocab = meta.get("vocab_size")
    if type(vocab) is not int or vocab != VOCAB_SIZE:
        raise _fail(R_VOCAB, f"vocab_size {vocab!r} != {VOCAB_SIZE}")
    tok = meta.get("tokenizer_hash")
    if tok != TOKENIZER_SHA256:
        raise _fail(R_TOKENIZER, f"tokenizer_hash {tok!r} is not the pinned {TOKENIZER_SHA256}")
    bucket = _meta_int(meta, "n_states_bucket")
    if bucket > MAX_BUCKET:
        raise _fail(
            R_BUCKET, f"n_states_bucket {bucket} > {MAX_BUCKET}, the largest the plugin supports"
        )
    if bucket not in BUCKET_LADDER:
        raise _fail(R_STRUCTURE, f"n_states_bucket {bucket} is not on the ladder {BUCKET_LADDER}")
    missing = [n for n in ARRAY_NAMES if n not in arrays]
    if missing:
        raise _fail(R_MISSING_ARRAY, f"payload lacks {missing}")
    extra = sorted(set(arrays) - set(ARRAY_NAMES))
    if extra:
        raise _fail(R_STRUCTURE, f"unknown arrays {extra}")
    for name, want in ARRAY_DTYPES.items():
        arr = arrays[name]
        if arr.dtype != want or arr.ndim != 1:
            raise _fail(R_STRUCTURE, f"{name} is {arr.dtype}[{arr.shape}], expected 1-d {want}")

    n_states = _meta_int(meta, "n_states")
    n_edges = _meta_int(meta, "n_edges")
    n_classes = _meta_int(meta, "n_classes")
    is_dfa = meta.get("is_dfa")
    if type(is_dfa) is not bool:
        raise _fail(R_STRUCTURE, "meta 'is_dfa' is not a bool")
    a = arrays
    if not 1 <= n_states < bucket:
        raise _fail(R_STRUCTURE, f"n_states {n_states} leaves no dead state in bucket {bucket}")
    for name in ("d", "is_final", "start_vector"):
        if a[name].shape != (bucket,):
            raise _fail(R_STRUCTURE, f"{name} has shape {a[name].shape}, bucket is {bucket}")
    for name in ("edge_src", "edge_dst", "edge_class", "class_id"):
        if a[name].shape != (n_edges,):
            raise _fail(R_STRUCTURE, f"{name} has shape {a[name].shape}, n_edges is {n_edges}")
    if a["class_size"].shape != (n_classes,):
        raise _fail(
            R_STRUCTURE, f"class_size has shape {a['class_size'].shape}, {n_classes} classes"
        )
    src, dst, cls = a["edge_src"], a["edge_dst"], a["edge_class"]
    if n_edges:
        if int(min(src.min(), dst.min())) < 0 or int(max(src.max(), dst.max())) >= n_states:
            raise _fail(R_STRUCTURE, f"an edge endpoint lies outside the {n_states} real states")
        if int(cls.min()) < 0 or int(cls.max()) >= n_classes:
            raise _fail(R_STRUCTURE, f"an edge class lies outside the {n_classes} classes")
    if not np.array_equal(cls, a["class_id"]):
        raise _fail(R_STRUCTURE, "edge_class and class_id disagree")
    # The compiler interns only labels some edge carries, so n_classes <= n_edges. Checked before any
    # per-class loop here and before the core builds [L, C] tensors (ruling W8).
    if np.unique(cls).size != n_classes:
        raise _fail(R_STRUCTURE, f"{n_classes} classes but edges use {np.unique(cls).size}")
    pairs = src.astype(np.int64) * bucket + dst.astype(np.int64)
    if np.unique(pairs).size != n_edges:
        raise _fail(
            R_STRUCTURE, "duplicated (src, dst) pair: one edge per pair carries the label union"
        )
    if bool(a["is_final"][n_states:].any()) or bool(a["start_vector"][n_states:].any()):
        raise _fail(R_STRUCTURE, "a padding state is final or a start state")
    if not bool(a["start_vector"].any()) or not bool(a["is_final"].any()):
        raise _fail(R_STRUCTURE, "no start state or no accepting state")
    if bool((a["class_size"] < 0).any()) or bool((a["class_size"] > VOCAB_SIZE).any()):
        raise _fail(R_STRUCTURE, "class_size outside [0, V]")
    _check_csr("sum", a, n_classes, a["class_size"])
    _check_csr("max", a, n_classes, a["class_size"])
    _check_tables_agree(a, n_classes)
    want_d = np.full(bucket, INF_DISTANCE, dtype=np.int64)
    want_d[:n_states] = _distance_to_final(n_states, src, dst, a["is_final"][:n_states])
    if not np.array_equal(a["d"].astype(np.int64), want_d):
        raise _fail(R_STRUCTURE, "d is not the distance to an accepting state")
    if is_dfa:
        _check_dfa_claim(a)

    for arr in arrays.values():
        arr.setflags(write=False)
    return WireAutomaton(
        name=_meta_str(meta, "name"),
        n_states=n_states,
        n_states_bucket=bucket,
        n_edges=n_edges,
        n_classes=n_classes,
        vocab_size=VOCAB_SIZE,
        is_dfa=is_dfa,
        tokenizer_hash=TOKENIZER_SHA256,
        schema_hash=_meta_str(meta, "schema_hash"),
        compiler_version=_meta_str(meta, "compiler_version"),
        **{name: arrays[name] for name in ARRAY_NAMES},
    )


def decode_payload(payload: str) -> WireAutomaton:
    """The validated automaton of a wire string, or ``CdRequestError``. The caller checks the size cap
    and the sha first (``xargs.parse_cd_xargs``); this re-checks the cap so it is safe on its own."""
    if not isinstance(payload, str):
        raise _fail(R_VALUE, f"automaton is {type(payload).__name__}, not str")
    if len(payload) > MAX_PAYLOAD_CHARS:
        raise _fail(R_SIZE, f"{len(payload)} chars > {MAX_PAYLOAD_CHARS}")
    meta, arrays = _unpack(payload)
    return _validate(meta, arrays)
