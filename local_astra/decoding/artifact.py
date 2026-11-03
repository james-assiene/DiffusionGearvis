"""The automaton wire artifact (T-0064): a compiled automaton as one ASCII string for ``vllm_xargs``.

Container, format version 1 (mirrored by the numpy-only decoder in ``plugin/.../cd/automaton.py``;
change both together and bump ``FORMAT_VERSION``):

- standard base64 (``+/``, padded) of a ZIP archive, deflate level 9, fixed timestamps and sorted entry
  order, so equal automata encode to equal strings (the sha256 of the string is the content address);
- ``meta.json``: a JSON object with at least ``format_version``, ``vocab_size``, ``tokenizer_hash``,
  ``n_states``, ``n_states_bucket``, ``n_edges``, ``n_classes`` and ``is_dfa``;
- ``<name>.npy`` for each of the 14 arrays: the six of ``CompiledAutomaton`` and the eight of
  ``ClassTables`` (``allow_pickle=False``).

``encode_automaton`` applies no size cap: ``CdRequest`` and the plugin enforce the 1 MiB cap, so the size
report can see every automaton. ``pack_payload``/``unpack_payload`` are the raw codec, mutual inverses,
with no validation.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from local_astra.decoding._vendor.dgfa_compile.automaton import CompiledAutomaton

__all__ = [
    "ARRAY_NAMES",
    "AUTOMATON_ARRAYS",
    "FORMAT_VERSION",
    "TABLE_ARRAYS",
    "encode_automaton",
    "pack_payload",
    "unpack_payload",
]

FORMAT_VERSION = 1
META_ENTRY = "meta.json"
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

_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def _entry(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    return info


def pack_payload(meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    """``meta`` and ``arrays`` into the wire string. No validation (tests craft corrupt payloads)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        meta_bytes = json.dumps(dict(meta), sort_keys=True, separators=(",", ":")).encode()
        zf.writestr(_entry(META_ENTRY), meta_bytes, compresslevel=9)
        for name in sorted(arrays):
            npy = io.BytesIO()
            np.lib.format.write_array(npy, np.asarray(arrays[name]), allow_pickle=False)
            zf.writestr(_entry(f"{name}.npy"), npy.getvalue(), compresslevel=9)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def unpack_payload(payload: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Inverse of ``pack_payload``. No validation; errors propagate as the codec raises them."""
    raw = base64.b64decode(payload.encode("ascii"), validate=True)
    meta: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in zf.namelist():
            with zf.open(name) as f:
                if name == META_ENTRY:
                    meta = json.loads(f.read().decode())
                elif name.endswith(".npy"):
                    arrays[name[: -len(".npy")]] = np.lib.format.read_array(f, allow_pickle=False)
    return meta, arrays


def encode_automaton(compiled: CompiledAutomaton) -> str:
    """The wire string of a compiled automaton (T-0063's ``CompiledAutomaton``); no size cap."""
    t = compiled.tables
    meta: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "name": compiled.name,
        "vocab_size": int(compiled.vocab_size),
        "tokenizer_hash": compiled.tokenizer_hash,
        "schema_hash": compiled.schema_hash,
        "compiler_version": compiled.compiler_version,
        "n_states": int(compiled.n_states),
        "n_states_bucket": int(compiled.n_states_bucket),
        "n_edges": int(compiled.n_edges),
        "n_classes": int(t.n_classes),
        "is_dfa": bool(compiled.is_dfa),
        "k_max": int(t.k_max),
        "max_neg_size": int(t.max_neg_size),
    }
    arrays = {name: np.asarray(getattr(compiled, name)) for name in AUTOMATON_ARRAYS}
    arrays.update({name: np.asarray(getattr(t, name)) for name in TABLE_ARRAYS})
    return pack_payload(meta, arrays)
