"""T-0095: name what differs between two lists of SSE chunk dicts, in full (no pytest truncation).

Used by T-0088's gpu row 2 (``tests/contract/test_llm_tools_wire_gpu.py``) to say how the live multi-canvas shape
differs from ``tests/fakes/vllm_sse.py``. CPU-only, no imports from ``local_astra``; its unit row is
``tests/unit/test_chunk_shape_diff.py``.

Equality is JSON equality, as the row's strict comparison (``json.dumps`` of each chunk) sees it: types count
(``1`` vs ``1.0`` vs ``True`` differ) and so does the key order of an object.
"""

from __future__ import annotations

from typing import Any

ABSENT = "<absent>"


def _same(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def key_diffs(a: Any, b: Any, path: str = "") -> list[tuple[str, str, str]]:
    """Every differing key path between ``a`` and ``b``: ``(path, repr-or-<absent> of a, repr-or-<absent> of b)``."""
    if _same(a, b):
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[tuple[str, str, str]] = []
        for k in list(a) + [k for k in b if k not in a]:
            p = _join(path, k)
            if k not in b:
                out.append((p, repr(a[k]), ABSENT))
            elif k not in a:
                out.append((p, ABSENT, repr(b[k])))
            else:
                out.extend(key_diffs(a[k], b[k], p))
        if not out:  # same keys and values, different order
            out.append((f"{path or '<chunk>'} key order", repr(list(a)), repr(list(b))))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out.extend(key_diffs(x, y, f"{path}[{i}]"))
        return out
    return [(path or "<chunk>", repr(a), repr(b))]


def first_mismatch(
    a: list[dict[str, Any]],
    b: list[dict[str, Any]],
    a_name: str = "built",
    b_name: str = "live",
) -> str | None:
    """``None`` when the lists are JSON-equal; otherwise a message naming the first differing chunk index and every
    differing key path in that chunk as ``path: <a value> vs <b value>``, values in full."""
    n = max(len(a), len(b))
    for i in range(n):
        if i >= len(a) or i >= len(b):
            left = repr(a[i]) if i < len(a) else "<absent chunk>"
            right = repr(b[i]) if i < len(b) else "<absent chunk>"
            return (
                f"chunk counts differ ({a_name} {len(a)} vs {b_name} {len(b)}); first differing chunk index {i}: "
                f"{left} vs {right}"
            )
        diffs = key_diffs(a[i], b[i])
        if diffs:
            lines = [
                f"first differing chunk index {i} of {a_name} {len(a)} vs {b_name} {len(b)} "
                f"(values: {a_name} vs {b_name}):"
            ]
            lines += [f"  {p}: {x} vs {y}" for p, x, y in diffs]
            return "\n".join(lines)
    return None
