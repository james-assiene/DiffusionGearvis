"""The v1 tool schemas (T-0080a criterion 1): load, validate and version ``tools_v1.json``.

The file is a JSON list of function declarations ``{"name", "description", "parameters"}`` where
``parameters`` is a JSON Schema object. ``load()`` checks what the grammar compiler and the scorer rely
on (unique names, ``required`` a subset of ``properties``, no ``$ref`` anywhere, each ``parameters`` a
valid JSON Schema of type ``object``) and versions the set by the sha256 of the file's bytes.
The opt-in live sets extend v1 in separate files (T-0140 ruling R-p): ``page_overview`` (T-0140) and ``wait`` (T-0202).
T-0215: each of the four live sets has a ``*_nosearch.json`` twin without ``google_search`` (a live session with no
search backend declares it); the remaining declarations are the twin's, in its order.

Pure Python plus ``jsonschema``: the conductor imports this, so nothing here may import
``local_astra.decoding``, torch, numpy or outlines (criterion 6).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

__all__ = [
    "TOOLS_V1_NOSEARCH_PATH",
    "TOOLS_V1_OVERVIEW_NOSEARCH_PATH",
    "TOOLS_V1_OVERVIEW_PATH",
    "TOOLS_V1_OVERVIEW_WAIT_NOSEARCH_PATH",
    "TOOLS_V1_OVERVIEW_WAIT_PATH",
    "TOOLS_V1_PATH",
    "TOOLS_V1_WAIT_NOSEARCH_PATH",
    "TOOLS_V1_WAIT_PATH",
    "ToolSet",
    "load",
]

TOOLS_V1_PATH = Path(__file__).with_name("tools_v1.json")
#: T-0140 (ruling R-p): v1's five declarations, in order, plus ``page_overview``; opt-in, live only.
TOOLS_V1_OVERVIEW_PATH = Path(__file__).with_name("tools_v1_overview.json")
#: T-0202 A1/A2 (the R-p pattern): v1's five declarations, then ``wait`` (no arguments); and the overview set's
#: six, then the same ``wait``. Opt-in, live only.
TOOLS_V1_WAIT_PATH = Path(__file__).with_name("tools_v1_wait.json")
TOOLS_V1_OVERVIEW_WAIT_PATH = Path(__file__).with_name("tools_v1_overview_wait.json")
#: T-0215 (the R-p pattern): the four sets above without ``google_search``, the rest byte-identical and in order;
#: declared only by a live session with no search backend.
TOOLS_V1_NOSEARCH_PATH = Path(__file__).with_name("tools_v1_nosearch.json")
TOOLS_V1_OVERVIEW_NOSEARCH_PATH = Path(__file__).with_name("tools_v1_overview_nosearch.json")
TOOLS_V1_WAIT_NOSEARCH_PATH = Path(__file__).with_name("tools_v1_wait_nosearch.json")
TOOLS_V1_OVERVIEW_WAIT_NOSEARCH_PATH = Path(__file__).with_name(
    "tools_v1_overview_wait_nosearch.json"
)


@dataclass(frozen=True)
class ToolSet:
    """A validated set of tool declarations; ``version`` is the sha256 hex of the file's bytes."""

    declarations: tuple[Mapping[str, Any], ...]
    version: str
    path: Path

    @property
    def names(self) -> tuple[str, ...]:
        """Tool names in file order."""
        return tuple(str(d["name"]) for d in self.declarations)

    def get(self, name: str) -> Mapping[str, Any] | None:
        for d in self.declarations:
            if d["name"] == name:
                return d
        return None

    def parameters(self, name: str) -> dict[str, Any]:
        """The arguments schema of ``name``; ``KeyError`` if the set has no such tool."""
        d = self.get(name)
        if d is None:
            raise KeyError(name)
        return dict(d["parameters"])

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return iter(self.declarations)

    def __len__(self) -> int:
        return len(self.declarations)


def _has_ref(node: Any) -> bool:
    if isinstance(node, Mapping):
        return "$ref" in node or any(_has_ref(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_ref(v) for v in node)
    return False


def _validate(raw: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("a tools file is a non-empty JSON list of function declarations")
    seen: set[str] = set()
    for i, d in enumerate(raw):
        if not isinstance(d, dict):
            raise ValueError(f"declaration {i} is not an object")
        name = d.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"declaration {i} has no string `name`")
        if name in seen:
            raise ValueError(f"duplicate tool name {name!r}")
        seen.add(name)
        params = d.get("parameters")
        if not isinstance(params, dict) or params.get("type") != "object":
            raise ValueError(f"{name}: `parameters` must be a JSON Schema of type object")
        props = params.get("properties")
        if not isinstance(props, dict):
            raise ValueError(f"{name}: `parameters.properties` must be an object")
        required = params.get("required", [])
        if not isinstance(required, list) or not set(required) <= set(props):
            raise ValueError(f"{name}: `required` {required!r} is not a subset of the properties")
        if _has_ref(d):
            raise ValueError(f"{name}: `$ref` is not supported")
        try:
            jsonschema.validators.validator_for(params).check_schema(params)
        except jsonschema.SchemaError as exc:
            raise ValueError(f"{name}: invalid JSON Schema: {exc.message}") from exc
    return tuple(raw)


def load(path: str | Path | None = None) -> ToolSet:
    """Load and validate a tools file (default: the shipped v1 set); ``ValueError`` if invalid."""
    p = Path(path) if path is not None else TOOLS_V1_PATH
    data = p.read_bytes()
    try:
        raw = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p}: not JSON: {exc}") from exc
    return ToolSet(
        declarations=_validate(raw),
        version=hashlib.sha256(data).hexdigest(),
        path=p,
    )
