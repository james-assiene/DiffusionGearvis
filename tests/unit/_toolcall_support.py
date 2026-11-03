"""Shared helpers for the T-0080a rows (tool schemas v1, router, scorer, tool-call grammars).

Nothing here imports the code under test at module level. ``need()`` imports it inside a row, so while
``local_astra.toolcall`` does not exist every row fails *individually* with an ``AssertionError`` (the
pending conftest tolerates only that and ``NotImplementedError``) naming the
missing module (the fast suite keeps collecting).

API the rows assume (proposals where docs/tasks/T-0080a.md is silent; see the K-notes in the module
docstrings):

- ``tools.load(path=None) -> ToolSet``; ``ToolSet.version`` is the sha256 hex of the file's bytes; an
  invalid file raises ``ValueError`` (pydantic's ``ValidationError`` is one).
- ``tools_v1.json`` holds Gemini/BFCL-style function declarations ``{"name", "description",
  "parameters": <JSON Schema object>}``, either as a top-level list or under one list-valued key.
- ``router.route(answer_text, *, thought_open=False) -> Route`` with ``kind``, ``tag``,
  ``spoken_prefix``, ``draft_call_text``, ``draft_name``; ``tag`` is ``"cc"``, ``"genui"``,
  ``"native"`` or ``None``.
- ``score.parse_call(text)`` -> a mapping ``{"name", "arguments"}`` (or an object with those
  attributes) on success; a reason (``str``, or an object with a ``reason`` attribute) on failure.
- ``score.schema_valid(call, toolset) -> bool``; ``score.exact(call, gt) -> bool`` with ``gt`` as built
  by :func:`gt` below.
"""

from __future__ import annotations

import copy
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
TOOLCALL_DIR = REPO / "local_astra" / "toolcall"
TOOLS_V1 = TOOLCALL_DIR / "tools_v1.json"

#: Q1 (human decision 2026-10-08): exactly these five tools in v1.
V1_NAMES = frozenset({"google_search", "click_at", "type_at", "scroll", "press_key"})

#: The literal text of each router tag (criterion 2; umbrella "Room for GenUI"; Q3).
TAG_LITERAL = {"cc": "<cc>", "genui": "<genui>", "native": "<|tool_call>"}


def need(modname: str) -> ModuleType:
    """Import a ``local_astra.toolcall`` module, failing the row (not the collection) if it is absent."""
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing.startswith("local_astra.toolcall") or (
            missing.startswith("local_astra.decoding.toolcall")
        ):
            raise AssertionError(
                f"T-0080a not implemented: `{modname}` does not exist ({exc})"
            ) from exc
        raise


def tools_mod() -> ModuleType:
    return need("local_astra.toolcall.tools")


def router_mod() -> ModuleType:
    return need("local_astra.toolcall.router")


def score_mod() -> ModuleType:
    return need("local_astra.toolcall.score")


def toolset() -> Any:
    return tools_mod().load()


# --------------------------------------------------------------------------- the shipped file


def raw_tools() -> Any:
    if not TOOLS_V1.is_file():
        raise AssertionError(f"T-0080a criterion 1: {TOOLS_V1.relative_to(REPO)} does not exist")
    return json.loads(TOOLS_V1.read_bytes())


def declarations(raw: Any) -> list[dict[str, Any]]:
    """The declaration list inside a parsed ``tools_v1.json`` (top-level list, or one list-valued key)."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        lists = [
            v
            for v in raw.values()
            if isinstance(v, list) and v and all(isinstance(d, dict) and "name" in d for d in v)
        ]
        if len(lists) == 1:
            return lists[0]
    raise AssertionError(
        "tools_v1.json: expected a list of function declarations, or an object with exactly one "
        f"list of declarations; got {type(raw).__name__}"
    )


def params_of(decl: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON Schema of a declaration's arguments (``parameters``, or ``function.parameters``)."""
    if "parameters" in decl:
        return dict(decl["parameters"])
    fn = decl.get("function")
    if isinstance(fn, Mapping) and "parameters" in fn:
        return dict(fn["parameters"])
    raise AssertionError(f"declaration {decl.get('name')!r} has no `parameters` schema")


def decl_by_name(raw: Any) -> dict[str, dict[str, Any]]:
    return {str(d["name"]): d for d in declarations(raw)}


def shipped_params(name: str) -> dict[str, Any]:
    decls = decl_by_name(raw_tools())
    assert name in decls, f"tools_v1.json has no declaration {name!r} (has {sorted(decls)})"
    return params_of(decls[name])


#: D1 (human decision 2026-10-08, after the size report): ``press_key(combo)``, one string from a fixed
#: list of ~20 keys and shortcuts. The named members below must all be present; the four arrows are named
#: by the file (``Up``/``ArrowUp``...), so they are matched by direction in ``combo_enum``'s check.
D1_NAMED_COMBOS = frozenset(
    {
        "Enter", "Escape", "Tab", "Backspace", "Delete", "Space", "Home", "End", "PageUp", "PageDown",
        "Ctrl+A", "Ctrl+C", "Ctrl+V", "Ctrl+X", "Ctrl+Z", "Ctrl+L", "Ctrl+T", "Ctrl+W", "Alt+Tab",
        "Ctrl+Enter",
    }
)  # fmt: skip


def combo_enum() -> list[str]:
    """The shipped ``press_key.combo`` enum (D1): a list of strings."""
    combo = shipped_params("press_key")["properties"].get("combo")
    assert isinstance(combo, dict), "D1: press_key must declare a `combo` property"
    enum = combo.get("enum")
    assert isinstance(enum, list) and all(isinstance(e, str) for e in enum), (
        f"D1: press_key.combo must be a string enum; got {combo!r}"
    )
    return list(enum)


def arrow_combos(enum: list[str]) -> dict[str, list[str]]:
    """The enum members naming each arrow direction (``Up``, ``ArrowUp``...; never ``PageUp``)."""
    return {
        d: [e for e in enum if e.endswith(d) and "Page" not in e and "+" not in e]
        for d in ("Up", "Down", "Left", "Right")
    }


def write_mutated(tmp_path: Path, mutate: Any, name: str = "tools.json") -> Path:
    """A copy of the shipped file with ``mutate(raw)`` applied (in place), written under ``tmp_path``."""
    raw = copy.deepcopy(raw_tools())
    mutate(raw)
    path = tmp_path / name
    path.write_text(json.dumps(raw, indent=2))
    return path


# --------------------------------------------------------------------------- results


def call_parts(res: Any) -> tuple[str, Any] | None:
    """``(name, arguments)`` of a successful ``parse_call`` result; ``None`` for a failure."""
    if isinstance(res, Mapping):
        if "name" in res and "arguments" in res:
            return str(res["name"]), res["arguments"]
        return None
    if isinstance(res, str):
        return None
    if hasattr(res, "name") and hasattr(res, "arguments") and not hasattr(res, "reason"):
        return str(res.name), res.arguments
    return None


def failure_reason(res: Any) -> str | None:
    """The reason of a failed ``parse_call`` result; ``None`` if ``res`` is a success."""
    if call_parts(res) is not None:
        return None
    if isinstance(res, str):
        return res
    reason = getattr(res, "reason", None)
    if reason is None and isinstance(res, Mapping):
        reason = res.get("reason")
    return None if reason is None else str(reason)


def call(name: str, **arguments: Any) -> dict[str, Any]:
    """A parsed call as ``parse_call`` returns it on success (criterion 3: ``{name, arguments}``)."""
    return {"name": name, "arguments": dict(arguments)}


def gt(name: str, **fields: Any) -> dict[str, Any]:
    """A ground-truth case (K-proposal; T-0080b's eval set uses this shape).

    - ``google_search``: ``keywords`` (list of str);
    - ``click_at``: ``bbox`` = ``[x0, y0, x1, y1]``, normalised 0-999, inclusive;
    - ``type_at``: ``bbox``, ``text``, ``press_enter``;
    - ``scroll``: ``direction``;
    - ``press_key``: ``combo`` (str; compared by string equality, D1).
    """
    return {"name": name, **fields}


# --------------------------------------------------------------------------- T-0085: the T-0084e stop, verbatim

#: The native pass-1 ``text`` of case ``click-01`` in the stopped T-0084e dev run at 9aee66d (the main checkout's
#: ``logs/t0084-native-dev-9aee66d/20261010T224150.646710Z-toolcall-dev.jsonl``, sha256 ``31f9fd7f...``, the
#: ``"pass": 1, "format": "native"`` row), copied verbatim. The model wrote a stray ``<channel|>`` after its spoken
#: sentence; the router at 9aee66d kept it in ``spoken_prefix`` and vLLM refused the T4 continuation with 400.
CLICK01_NATIVE_TEXT = (
    '<|channel>thought\nThe user wants to click the "Sign in" button.\nI need to find the coordinates of the '
    '"Sign in" button on the screen.\nThe "Sign in" button is a blue button located at approximately (488, 517).'
    "<channel|>I will click the Sign in button.<channel|><|tool_call>call:click_at{x:488,y:517}<tool_call|>"
)
#: Its ``token_ids`` in the same row (the decode up to the stop id 50 is ``CLICK01_NATIVE_TEXT``).
_CLICK01_IDS_TEXT = (
    "100 45518 107 818 2430 8150 531 4845 506 623 10150 528 236775 4309 236761 107 236777 1202 531 1586 "
    "506 15375 529 506 623 10150 528 236775 4309 580 506 4322 236761 107 818 623 10150 528 236775 4309 "
    "563 496 3730 4309 5628 657 8296 568 236812 236828 236828 236764 236743 236810 236770 236832 769 101 "
    "236777 795 4845 506 9753 528 4309 236761 101 48 6639 236787 4513 236779 502 236782 236781 236787 "
    "236812 236828 236828 236764 236762 236787 236810 236770 236832 236783 49 50"
)
CLICK01_NATIVE_IDS = [int(t) for t in _CLICK01_IDS_TEXT.split()]
#: What the router at 9aee66d stored for that row (``route.spoken_prefix``), and what T-0085 criterion 2 says it is.
CLICK01_STORED_SPOKEN = "I will click the Sign in button.<channel|>"
CLICK01_SPOKEN = "I will click the Sign in button."
CLICK01_DRAFT = "call:click_at{x:488,y:517}<tool_call|>"
#: The cc pass-1 ``text`` of the same case in the same run (it routed cleanly: no stray close).
CLICK01_CC_TEXT = (
    '<|channel>thought\nThe user wants to click the "Sign in" button.\nI need to find the coordinates of the '
    '"Sign in" button on the screen.\nThe "Sign in" button is located at approximately (499, 515).<channel|>'
    'I will click the Sign in button.\n<cc>{"name": "click_at", "arguments": {"x": 499, "y": 515}}'
)
#: The structural markers of T-0085 criterion 3(b), verbatim from the task.
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
