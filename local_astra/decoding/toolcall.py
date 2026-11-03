"""Tool-call grammars with the thought variants T1/T2/T4 (T-0080a criterion 4). ``cd`` group only.

``compile_toolcall(tools, *, scope, thought, whitespace="json")`` compiles the constrained pass of the
two-pass tool-call mode (umbrella ``docs/tasks/T-0080.md``, "Thought handling") into a
``CompiledAutomaton`` with the vendored diffgemma-fa compiler, over the pinned HF vocabulary:

- ``CALL`` = ``{"name": <const>, "arguments": <the tool's parameters schema>}`` in the chosen JSON
  whitespace, keys in declared order. ``scope=<tool name>`` is that tool's CALL alone (the live path,
  D2); ``scope="union"`` is the alternation of every tool's CALL (eval/reporting only, D2);
  ``scope="name"`` is the name picker ``{"name": "<one of the v1 names>"`` (below);
- ``T1`` = ``100 · Σ_name{1,8} · 107 · FREE* · 101 · ["<cc>"] · WS · CALL`` -- a regenerated thought.
  FREE is every id except ``RESERVED_TOKENS`` (PAD, the stop ids, 100, 101, 105) and the stop set of the
  pinned ``generation_config.json``. The single-id markers make "no 101 inside the thought" a single
  self-looping state (SPEC §3.6), built as DFA surgery like ``prepend_channel_header``;
- ``T2`` = the existing ``prepend_channel_header`` (``100 · Σ_name{1,8} · 107 · 101``, the empty
  thought) ``· ["<cc>"] · WS · CALL``;
- ``T4`` = ``WS · CALL``: no header and no tag (the pass-2 prompt already ends with ``<cc>``).

``["<cc>"] · WS`` is part of the byte-level regex, so the lift admits every tokenisation of the tag and
JSON whitespace between the tag and ``{`` (ruling K13). Every automaton is ``FA · STOP · Σ*`` (T-0063).

**Integer bounds (``click_at``/``type_at`` x, y in 0-999).** outlines-core drops ``minimum``/``maximum``
(``_vendor/dgfa_compile/schema.py``, ``_SILENTLY_DROPPED``), so the bounds are enforced here instead of
being allowed through: before the schema reaches outlines, every ``{"type": "integer", "minimum": 0,
"maximum": 10**k - 1}`` node is replaced by a unique string ``const`` sentinel, and the sentinel's
rendering in the regex (``"<sentinel>"``, exactly once) is replaced by ``(0|[1-9][0-9]{0,k-1})``, the
canonical JSON integers 0 .. 10**k - 1. Any other numeric bound raises (fail loud, as the vendored
pre-pass does); ``minimum``/``maximum`` are never put on the ``allow`` list.

**The name picker (D2, ``scope="name"``).** When the draft names no v1 tool, pass 2 first decodes
``[header] ["<cc>"] WS {"name": "<name>"`` + STOP under the same thought variant; the conductor then
runs the per-tool grammar of the picked name. The picker's language is the literal prefix every per-tool
CALL starts with (checked in ``name_regex``), not a bare name, because (1) that is how the model writes a
call (``{"name": "`` after ``<cc>`` is on-distribution, a bare ``google_search`` is not), and (2) the
picked canvas is then token for token a prefix of the second step's language, so step 2 can be seeded
with it (T-0081) or simply re-derive the name, which its grammar pins as a const. Step 2 is an ordinary
pass-2 request with ``compile_toolcall(tools, scope=<picked>, thought=<same>)``; the second automaton
does not depend on the first. T2/T4 are the sensible live variants for the picker; T1 is compiled too,
but it re-denoises the thought in both steps.

**The native format (T-0084a, ``call_format="native"``).** The model's own call syntax, as the pinned
chat template (sha ``36e3a42e...``) renders it: ``CALL_native = call:NAME{ARGS}`` · 49, ARGS = every declared
property in jinja ``dictsort`` order (case-insensitive) as ``key:VALUE`` joined by ``,``, no whitespace
anywhere. VALUE: a bounded integer as above, an unbounded one ``-?(0|[1-9][0-9]*)``, a boolean
``true|false``, a string enum ``52 · literal · 52``, a string ``52 · BODY · 52`` with BODY = any text with
no ``<|`` and no ``|>`` substring (non-empty under ``minLength: 1``). Variants: ``T4`` = CALL_native (the
pass-2 prompt ends with 48); ``T2`` = the empty-thought header · 48 · CALL_native; ``T1`` =
``prepend_thought_header`` · 48 · CALL_native with 48 and 49 also out of the thought's FREE set (QN5);
``scope="name"`` = [header] [48, not in T4] ``call:NAME{`` + STOP.

*How the markers stay single ids.* ``build_vocabulary()`` maps 48/49/52 to their literal bytes, so lifting
``<|"|>`` as text would also admit its multi-token spellings, and a private-use placeholder is spellable by
byte-fallback pieces. Neither is used: the native grammar is a **concatenation of lifted text segments
joined by marker edges** (DFA surgery). Every text segment is lifted on its own with 48, 49 and 52 dropped
from its labels (with the reserved ids), and the segments are chained by single-id edges ``final --m-->
next start``. A token never spans a marker (each marker is its own id), so the token language of the
chain is exactly the segments' token languages joined by the markers, and every lift keeps all the
tokenisations of the plain text. The marker labels are disjoint from every segment's labels, so the
chain is deterministic; it is minimized once more. ``union`` (eval-only) is the subset construction over
the per-tool calls, minimized. ``build_vocabulary()``'s default is untouched.

``report_sizes()`` prints |S| and the bucket of every (call_format, scope, thought) for the v1 tools.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from local_astra.decoding import vocab
from local_astra.decoding._vendor.dgfa_compile import automaton as _automaton
from local_astra.decoding._vendor.dgfa_compile import schema as _schema
from local_astra.decoding._vendor.dgfa_compile import vocab as _vendored_vocab
from local_astra.decoding._vendor.dgfa_compile.lift import lift_regex
from local_astra.decoding._vendor.dgfa_compile.minimize import Dfa, minimize
from local_astra.decoding.grammars import HEADER_TOKENS, WHITESPACE_PATTERNS, compiler_version
from local_astra.toolcall.tools import ToolSet

__all__ = [
    "CALL_FORMATS",
    "CC_TAG",
    "LIVE_MAX_STATES",
    "NAME_SCOPE",
    "THOUGHTS",
    "UNION_SCOPE",
    "call_regex",
    "compile_toolcall",
    "name_regex",
    "native_call_parts",
    "native_name_regex",
    "prepend_thought_header",
    "report_sizes",
]

#: The thought variants (umbrella T-0080; T3 is rejected for now, Q5).
THOUGHTS: tuple[str, ...] = ("T1", "T2", "T4")
#: The router's tool-call tag (``local_astra.toolcall.router.TAGS["cc"]``).
CC_TAG = "<cc>"

CHANNEL_OPEN = 100
CHANNEL_CLOSE = 101
NEWLINE = 107

#: D2: the name picker's scope, and the all-tools alternation (eval/reporting only).
NAME_SCOPE = "name"
UNION_SCOPE = "union"
#: ``.claude/rules/gpu-services.md``: automata on the live path must be <= 128 states (build gate).
LIVE_MAX_STATES = 128

_SENTINEL = "ZZT0080aINT{}ZZ"

#: T-0084a: the tool-call formats. ``json`` is the T-0080a format (the default, payloads unchanged).
CALL_FORMATS: tuple[str, ...] = ("json", "native")
#: The native format's structural markers, each a single id of the pinned tokenizer.
TOOL_CALL_OPEN = 48  # <|tool_call>
TOOL_CALL_CLOSE = 49  # <tool_call|>
STRING_DELIM = 52  # <|"|>
NATIVE_MARKERS = frozenset({TOOL_CALL_OPEN, TOOL_CALL_CLOSE, STRING_DELIM})

#: One step of BODY's loop: no ``<|`` and no ``|>`` (a 3-state byte automaton: after ``<``, after ``|``,
#: elsewhere). Every item ends on a character that is neither ``<`` nor ``|``, so items concatenate
#: freely; the tail ``\|*<*`` is the only way to end on ``|`` or ``<``.
_BODY_ITEM = r"(?:[^<|]|\|*<+[^<|]|\|+[^<|>])"
#: Any text with no ``<|`` and no ``|>`` substring (empty included; documentation and checks only --
#: the grammar lifts the non-empty form and adds the empty string on the automaton, ``MaybeEmpty``).
BODY_REGEX = rf"{_BODY_ITEM}*\|*<*"
#: The same, non-empty (``minLength: 1``).
BODY_NONEMPTY_REGEX = rf"(?:{_BODY_ITEM}+\|*<*|{_BODY_ITEM}*(?:\|+<*|<+))"


def _int_range_regex(minimum: Any, maximum: Any, path: str) -> str:
    """``(0|[1-9][0-9]{0,k-1})`` for ``[0, 10**k - 1]``; anything else is refused."""
    ok = (
        isinstance(minimum, int)
        and isinstance(maximum, int)
        and minimum == 0
        and maximum >= 9
        and set(str(maximum)) == {"9"}
    )
    if not ok:
        raise ValueError(
            f"{path}: integer bounds [{minimum}, {maximum}] are not supported "
            "(only [0, 10**k - 1]; outlines drops numeric bounds)"
        )
    k = len(str(maximum))
    return f"(0|[1-9][0-9]{{0,{k - 1}}})"


def _bounded_ints(node: Any, out: dict[str, str], path: str) -> Any:
    """``node`` with every bounded integer replaced by a const sentinel recorded in ``out``."""
    if isinstance(node, list):
        return [_bounded_ints(v, out, f"{path}[{i}]") for i, v in enumerate(node)]
    if not isinstance(node, dict):
        return node
    if "minimum" in node or "maximum" in node:
        if node.get("type") != "integer":
            raise ValueError(f"{path}: numeric bounds on a non-integer are not supported")
        sentinel = _SENTINEL.format(len(out))
        out[sentinel] = _int_range_regex(node.get("minimum"), node.get("maximum"), path)
        return {"const": sentinel}
    return {k: _bounded_ints(v, out, f"{path}.{k}") for k, v in node.items()}


def call_regex(name: str, parameters: Mapping[str, Any], whitespace_pattern: str | None) -> str:
    """The byte-level regex of ``{"name": name, "arguments": <parameters>}``."""
    sentinels: dict[str, str] = {}
    params = _bounded_ints(copy.deepcopy(dict(parameters)), sentinels, name)
    call_schema = {
        "type": "object",
        "properties": {"name": {"const": name}, "arguments": params},
        "required": ["name", "arguments"],
    }
    regex = _schema.build_regex(call_schema, whitespace_pattern=whitespace_pattern)
    for sentinel, int_regex in sentinels.items():
        literal = f'"{sentinel}"'
        if regex.count(literal) != 1:
            raise AssertionError(f"{name}: sentinel {sentinel} rendered {regex.count(literal)}x")
        regex = regex.replace(literal, int_regex)
    return regex


def prepend_thought_header(
    dfa: Dfa,
    *,
    vocab_size: int,
    reserved: Iterable[int],
    max_name_tokens: int = HEADER_TOKENS,
) -> Dfa:
    """``100 · Σ_name{1,n} · 107 · FREE* · 101 · dfa`` (T1), as DFA surgery.

    The name part is ``prepend_channel_header``'s (same alphabet, same bound); the state after the
    name's newline loops on FREE = every id outside ``reserved ∪ {100, 101}`` and leaves on 101. FREE
    and 101 are disjoint, so the result stays deterministic.
    """
    if max_name_tokens < 1:
        raise ValueError("max_name_tokens must be >= 1")
    reserved = frozenset(int(t) for t in reserved)
    vocab_ids = frozenset(range(vocab_size))
    name_alphabet = sorted(vocab_ids - {CHANNEL_OPEN, CHANNEL_CLOSE, NEWLINE, *reserved})
    free = sorted(vocab_ids - {CHANNEL_OPEN, CHANNEL_CLOSE, *reserved})

    h_open = 0
    consumed = list(range(1, 2 + max_name_tokens))
    in_thought = consumed[-1] + 1
    shift = in_thought + 1

    trans: list[tuple[int, int, int]] = [(h_open, CHANNEL_OPEN, consumed[0])]
    for i in range(max_name_tokens):
        trans.extend((consumed[i], v, consumed[i + 1]) for v in name_alphabet)
    for i in range(1, max_name_tokens + 1):
        trans.append((consumed[i], NEWLINE, in_thought))
    trans.extend((in_thought, v, in_thought) for v in free)
    trans.append((in_thought, CHANNEL_CLOSE, dfa.start + shift))
    trans.extend((src + shift, lbl, dst + shift) for src, lbl, dst in dfa.transitions)
    return Dfa(
        n_states=dfa.n_states + shift,
        transitions=tuple(trans),
        start=h_open,
        finals=frozenset(f + shift for f in dfa.finals),
    )


def name_regex(tools: ToolSet, whitespace_pattern: str | None) -> str:
    """The name picker: ``{"name": "<one v1 name>"`` -- the shared prefix of every tool's CALL.

    Built in outlines' own object rendering and checked to be a literal prefix of each tool's CALL
    regex, so the picker's output is a prefix the picked tool's per-tool grammar also accepts.
    """
    ws = whitespace_pattern if whitespace_pattern is not None else "[ ]?"
    head = r"\{" + ws + '"name"' + ws + ":" + ws
    alts = []
    for n in tools.names:
        literal = '"' + re.escape(n) + '"'
        call = call_regex(n, tools.parameters(n), whitespace_pattern)
        if not call.startswith(head + literal):
            raise AssertionError(
                f"{n}: CALL does not start with the name prefix {head + literal!r}"
            )
        alts.append(literal)
    return head + "(" + "|".join(alts) + ")"


def _lit(text: str) -> str:
    """``text`` as a regex literal: ASCII letters, digits and ``_`` as themselves, the rest ``\\x{..}``."""
    return "".join(
        c if c.isascii() and (c.isalnum() or c == "_") else f"\\x{{{ord(c):X}}}" for c in text
    )


@dataclasses.dataclass(frozen=True)
class MaybeEmpty:
    """A native text segment that may also be empty: ``regex`` is its non-empty part. outlines-core
    cannot index a regex that matches the empty string under the ``\\z`` anchor (no anchored universal
    start state), so the empty case is added on the lifted automaton instead (``_or_empty``)."""

    regex: str


#: A native part: regex text (never empty-matching), a marker id, or a possibly-empty text segment.
NativePart = str | int | MaybeEmpty


def _native_value(schema: Mapping[str, Any], path: str) -> list[NativePart]:
    """The parts of one native VALUE (criterion 2): regex text, and 52 around strings."""
    t = schema.get("type")
    known = {"type", "description"}
    if t == "integer":
        known |= {"minimum", "maximum"}
        if "minimum" in schema or "maximum" in schema:
            value = _int_range_regex(schema.get("minimum"), schema.get("maximum"), path)
        else:
            value = "-?(0|[1-9][0-9]*)"
        parts: list[NativePart] = [value]
    elif t == "boolean":
        parts = ["(true|false)"]
    elif t == "string":
        known |= {"enum", "minLength"}
        min_length = schema.get("minLength", 0)
        if not isinstance(min_length, int) or min_length not in (0, 1):
            raise ValueError(f"{path}: minLength {min_length!r} is not supported natively (0 or 1)")
        if "enum" in schema:
            members = schema["enum"]
            if not members or not all(isinstance(m, str) for m in members):
                raise ValueError(f"{path}: a native enum must list strings")
            literals = [_lit(m) for m in members if m]
            body: list[NativePart] = []
            if literals:
                alts = "(" + "|".join(literals) + ")"
                body = [MaybeEmpty(alts) if "" in members else alts]
        elif min_length == 1:
            body = [BODY_NONEMPTY_REGEX]
        else:
            body = [MaybeEmpty(BODY_NONEMPTY_REGEX)]
        parts = [STRING_DELIM, *body, STRING_DELIM]
    else:
        raise ValueError(f"{path}: type {t!r} is not supported by the native format")
    extra = sorted(set(schema) - known)
    if extra:
        raise ValueError(f"{path}: keywords {extra} are not supported by the native format")
    return parts


def native_call_parts(name: str, parameters: Mapping[str, Any]) -> list[NativePart]:
    """``call:NAME{ARGS}`` · 49 as alternating regex text and marker ids (52, 49), adjacent text merged.

    ARGS lists every property in jinja ``dictsort`` order (case-insensitive, stable), as ``key:VALUE``
    joined by ``,``. An optional property or a property type other than integer, boolean or string
    raises ``ValueError``.
    """
    if parameters.get("type") != "object":
        raise ValueError(f"{name}: parameters must be an object")
    props = parameters.get("properties", {})
    required = set(parameters.get("required", ()))
    if set(props) != required:
        raise ValueError(
            f"{name}: optional properties {sorted(set(props) - required)} are not supported "
            "by the native format (every property must be required)"
        )
    raw: list[NativePart] = ["call:" + _lit(name) + r"\{"]
    for i, key in enumerate(sorted(props, key=str.lower)):
        raw.append(("," if i else "") + _lit(key) + ":")
        raw.extend(_native_value(props[key], f"{name}.{key}"))
    raw.extend([r"\}", TOOL_CALL_CLOSE])
    parts: list[NativePart] = []
    for p in raw:
        if isinstance(p, str) and parts and isinstance(parts[-1], str):
            parts[-1] += p
        else:
            parts.append(p)
    return parts


def native_name_regex(tools: ToolSet) -> str:
    """The native name picker ``call:(<one v1 name>)\\{``, checked to be a literal prefix of each tool's
    first native segment (so the picked canvas is token for token a prefix of the per-tool language)."""
    alts = []
    for n in tools.names:
        first = native_call_parts(n, tools.parameters(n))[0]
        literal = "call:" + _lit(n) + r"\{"
        if not (isinstance(first, str) and first.startswith(literal)):
            raise AssertionError(f"{n}: the native CALL does not start with {literal!r}")
        alts.append(_lit(n))
    return "call:(" + "|".join(alts) + r")\{"


@functools.cache
def _lift_native(regex: str) -> Dfa:
    """A text segment, lifted with the markers 48/49/52 (and the reserved ids) dropped from its labels."""
    drop = frozenset(_vendored_vocab.RESERVED_TOKENS | NATIVE_MARKERS)
    return lift_regex(regex, vocab.build_vocabulary(), drop_labels=drop).dfa


def _chain(parts: Iterable[Dfa | int]) -> Dfa:
    """Concatenate DFAs and single marker ids: ``final --m--> next``; a DFA's start is merged into the
    state its preceding marker enters (or the chain's start). Callers never put two DFAs side by side,
    so the merged state has no edges of its own yet; marker labels are disjoint from every DFA's labels,
    so the chain is deterministic."""
    trans: list[tuple[int, int, int]] = []
    entry, n = 0, 1
    finals: frozenset[int] = frozenset({0})
    fresh = True  # the entry state has no outgoing edges yet
    for part in parts:
        if isinstance(part, int):
            new = n
            n += 1
            trans.extend((f, part, new) for f in sorted(finals))
            entry, finals, fresh = new, frozenset({new}), True
            continue
        if not fresh:
            raise AssertionError("two text segments side by side: merge them into one regex")
        mapping: dict[int, int] = {}
        for s in range(part.n_states):
            if s == part.start:
                mapping[s] = entry
            else:
                mapping[s] = n
                n += 1
        trans.extend((mapping[s], lbl, mapping[d]) for s, lbl, d in part.transitions)
        finals, fresh = frozenset(mapping[f] for f in part.finals), False
    return Dfa(n_states=n, transitions=tuple(trans), start=0, finals=finals)


def _or_empty(dfa: Dfa) -> Dfa:
    """``dfa`` plus the empty string: a fresh accepting start state with copies of the start's edges."""
    s0 = dfa.n_states
    copies = tuple((s0, lbl, d) for s, lbl, d in dfa.transitions if s == dfa.start)
    return Dfa(
        n_states=s0 + 1,
        transitions=dfa.transitions + copies,
        start=s0,
        finals=dfa.finals | {s0},
    )


def _native_part_dfa(part: NativePart) -> Dfa | int:
    if isinstance(part, MaybeEmpty):
        return _or_empty(_lift_native(part.regex))
    if isinstance(part, str):
        return _lift_native(part)
    return part


def _native_call_dfa(name: str, parameters: Mapping[str, Any]) -> Dfa:
    parts = native_call_parts(name, parameters)
    return minimize(_chain(_native_part_dfa(p) for p in parts)).dfa


def _determinize(dfas: Sequence[Dfa]) -> Dfa:
    """The union of ``dfas`` by the subset construction (union scope, eval-only)."""
    out: dict[int, dict[int, int]] = {}
    starts: list[int] = []
    finals: set[int] = set()
    offset = 0
    for d in dfas:
        for s, lbl, t in d.transitions:
            out.setdefault(s + offset, {})[lbl] = t + offset
        starts.append(d.start + offset)
        finals.update(f + offset for f in d.finals)
        offset += d.n_states
    start = frozenset(starts)
    index = {start: 0}
    todo = [start]
    trans: list[tuple[int, int, int]] = []
    new_finals: set[int] = set()
    while todo:
        subset = todo.pop()
        src = index[subset]
        if subset & finals:
            new_finals.add(src)
        moves: dict[int, set[int]] = {}
        for s in subset:
            for lbl, t in out.get(s, {}).items():
                moves.setdefault(lbl, set()).add(t)
        for lbl in sorted(moves):
            dst_set = frozenset(moves[lbl])
            if dst_set not in index:
                index[dst_set] = len(index)
                todo.append(dst_set)
            trans.append((src, lbl, index[dst_set]))
    return Dfa(n_states=len(index), transitions=tuple(trans), start=0, finals=frozenset(new_finals))


_NATIVE_CALLS: dict[tuple[str, str], tuple[Dfa, list[Any]]] = {}


def _native_grammar(tools: ToolSet, scope: str, thought: str) -> tuple[Dfa, list[Any]]:
    """The native grammar before stop augmentation, and its source description (for the hash)."""
    if thought not in THOUGHTS:
        raise ValueError(f"thought must be one of {THOUGHTS}, not {thought!r}")
    key = (tools.version, scope)
    if key not in _NATIVE_CALLS:
        _NATIVE_CALLS[key] = _native_call(tools, scope)
    call, source = _NATIVE_CALLS[key]
    if thought == "T4":
        return call, source
    return _chain([TOOL_CALL_OPEN, call]), [TOOL_CALL_OPEN, *source]


def _native_call(tools: ToolSet, scope: str) -> tuple[Dfa, list[Any]]:
    """The scope's language after 48 (shared by T1/T2/T4), and its source description."""
    if scope in (NAME_SCOPE, UNION_SCOPE) and scope in tools.names:
        raise ValueError(f"a tool may not be called {scope!r} (it is a reserved scope)")
    source: list[Any]
    if scope == NAME_SCOPE:
        regex = native_name_regex(tools)
        call = _lift_native(regex)
        source = [regex]
    elif scope == UNION_SCOPE:
        call = minimize(
            _determinize([_native_call_dfa(n, tools.parameters(n)) for n in tools.names])
        ).dfa
        source = [native_call_parts(n, tools.parameters(n)) for n in tools.names]
    elif scope in tools.names:
        call = _native_call_dfa(scope, tools.parameters(scope))
        source = native_call_parts(scope, tools.parameters(scope))
    else:
        raise ValueError(
            f"scope must be {NAME_SCOPE!r}, {UNION_SCOPE!r} or one of {list(tools.names)}, "
            f"not {scope!r}"
        )
    return call, source


def _grammar_regex(tools: ToolSet, scope: str, thought: str, ws: str | None) -> str:
    if thought not in THOUGHTS:
        raise ValueError(f"thought must be one of {THOUGHTS}, not {thought!r}")
    if scope in (NAME_SCOPE, UNION_SCOPE) and scope in tools.names:
        raise ValueError(f"a tool may not be called {scope!r} (it is a reserved scope)")
    if scope == NAME_SCOPE:
        calls = name_regex(tools, ws)
    elif scope == UNION_SCOPE:
        calls = "|".join(f"({call_regex(n, tools.parameters(n), ws)})" for n in tools.names)
    elif scope in tools.names:
        calls = f"({call_regex(scope, tools.parameters(scope), ws)})"
    else:
        raise ValueError(
            f"scope must be {NAME_SCOPE!r}, {UNION_SCOPE!r} or one of {list(tools.names)}, "
            f"not {scope!r}"
        )
    lead = ws if ws is not None else "[ ]?"
    tag = "" if thought == "T4" else f"({re.escape(CC_TAG)})?"
    return f"{tag}{lead}({calls})"


@functools.cache
def _lift(regex: str) -> Dfa:
    return lift_regex(regex, vocab.build_vocabulary()).dfa


def _compile(version: str, regex: str, scope: str, thought: str, whitespace: str) -> Any:
    grammar = _lift(regex)
    source = {"kind": "toolcall", "tools_version": version, "scope": scope, "regex": regex}
    options = {"thought": thought, "whitespace": whitespace, "header_tokens": HEADER_TOKENS}
    return _finish(grammar, f"toolcall:{scope}:{thought}", thought, source, options, frozenset())


def _compile_native(tools: ToolSet, scope: str, thought: str) -> Any:
    grammar, parts = _native_grammar(tools, scope, thought)
    source = {
        "kind": "toolcall",
        "call_format": "native",
        "tools_version": tools.version,
        "scope": scope,
        "parts": parts,
    }
    options = {"thought": thought, "header_tokens": HEADER_TOKENS, "body": "no <| no |>"}
    # QN5: the native T1 thought may not hold 48 or 49.
    thought_extra = frozenset({TOOL_CALL_OPEN, TOOL_CALL_CLOSE})
    return _finish(
        grammar, f"toolcall:native:{scope}:{thought}", thought, source, options, thought_extra
    )


def _finish(
    grammar: Dfa,
    name: str,
    thought: str,
    source: dict[str, Any],
    options: dict[str, Any],
    thought_extra: frozenset[int],
) -> Any:
    """Header (T1/T2), stop augmentation, tables and the cache stamps."""
    vocab_size = int(vocab.load_tokenizer().get_vocab_size(with_added_tokens=True))
    end_tokens = vocab.stop_token_ids()
    reserved = tuple(
        sorted({*end_tokens, *_vendored_vocab.RESERVED_TOKENS, *thought_extra} - {100, 101})
    )
    if thought == "T1":
        grammar = prepend_thought_header(grammar, vocab_size=vocab_size, reserved=reserved)
    elif thought == "T2":
        grammar = _automaton.prepend_channel_header(
            grammar,
            vocab_size=vocab_size,
            max_name_tokens=HEADER_TOKENS,
            reserved=(*end_tokens, _vendored_vocab.PAD_TOKEN),
        )
    compiled = _automaton.compile_automaton(
        grammar, name=name, end_tokens=end_tokens, vocab_size=vocab_size
    )
    return dataclasses.replace(
        compiled,
        schema_hash=_automaton.schema_fingerprint(source, **options),
        tokenizer_hash=vocab.tokenizer_hash(),
        compiler_version=compiler_version(),
    )


_CACHE: dict[tuple[str, str, str, str, str], Any] = {}


def compile_toolcall(
    tools: ToolSet,
    *,
    scope: str,
    thought: str,
    whitespace: str = "json",
    call_format: str = "json",
) -> Any:
    """The ``CompiledAutomaton`` of the constrained pass for ``scope`` x ``thought``.

    Cached per ``(tools.version, scope, thought, whitespace, call_format)`` (the version is the sha256
    of the tools file's bytes). ``call_format="json"`` is the T-0080a grammar (payloads unchanged);
    ``"native"`` is the template's ``call:NAME{...}`` format (T-0084a, module docstring), where
    ``whitespace`` does not apply (the native format has none). An unknown scope, thought, whitespace or
    call format raises ``ValueError``.
    """
    if not isinstance(call_format, str) or call_format not in CALL_FORMATS:
        raise ValueError(f"call_format must be one of {CALL_FORMATS}, not {call_format!r}")
    if whitespace not in WHITESPACE_PATTERNS:
        raise ValueError(f"whitespace must be one of {sorted(WHITESPACE_PATTERNS)}")
    key = (tools.version, scope, thought, whitespace, call_format)
    if call_format == "native":
        if key not in _CACHE:
            _CACHE[key] = _compile_native(tools, scope, thought)
        return _CACHE[key]
    regex = _grammar_regex(tools, scope, thought, WHITESPACE_PATTERNS[whitespace])
    if key not in _CACHE:
        _CACHE[key] = _compile(tools.version, regex, scope, thought, whitespace)
    return _CACHE[key]


def report_sizes(tools: ToolSet | None = None, *, whitespace: str = "json") -> list[dict[str, Any]]:
    """Print |S|, the bucket and the payload size of every (call_format, scope, thought); return the rows.

    ``union`` is marked ``path=eval-only`` (D2: never used on the live path); every other row is
    ``path=live`` and is flagged ``OVER-LIVE-GATE-128`` if |S| exceeds ``LIVE_MAX_STATES``. A bucket
    above 512 is flagged ``OVER-512``. Nothing is dropped. ``whitespace`` applies to the JSON rows only.
    """
    from local_astra.decoding import artifact
    from local_astra.toolcall import tools as _tools

    ts = tools if tools is not None else _tools.load()
    print(
        f"T-0080a/T-0084a tool-call grammar sizes: tools {ts.path.name} sha256 {ts.version[:12]}, "
        f"whitespace={whitespace} (json rows), tokenizer {vocab.TOKENIZER_SHA256[:8]}, "
        f"model rev {vocab.MODEL_REVISION[:7]}, {compiler_version()}"
    )
    rows: list[dict[str, Any]] = []
    for call_format in CALL_FORMATS:
        for scope in (NAME_SCOPE, *ts.names, UNION_SCOPE):
            for thought in THOUGHTS:
                a = compile_toolcall(
                    ts, scope=scope, thought=thought, whitespace=whitespace, call_format=call_format
                )
                chars = len(artifact.encode_automaton(a))
                live = scope != UNION_SCOPE
                row: dict[str, Any] = {
                    "call_format": call_format,
                    "scope": scope,
                    "path": "live" if live else "eval-only",
                    "thought": thought,
                    "n_states": int(a.n_states),
                    "bucket": int(a.n_states_bucket),
                    "n_edges": int(a.n_edges),
                    "payload_chars": chars,
                    "is_dfa": bool(a.is_dfa),
                }
                rows.append(row)
                flag = ""
                if live and row["n_states"] > LIVE_MAX_STATES:
                    flag += f"  OVER-LIVE-GATE-{LIVE_MAX_STATES}"
                if row["bucket"] > 512:
                    flag += "  OVER-512"
                print(
                    f"call_format={call_format} scope={scope} path={row['path']} thought={thought} "
                    f"|S|={row['n_states']} bucket={row['bucket']} edges={row['n_edges']} "
                    f"payload_chars={chars} is_dfa={row['is_dfa']}{flag}"
                )
    return rows
