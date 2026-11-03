"""The grammars of the JAX baseline, compiled on the host into token automata (T-0063).

Two grammars, each built with the vendored diffgemma-fa compiler (``_vendor/dgfa_compile``, commit
27c7d0d) over the HF ``tokenizer.json`` vocabulary of the pinned model revision (``vocab.py``):

- **BFCL-Live simple** (``compile_bfcl``): the function's parameter block as a JSON grammar, with the
  options of the accuracy-parity arm ``exp_e5_grammar130_map`` (``BFCL_BASELINE_OPTIONS``): the
  hand-fitted "pretty" separators ``PRETTY_WS``, the optional ````` ```json ````` fence,
  case-insensitive enums, expanded wildcards (``type: any``), non-empty required strings and the
  channel header. ``PRETTY_WS``, ``ALLOW`` and ``case_insensitive_enums`` are ported verbatim from
  diffgemma-fa ``eval/run.py`` at 27c7d0d, which is not part of the vendored ``compile/`` package
  (it imports JAX); see ``_vendor/PROVENANCE.md``.
- **Sudoku 4x4** (``compile_sudoku``): ``sudoku_regex(puzzle, row_separator="\\n")`` exactly as
  diffgemma-fa ``eval/run_tasks.py`` calls it: every given a literal, every blank ``[1-4]``, rows
  joined by a newline, the default 8-token header.

Every automaton is ``HEADER · grammar · STOP · Σ*`` with ``HEADER = 100 · Σ_name{1,8} · 107 · 101``
and the stop set read from the pinned ``generation_config.json`` (``[1, 106, 50]``).

**The header models thinking OFF** -- an empty thought block, ``<|channel>NAME\\n<channel|>``, which
is what the model emits when ``enable_thinking=false``. That is right for the evaluation (T-0070 runs
with thinking off for comparability with the JAX numbers) and **wrong for the conductor**, where
thinking is always on and a thought with content is outside this language. The conductor needs SPEC
§3.6's FREE* thought region, which is deferred (docs/tasks/T-0062.md).

Cache key: ``(schema_hash, tokenizer_hash, compiler_version)``. ``schema_hash`` covers the grammar
source and every option below; ``tokenizer_hash`` is the sha256 of ``tokenizer.json``;
``compiler_version`` names the vendored commit and the outlines-core version.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from importlib import metadata
from typing import Any

from local_astra.decoding import vocab
from local_astra.decoding._vendor.dgfa_compile import automaton as _automaton
from local_astra.decoding._vendor.dgfa_compile import pipeline as _pipeline
from local_astra.decoding._vendor.dgfa_compile import schema as _schema
from local_astra.decoding._vendor.dgfa_compile.tasks.grammars import sudoku_regex

__all__ = [
    "ALLOW",
    "BFCL_BASELINE_OPTIONS",
    "DGFA_COMMIT",
    "HEADER_TOKENS",
    "PRETTY_WS",
    "WHITESPACE_PATTERNS",
    "case_insensitive_enums",
    "compile_bfcl",
    "compile_sudoku",
    "compiler_version",
]

#: The diffgemma-fa commit the compiler was vendored from.
DGFA_COMMIT = "27c7d0d3cc9c9e338d2fe4f6a1e64d3da01c8c03"

#: Name tokens the channel header admits (``pipeline.compile_regex``'s default, and run_tasks.py's
#: value without ``--think``).
HEADER_TOKENS = 8

#: diffgemma-fa ``eval/run.py`` (27c7d0d): schema keywords tolerated although outlines drops them.
ALLOW: tuple[str, ...] = (
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "first_match_wins",
    "additionalProperties",
)

#: diffgemma-fa ``eval/run.py`` (27c7d0d), E4/P1: the model's observed separators (indent-2 pretty
#: printing). Historical and narrow (misses tabs), but it is what the baseline arm ran.
PRETTY_WS = r"( |\n {0,6})?"

#: diffgemma-fa ``eval/run.py`` ``WHITESPACE_PATTERNS``: ``--whitespace`` -> the compiler's pattern.
#: ``None`` means outlines' own ``[ ]?``.
WHITESPACE_PATTERNS: Mapping[str, str | None] = {
    "json": _schema.JSON_WS,
    "stock": None,
    "pretty": PRETTY_WS,
}

#: The grammar options of ``exp_e5_grammar130_map`` (the accuracy-parity arm).
BFCL_BASELINE_OPTIONS: Mapping[str, Any] = {
    "whitespace": "pretty",
    "fence": True,
    "ci_enums": True,
    "expand_wildcard": True,
    "nonempty_strings": True,
    "channel_header": True,
}


def case_insensitive_enums(schema: Any) -> Any:
    """diffgemma-fa ``eval/run.py::_case_insensitive_enums`` (27c7d0d), E4/P4.

    Expands each string enum literal to its variants ``(e, e.lower(), e.upper(), e.capitalize())``,
    first occurrence kept, order preserved. BFCL's scorer lowercases, so every variant scores alike.
    """
    if isinstance(schema, list):
        return [case_insensitive_enums(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k == "enum" and isinstance(v, list):
            seen: set[Any] = set()
            variants: list[Any] = []
            for e in v:
                for cand in (
                    (e, e.lower(), e.upper(), e.capitalize()) if isinstance(e, str) else (e,)
                ):
                    if cand not in seen:
                        seen.add(cand)
                        variants.append(cand)
            out[k] = variants
        else:
            out[k] = case_insensitive_enums(v)
    return out


def compiler_version() -> str:
    """Names everything besides the grammar source and the tokenizer that decides the arrays."""
    return (
        f"dgfa_compile@{DGFA_COMMIT[:7]}+local_astra.T-0063;"
        f"outlines-core=={metadata.version('outlines-core')}"
    )


def _stamp(automaton: Any, source: Mapping[str, Any], options: Mapping[str, Any]) -> Any:
    """Fill in the cache key the vendored ``compile_regex`` leaves empty or partial."""
    return dataclasses.replace(
        automaton,
        schema_hash=_automaton.schema_fingerprint(dict(source), **options),
        tokenizer_hash=vocab.tokenizer_hash(),
        compiler_version=compiler_version(),
    )


def _common_kwargs() -> dict[str, Any]:
    return {
        "vocabulary": vocab.build_vocabulary(),
        "vocab_size": int(vocab.load_tokenizer().get_vocab_size(with_added_tokens=True)),
        "end_tokens": vocab.stop_token_ids(),
        "header_tokens": HEADER_TOKENS,
    }


def compile_bfcl(function: Mapping[str, Any], **overrides: Any) -> Any:
    """Compile one BFCL function's arguments object into a ``CompiledAutomaton``.

    ``overrides`` replaces any key of ``BFCL_BASELINE_OPTIONS``; an unknown key raises ``TypeError``.
    The grammar is the arguments object only (no function name), as in the baseline arm.
    """
    unknown = set(overrides) - set(BFCL_BASELINE_OPTIONS)
    if unknown:
        raise TypeError(f"unknown grammar option(s): {sorted(unknown)}")
    opts = {**BFCL_BASELINE_OPTIONS, **overrides}
    if opts["whitespace"] not in WHITESPACE_PATTERNS:
        raise ValueError(f"whitespace must be one of {sorted(WHITESPACE_PATTERNS)}")
    params = function["parameters"]
    if opts["ci_enums"]:
        params = case_insensitive_enums(params)
    report = _pipeline.compile_json_schema(
        params,
        name=str(function.get("name", "")),
        from_bfcl=True,
        allow=ALLOW,
        allow_wildcard=True,
        expand_wildcard=bool(opts["expand_wildcard"]),
        whitespace_pattern=WHITESPACE_PATTERNS[opts["whitespace"]],
        fence=bool(opts["fence"]),
        nonempty_required_strings=bool(opts["nonempty_strings"]),
        channel_header=bool(opts["channel_header"]),
        **_common_kwargs(),
    )
    source = {"kind": "bfcl", "function": dict(function)}
    return _stamp(report.automaton, source, {**opts, "header_tokens": HEADER_TOKENS})


def compile_sudoku(puzzle: Sequence[Sequence[int]]) -> Any:
    """Compile the Sudoku grammar that pins the givens of ``puzzle`` (0 = blank)."""
    grid = tuple(tuple(int(c) for c in row) for row in puzzle)
    regex = sudoku_regex(grid, row_separator="\n")
    report = _pipeline.compile_regex(regex, name="sudoku", **_common_kwargs())
    source = {"kind": "sudoku", "puzzle": [list(r) for r in grid], "regex": regex}
    return _stamp(
        report.automaton,
        source,
        {"row_separator": "\n", "channel_header": True, "header_tokens": HEADER_TOKENS},
    )
