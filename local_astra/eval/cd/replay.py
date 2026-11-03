"""Constraint satisfaction by host replay, and the answer-region join check (T-0070 criterion 5).

The harness never trusts the plugin's verdict: it compiles its own automaton for the record (the same
compile that produced the payload it sent) and replays the committed ids through it with T-0063's
set-of-states simulator (``report.accepts``). The join check compares ``answer_region`` of the
detokenised committed ids with ``answer_region`` of what was served; ruling E3 rebuilds the served text
as ``<|channel>`` + reasoning + ``<channel|>`` + content, because ``--reasoning-parser gemma4`` moves the
header into ``reasoning`` and ``skip_special_tokens`` drops the delimiters from ``content``.

Compiles go through ``grammars.compile_*`` as module attributes, so a caller-installed cache applies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from local_astra.eval.cd._vendor.tasks import answer_region

__all__ = [
    "CHANNEL_OPEN_ID",
    "STOP_IDS",
    "accepts",
    "compile_automaton",
    "detokenize",
    "join_matches",
    "served_text",
    "terminated",
]

#: ``<|channel>``: the first id of a header-language prefix (hard stop (f)'s header rate).
CHANNEL_OPEN_ID = 100
#: The pinned ``generation_config.json`` ``eos_token_id`` (EOS, ``<turn|>``, ``<|tool_response>``).
STOP_IDS = frozenset({1, 106, 50})
WHITESPACES = ("json", "pretty")


def compile_automaton(task: str, record: Any, *, whitespace: str = "json") -> Any:
    """The record's automaton. BFCL: ``BFCL_BASELINE_OPTIONS`` with ``whitespace`` standard JSON
    (T-0062 decision 11) or, for the parity arm, the baseline's own pretty whitespace."""
    from local_astra.decoding import grammars

    if task == "sudoku":
        return grammars.compile_sudoku(record.puzzle)
    if task != "bfcl":
        raise ValueError(f"unknown task {task!r}")
    if whitespace == "pretty":
        return grammars.compile_bfcl(record.function)  # BFCL_BASELINE_OPTIONS exactly
    if whitespace == "json":
        return grammars.compile_bfcl(record.function, whitespace="json")
    raise ValueError(f"whitespace must be one of {WHITESPACES}")


def accepts(automaton: Any, ids: Sequence[int]) -> bool:
    from local_astra.decoding import report

    return bool(report.accepts(automaton, list(ids)))


def terminated(ids: Sequence[int]) -> bool:
    """Did the committed output reach a stop token?"""
    return any(int(t) in STOP_IDS for t in ids)


def detokenize(ids: Sequence[int]) -> str:
    """The pinned HF tokenizer's decoding, special pieces kept (``<|channel>``, ``<eos>``...)."""
    from local_astra.decoding import vocab

    text: str = vocab.load_tokenizer().decode([int(t) for t in ids], skip_special_tokens=False)
    return text


def served_text(reasoning: str | None, content: str | None) -> str:
    """Ruling E3: the served message with its header rebuilt (``content`` alone when no reasoning)."""
    body = content or ""
    if reasoning is None:
        return body
    return "<|channel>" + reasoning + "<channel|>" + body


def join_matches(committed_text: str, served: str) -> bool:
    return str(answer_region(committed_text)) == str(answer_region(served))
