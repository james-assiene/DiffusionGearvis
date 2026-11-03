"""Size report and state gate for compiled automata (T-0063 criteria 6-7).

- ``gate(automaton, max_states)`` raises ``TooLarge`` when ``|S|`` exceeds the limit. The live-path
  default is 128 (docs/07 §0); the evaluation passes 512 (T-0062 decision 5).
- ``compile_report(records, max_states=...)`` compiles every record and returns one row per record,
  in order: a ``CompileRow`` when it passes the gate, a ``TooLargeRecord`` when it does not. No record
  is dropped; the harness counts ``TooLargeRecord`` rows.
- ``sudoku_answer_acceptance(rows, puzzles)`` is criterion 6's named function: for each preserved
  Sudoku emission, whether its answer region is accepted by the compiled Sudoku automaton.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from local_astra.decoding import grammars, vocab
from local_astra.decoding._vendor.dgfa_compile.bfcl_data import BfclRecord
from local_astra.decoding._vendor.dgfa_compile.tasks import answer_region
from local_astra.decoding._vendor.dgfa_compile.tasks.sudoku import SudokuRecord

__all__ = [
    "EVAL_MAX_STATES",
    "LIVE_MAX_STATES",
    "CompileRow",
    "TooLarge",
    "TooLargeRecord",
    "accepts",
    "compile_record",
    "compile_report",
    "gate",
    "sudoku_answer_acceptance",
]

#: docs/07 §0: the tree's per-step cost at |S| > 128 is not affordable on the live path.
LIVE_MAX_STATES = 128
#: T-0062 decision 5: the plugin supports buckets up to 512; the evaluation gates there.
EVAL_MAX_STATES = 512

#: The Phase 0 header ``<|channel>thought\n<channel|>`` (ids 100, 45518, 107, 101).
_PHASE0_HEADER = (100, 45518, 107, 101)
_EOS = 1


class TooLarge(Exception):
    """An automaton has more states than the gate admits."""

    def __init__(self, n_states: int, max_states: int, name: str = "") -> None:
        self.n_states = n_states
        self.max_states = max_states
        self.name = name
        where = f" {name!r}" if name else ""
        super().__init__(f"automaton{where} has |S| = {n_states} > max_states = {max_states}")


def gate(automaton: Any, max_states: int = LIVE_MAX_STATES) -> None:
    """Raise ``TooLarge`` if ``automaton.n_states > max_states``; return normally otherwise."""
    n = int(automaton.n_states)
    if n > max_states:
        raise TooLarge(n, max_states, str(getattr(automaton, "name", "")))


@dataclasses.dataclass(frozen=True)
class CompileRow:
    """One record within the gate. ``payload_bytes`` is filled in by T-0064."""

    id: str
    n_states: int
    bucket: int
    is_dfa: bool
    n_classes: int
    k_max: int
    max_neg_size: int
    build_seconds: float
    payload_bytes: int | None = None


@dataclasses.dataclass(frozen=True)
class TooLargeRecord:
    """One record above the gate: counted by the harness, never dropped."""

    id: str
    n_states: int
    bucket: int
    max_states: int
    build_seconds: float


def compile_record(record: Any) -> Any:
    """Compile a vendored ``SudokuRecord`` or single-function ``BfclRecord``."""
    if isinstance(record, SudokuRecord):
        return grammars.compile_sudoku(record.puzzle)
    if isinstance(record, BfclRecord):
        if len(record.functions) != 1:
            raise ValueError(
                f"{record.id}: {len(record.functions)} functions; only single-function records "
                "compile here"
            )
        return grammars.compile_bfcl(record.functions[0])
    raise TypeError(f"unsupported record type {type(record).__name__}")


def compile_report(
    records: Iterable[Any], *, max_states: int = LIVE_MAX_STATES
) -> list[CompileRow | TooLargeRecord]:
    """One row per record, in order; a record above ``max_states`` yields a ``TooLargeRecord``."""
    rows: list[CompileRow | TooLargeRecord] = []
    for record in records:
        t0 = time.perf_counter()
        a = compile_record(record)
        seconds = time.perf_counter() - t0
        try:
            gate(a, max_states)
        except TooLarge:
            rows.append(
                TooLargeRecord(
                    id=str(record.id),
                    n_states=int(a.n_states),
                    bucket=int(a.n_states_bucket),
                    max_states=max_states,
                    build_seconds=seconds,
                )
            )
            continue
        t = a.tables
        rows.append(
            CompileRow(
                id=str(record.id),
                n_states=int(a.n_states),
                bucket=int(a.n_states_bucket),
                is_dfa=bool(a.is_dfa),
                n_classes=int(t.n_classes),
                k_max=int(t.k_max),
                max_neg_size=int(t.max_neg_size),
                build_seconds=seconds,
            )
        )
    return rows


def sudoku_answer_acceptance(
    rows: Sequence[Mapping[str, Any]],
    puzzles: Mapping[str, Sequence[Sequence[int]]],
) -> dict[str, bool]:
    """Criterion 6: is each emission's answer region accepted by its puzzle's compiled automaton?

    The answer region is diffgemma-fa's ``answer_region`` (header, scratchpad and unscored tail
    removed). It is framed as the model frames an answer -- the Phase 0 header, the region's HF
    tokens, EOS -- and simulated through the automaton. One verdict per row, keyed by ``id``.
    """
    automata: dict[tuple[tuple[int, ...], ...], Any] = {}
    out: dict[str, bool] = {}
    for row in rows:
        rid = str(row["id"])
        grid = tuple(tuple(int(c) for c in r) for r in puzzles[rid])
        if grid not in automata:
            automata[grid] = grammars.compile_sudoku(grid)
        tokens = [*_PHASE0_HEADER, *vocab.encode(answer_region(str(row["text"]))), _EOS]
        out[rid] = accepts(automata[grid], tokens)
    return out


def accepts(automaton: Any, tokens: Sequence[int]) -> bool:
    """Set-of-states simulation over the artifact's arrays (correct for NFAs).

    Class membership is read from the CSR with its polarity, never by materialising a complement
    class (the vendored ``validate.Simulator`` builds a 262 144-element set per negated class).
    """
    t = automaton.tables
    out: dict[int, list[tuple[int, int]]] = {}
    for e in range(int(automaton.n_edges)):
        out.setdefault(int(automaton.edge_src[e]), []).append(
            (int(automaton.edge_dst[e]), int(automaton.edge_class[e]))
        )
    stored: dict[int, frozenset[int]] = {}

    def member(cls: int, token: int) -> bool:
        if cls not in stored:
            lo, hi = int(t.sum_indptr[cls]), int(t.sum_indptr[cls + 1])
            stored[cls] = frozenset(int(x) for x in t.sum_indices[lo:hi])
        return (token in stored[cls]) != bool(t.sum_is_neg[cls])

    cur = {int(i) for i in automaton.start_vector.nonzero()[0]}
    for tok in tokens:
        cur = {dst for s in cur for dst, cls in out.get(s, ()) if member(cls, tok)}
        if not cur:
            return False
    return any(bool(automaton.is_final[s]) for s in cur)
