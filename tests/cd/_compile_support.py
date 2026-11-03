"""Shared oracles and fixtures for the T-0063 rows (the host constraint compiler).

Nothing here imports the code under test at module level: ``need()`` imports it lazily inside a row, so
while ``local_astra.decoding`` does not exist every row fails *individually* with a message naming the
missing module, instead of one collection error hiding which criteria are covered.

The oracles are deliberately independent of the implementation:

- token ids come from the ``tokenizers`` library reading the pinned ``tokenizer.json`` directly;
- acceptance is decided by ``walk()``, a plain set-of-states walk over the artifact's documented arrays
  (``edge_src``/``edge_dst``/``edge_class``, the class CSR ``sum_indptr``/``sum_indices`` with the
  ``sum_is_neg`` polarity, ``start_vector``, ``is_final``; diffgemma-fa ``compile/automaton.py`` and
  ``classes.py`` at 27c7d0d). It never materialises a complement class, so it is independent of the
  vendored ``Simulator`` as well;
- the Sudoku regex and the answer region are re-stated from diffgemma-fa's documented behaviour and
  checked with ``re.fullmatch`` -- a different mechanism from the token automaton.

Compiles are memoised per process (``functools.cache``) because the same automaton is asserted on by
several modules; the determinism row (criterion 9) bypasses the cache on purpose.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.cd import _compile_cache

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "cd"
PRESERVED = FIXTURES / "dgfa_preserved"
BFCL_DIR = FIXTURES / "bfcl"
BFCL_DATA = BFCL_DIR / "BFCL_v4_live_simple.json"
BFCL_ANSWERS = BFCL_DIR / "possible_answer" / "BFCL_v4_live_simple.json"
GENERATION_CONFIG = FIXTURES / "model" / "generation_config.json"

MODEL_ID = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
MODEL_REVISION = "2ea837236295d617ac27f8c17d61228081932c40"  # services/versions.lock
#: sha256 of the pinned snapshot's tokenizer.json, measured 2026-09-29 (tests/fixtures/cd/model/MANIFEST.md).
TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
DGFA_COMMIT = "27c7d0d"

VOCAB_SIZE = 262_144
PAD, EOS, TOOL_RESPONSE, CHANNEL_OPEN, CHANNEL_CLOSE, TURN_OPEN, TURN_CLOSE = (
    0,
    1,
    50,
    100,
    101,
    105,
    106,
)
NEWLINE, THOUGHT = 107, 45518
#: diffgemma-fa Phase 0 (compile/vocab.py, SPEC §3.5 trap 3): EOS, END_OF_TURN, BEGIN_OF_TOOL_RESPONSE.
STOP_IDS = frozenset({EOS, TURN_CLOSE, TOOL_RESPONSE})
#: T-0063 criterion 4.
RESERVED_IDS = frozenset(
    {PAD, EOS, TOOL_RESPONSE, CHANNEL_OPEN, CHANNEL_CLOSE, TURN_OPEN, TURN_CLOSE}
)
#: Phase 0: every generation from the released model opens with exactly these ids (SPEC §3.6).
PHASE0_HEADER = (CHANNEL_OPEN, THOUGHT, NEWLINE, CHANNEL_CLOSE)

#: The bucket ladder of SPEC §5.5 (powers of two); a bucket holds the real states plus one dead state.
LADDER = (16, 32, 64, 128, 256, 512, 1024, 2048)

VENDOR_PKG = "local_astra.decoding._vendor.dgfa_compile"


# --------------------------------------------------------------------------- the code under test


def need(modname: str) -> ModuleType:
    """Import a module of the implementation, failing the row (not the collection) if it is absent.

    Only a missing ``local_astra.decoding*`` module is converted into a named failure; any other import
    error (torch, outlines_core, a bug inside the module) propagates as itself.
    """
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing.startswith("local_astra.decoding") and (
            modname == missing or modname.startswith(missing + ".")
        ):
            pytest.fail(f"T-0063 not implemented: `{modname}` does not exist ({exc})")
        raise


def grammars() -> ModuleType:
    """The implementation's ``grammars`` module, with the test-side compile cache installed
    (``tests/cd/_compile_cache.py``: memo + on-disk cache; the unwrapped functions stay reachable)."""
    mod = need("local_astra.decoding.grammars")
    _compile_cache.install(mod, need(f"{VENDOR_PKG}.automaton"))
    return mod


def report() -> ModuleType:
    grammars()  # compile_report / sudoku_answer_acceptance call through grammars.compile_*
    return need("local_astra.decoding.report")


def fresh_compile(kind: str, arg: Any, **overrides: Any) -> Any:
    """Uncached compile (neither memo nor disk): for the rows that check the cache itself."""
    return _compile_cache.compile_fresh(grammars(), kind, arg, overrides)


def warm_sudoku(cut: str) -> None:
    """Precompile a Sudoku cut into the disk cache in parallel (no-op when warm or disabled)."""
    _compile_cache.precompile(grammars(), "sudoku", [p for _, p, _ in sudoku_cut(cut)])


def warm_bfcl(ids: Sequence[str]) -> None:
    """Precompile ``ids`` together with both BFCL cuts in one pool, so the long ``type: any`` schemas of
    either cut overlap with everything else on a cold cache."""
    wanted = list(dict.fromkeys([*ids, *bfcl_test_cut_ids(), *bfcl_dev_cut_ids()]))
    fns = [bfcl_function(rid) for rid in wanted]
    _compile_cache.precompile(grammars(), "bfcl", fns)


def vocab() -> ModuleType:
    return need("local_astra.decoding.vocab")


def vendored(submodule: str) -> ModuleType:
    return need(f"{VENDOR_PKG}.{submodule}")


# --------------------------------------------------------------------------- tokenizer (the oracle)


def _hf_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.cache
def tokenizer_json_path() -> Path:
    path = (
        _hf_hub_cache()
        / f"models--{MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / MODEL_REVISION
        / "tokenizer.json"
    )
    if not path.is_file():
        pytest.fail(
            f"the pinned tokenizer.json is not in the local HF cache: {path}. The rows replay text "
            f"through {MODEL_ID} rev {MODEL_REVISION}; fetch that snapshot (tokenizer files only)."
        )
    digest = sha256_file(path)
    if digest != TOKENIZER_SHA256:
        pytest.fail(f"{path} has sha256 {digest}, not the pinned {TOKENIZER_SHA256}")
    return path


@functools.cache
def hf_tokenizer() -> Any:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_json_path()))


@functools.cache
def encode(text: str) -> tuple[int, ...]:
    """HF tokenisation of ``text``: special pieces (``<|channel>``, ``<turn|>``...) parsed, no BOS."""
    return tuple(hf_tokenizer().encode(text, add_special_tokens=False).ids)


def replay_ids(text: str) -> list[int]:
    """The replay rule for a preserved emission (tests/fixtures/cd/dgfa_preserved/MANIFEST.md).

    The JAX harness stored ``decode(tokens)`` after stripping trailing PAD; EOS (1) decodes to nothing,
    so it is re-appended. A visible stop already in the text makes that EOS fall in the Σ tail.
    """
    return [*encode(text), EOS]


# --------------------------------------------------------------------------- automaton walker (oracle)


class Walker:
    """Set-of-states simulation over a compiled automaton's arrays; correct for NFAs too."""

    def __init__(self, automaton: Any) -> None:
        import numpy as np

        self.a = automaton
        t = automaton.tables
        self.n_classes = int(t.n_classes)
        self.stored: list[frozenset[int]] = []
        self.neg: list[bool] = []
        for c in range(self.n_classes):
            lo, hi = int(t.sum_indptr[c]), int(t.sum_indptr[c + 1])
            self.stored.append(frozenset(int(x) for x in t.sum_indices[lo:hi]))
            self.neg.append(bool(t.sum_is_neg[c]))
        self.out: dict[int, list[tuple[int, int]]] = {}
        self.edges: list[tuple[int, int, int]] = []
        for e in range(len(automaton.edge_src)):
            src, dst, cls = (
                int(automaton.edge_src[e]),
                int(automaton.edge_dst[e]),
                int(automaton.edge_class[e]),
            )
            self.out.setdefault(src, []).append((dst, cls))
            self.edges.append((src, dst, cls))
        self.start = frozenset(int(i) for i in np.nonzero(np.asarray(automaton.start_vector))[0])
        self.finals = frozenset(int(i) for i in np.nonzero(np.asarray(automaton.is_final))[0])

    def member(self, cls: int, token: int) -> bool:
        return (token in self.stored[cls]) != self.neg[cls]

    def members_among(self, cls: int, tokens: Iterable[int]) -> set[int]:
        return {t for t in tokens if self.member(cls, t)}

    def class_size(self, cls: int) -> int:
        n = len(self.stored[cls])
        return VOCAB_SIZE - n if self.neg[cls] else n

    def run(self, tokens: Sequence[int]) -> frozenset[int]:
        cur = self.start
        for tok in tokens:
            nxt = {dst for s in cur for dst, cls in self.out.get(s, ()) if self.member(cls, tok)}
            cur = frozenset(nxt)
            if not cur:
                break
        return cur

    def accepts(self, tokens: Sequence[int]) -> bool:
        return bool(self.run(tokens) & self.finals)

    def acc(self) -> int:
        """ACC: after stop augmentation it is the only accepting state (automaton.py docstring)."""
        assert len(self.finals) == 1, (
            f"expected exactly one accepting state (ACC), got {sorted(self.finals)}"
        )
        return next(iter(self.finals))

    def body_start(self) -> int:
        """The state entered by the header's ``<channel|>`` (101) edge: where the grammar region starts."""
        acc = self.acc()
        hits = [(s, d) for s, d, c in self.edges if s != acc and self.member(c, CHANNEL_CLOSE)]
        assert len(hits) == 1, f"expected exactly one <channel|> edge outside ACC, got {hits}"
        return hits[0][1]

    def body_states(self) -> set[int]:
        """States reachable from ``body_start`` without entering ACC (grammar + optional fence)."""
        acc = self.acc()
        seen = {self.body_start()}
        frontier = list(seen)
        while frontier:
            s = frontier.pop()
            for dst, _ in self.out.get(s, ()):
                if dst != acc and dst not in seen:
                    seen.add(dst)
                    frontier.append(dst)
        return seen


_WALKERS: dict[int, tuple[Any, Walker]] = {}


def walker(automaton: Any) -> Walker:
    hit = _WALKERS.get(id(automaton))
    if hit is not None and hit[0] is automaton:
        return hit[1]
    w = Walker(automaton)
    _WALKERS[id(automaton)] = (automaton, w)
    return w


# --------------------------------------------------------------------------- fixtures


@functools.cache
def preserved(name: str) -> Mapping[str, Any]:
    path = PRESERVED / f"{name}.json"
    data: Mapping[str, Any] = json.loads(path.read_text())
    return data


@functools.cache
def sudoku_cut(
    cut: str,
) -> tuple[tuple[str, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]], ...]:
    """``(id, puzzle, solution)`` for the ``test`` (250, seed 0) or ``dev`` (100, seed 1) cut."""
    data = json.loads((PRESERVED / "sudoku_puzzles.json").read_text())
    return tuple(
        (
            r["id"],
            tuple(tuple(row) for row in r["puzzle"]),
            tuple(tuple(row) for row in r["solution"]),
        )
        for r in data["cuts"][cut]["records"]
    )


def bfcl_missing_message() -> str:
    return (
        f"BFCL fixture missing: {BFCL_DATA.relative_to(REPO)} (and "
        f"{BFCL_ANSWERS.relative_to(REPO)}). Human action (docs/tasks/T-0062.md, decision 3): the "
        "human approves the gorilla commit and the files land in tests/fixtures/cd/bfcl/ with their "
        "MANIFEST.md. Criteria 5 and 7 cannot be proved for BFCL until then; do not substitute another "
        "split."
    )


@functools.cache
def bfcl_records() -> tuple[Mapping[str, Any], ...]:
    """Every live_simple record, in file order (JSON Lines). Fails loudly if the fixture is absent."""
    if not BFCL_DATA.is_file():
        pytest.fail(bfcl_missing_message())
    rows = []
    for line in BFCL_DATA.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return tuple(rows)


@functools.cache
def bfcl_by_id() -> Mapping[str, Mapping[str, Any]]:
    return {r["id"]: r for r in bfcl_records()}


def bfcl_function(record_id: str) -> Mapping[str, Any]:
    rec = bfcl_by_id().get(record_id)
    if rec is None:
        pytest.fail(f"{record_id} is not in {BFCL_DATA.relative_to(REPO)}")
    fns = rec["function"]
    assert len(fns) == 1, f"{record_id}: live_simple records carry one function, got {len(fns)}"
    fn: Mapping[str, Any] = fns[0]
    return fn


def bfcl_test_cut_ids() -> list[str]:
    """T-0070 criterion 1: exactly the 130 ids of ``exp_e5_grammar130_map``, in its order."""
    return [r["id"] for r in preserved("exp_e5_grammar130_map")["rows"]]


def bfcl_dev_cut_ids() -> list[str]:
    """live_simple records 130-229 (n = 100) among the single-function records, file order."""
    single = [r["id"] for r in bfcl_records() if len(r.get("function") or ()) == 1]
    return single[130:230]


# --------------------------------------------------------------------------- memoised compiles


@functools.cache
def compiled_bfcl(record_id: str) -> Any:
    return grammars().compile_bfcl(bfcl_function(record_id))


@functools.cache
def compiled_sudoku(puzzle: tuple[tuple[int, ...], ...]) -> Any:
    return grammars().compile_sudoku(puzzle)


@functools.cache
def _compiled_function(declared_json: str) -> Any:
    return grammars().compile_bfcl(json.loads(declared_json))


def compiled_function(function: Mapping[str, Any]) -> Any:
    """Compile a hand-written BFCL function dict with the baseline options (memoised).

    The memo key is ``json.dumps`` **without** ``sort_keys``: the key order of ``properties`` is part of
    the grammar (diffgemma-fa ``compile_json_schema`` builds an ordered grammar and does not admit
    reordered keys), so the dict that is compiled must keep the declared order, and two dicts that
    differ only in key order are different grammars and must not share a memo entry.
    """
    return _compiled_function(json.dumps(function))


# --------------------------------------------------------------------------- Sudoku oracles


def sudoku_regex_oracle(puzzle: Sequence[Sequence[int]]) -> str:
    """diffgemma-fa ``sudoku_regex(puzzle, row_separator="\\n")`` as ``eval/run_tasks.py`` calls it:
    each given is a literal, each blank ``[1-n]``, cells unseparated, rows joined by a newline."""
    n = len(puzzle)
    digits = "".join(str(d) for d in range(1, n + 1))
    rows = ["".join(str(c) if c else f"[{digits}]" for c in row) for row in puzzle]
    return "\n".join(rows)


_TAIL_MARKER = re.compile(r"<\|(?:channel|turn|tool_response)>|<(?:channel|turn)\|>|<eos>|<pad>")


def answer_region_oracle(text: str) -> str:
    """diffgemma-fa ``compile/tasks/__init__.py::answer_region`` (27c7d0d), re-stated: drop the header up
    to ``<channel|>``, the ``ANSWER:`` scratchpad, and everything from the first sentinel onwards (a
    sentinel at position 0 is an opener and is skipped); strip whitespace."""
    body = text.split("<channel|>", 1)[1] if "<channel|>" in text else text
    if "ANSWER:" in body:
        body = body.split("ANSWER:", 1)[1]
    while True:
        body = body.lstrip()
        m = _TAIL_MARKER.search(body)
        if m is None:
            break
        if m.start() == 0:
            body = body[m.end() :]
            continue
        body = body[: m.start()]
        break
    return body.strip()


def field(row: Any, name: str) -> Any:
    """Read a report field from a mapping or an attribute, without prescribing which."""
    if isinstance(row, Mapping):
        if name not in row:
            raise AssertionError(f"report row has no field {name!r}: {sorted(row)}")
        return row[name]
    if not hasattr(row, name):
        raise AssertionError(f"report row {type(row).__name__} has no field {name!r}")
    return getattr(row, name)


def expected_bucket(n_states: int) -> int:
    """Smallest ladder entry holding the real states plus the absorbing dead state (SPEC §5.5)."""
    for b in LADDER:
        if n_states + 1 <= b:
            return b
    b = LADDER[-1]
    while b < n_states + 1:
        b *= 2
    return b
