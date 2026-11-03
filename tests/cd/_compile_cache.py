"""Test-side compile cache for the T-0063 rows: per-process memo + on-disk cache + parallel precompile.

The compiler is deterministic (criterion 9 is asserted uncached, in fresh processes), so rows may share
one compile per record. ``install()`` replaces ``compile_bfcl`` / ``compile_sudoku`` on the
implementation's ``grammars`` module with memoising wrappers; ``report`` calls through the module
attribute, so ``compile_report`` and ``sudoku_answer_acceptance`` share the memo too.

On-disk cache: ``$(git rev-parse --git-common-dir)/local_astra/cd_compile_cache/``, one ``.npz`` per
automaton, written with the vendored ``automaton.save`` and read with ``automaton.load`` (which re-checks
the structural invariants). The key hashes the kind, the payload **in declared key order**, every
override, ``BFCL_BASELINE_OPTIONS``, the tokenizer sha256, ``grammars.compiler_version()`` and the bytes
of ``local_astra/decoding/**/*.py``, so any change to the compiler misses. ``ASTRA_CD_COMPILE_CACHE=0``
disables the disk cache (the per-process memo stays).

Uncached on purpose: ``ORIGINALS`` (the unwrapped functions) serve the rows that prove a cached load
equals a fresh compile, the build-seconds timing row, and the criterion-9 determinism subprocesses.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

CACHE_ENV = "ASTRA_CD_COMPILE_CACHE"
_REPO = Path(__file__).resolve().parents[2]
_TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
_POOL_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))
#: Address-space cap per pool worker. Measured 2026-09-29: one BFCL-Live ``type: any`` schema peaks at
#: ~21 GB RSS, and eight such workers OOM-killed on this shared 60 GB box. A worker that hits the cap
#: gives up; that record then compiles in the test process, serially, when a row asks for it.
_WORKER_AS_BYTES = 6 << 30

ORIGINALS: dict[str, Callable[..., Any]] = {}
_MEMO: dict[str, Any] = {}
STATS = {"memo": 0, "disk": 0, "compiled": 0}
_IMPL_DIGEST: list[str] = []
#: Files under ``local_astra/decoding`` that do not decide a compiled automaton's arrays, so they stay out
#: of the digest: ``artifact.py`` (T-0064) only serialises a finished automaton for the wire. Everything
#: else, including any new module, still invalidates the cache.
_NOT_COMPILER = frozenset({"artifact.py"})


def disk_enabled() -> bool:
    return os.environ.get(CACHE_ENV, "1") != "0"


def cache_dir() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    common = Path(out) if Path(out).is_absolute() else (_REPO / out)
    d = common.resolve() / "local_astra" / "cd_compile_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _implementation_digest() -> str:
    if not _IMPL_DIGEST:
        h = hashlib.sha256()
        root = _REPO / "local_astra" / "decoding"
        for p in sorted(root.rglob("*.py")):
            if p.relative_to(root).as_posix() in _NOT_COMPILER:
                continue
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
        _IMPL_DIGEST.append(h.hexdigest())
    return _IMPL_DIGEST[0]


def payload(kind: str, arg: Any, overrides: Mapping[str, Any]) -> str:
    if kind == "sudoku":
        arg = [[int(c) for c in row] for row in arg]
    # No sort_keys on the payload: the declared property order is part of the grammar.
    return json.dumps([kind, arg, sorted(overrides.items())])


def disk_key(grammars: ModuleType, payload_json: str) -> str:
    parts = [
        payload_json,
        json.dumps(dict(grammars.BFCL_BASELINE_OPTIONS), sort_keys=True),
        _TOKENIZER_SHA256,
        str(grammars.compiler_version()),
        _implementation_digest(),
    ]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def compile_fresh(grammars: ModuleType, kind: str, arg: Any, overrides: Mapping[str, Any]) -> Any:
    """An uncached compile through the implementation's own function (never the memo or the disk)."""
    fn = ORIGINALS.get(f"compile_{kind}") or getattr(grammars, f"compile_{kind}")
    if kind == "sudoku":
        return fn(tuple(tuple(int(c) for c in row) for row in arg))
    return fn(arg, **overrides)


def _save(automaton_mod: ModuleType, a: Any, path: Path) -> None:
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    automaton_mod.save(a, str(tmp))
    os.replace(tmp, path)


def _get(
    grammars: ModuleType,
    automaton_mod: ModuleType,
    kind: str,
    arg: Any,
    overrides: Mapping[str, Any],
) -> Any:
    key = payload(kind, arg, overrides)
    hit = _MEMO.get(key)
    if hit is not None:
        STATS["memo"] += 1
        return hit
    path = cache_dir() / f"{disk_key(grammars, key)}.npz" if disk_enabled() else None
    if path is not None and path.is_file():
        a = automaton_mod.load(str(path))
        STATS["disk"] += 1
    else:
        a = compile_fresh(grammars, kind, arg, overrides)
        STATS["compiled"] += 1
        if path is not None:
            _save(automaton_mod, a, path)
    _MEMO[key] = a
    return a


def install(grammars: ModuleType, automaton_mod: ModuleType) -> None:
    """Wrap the module's compile functions once per process (idempotent)."""
    if ORIGINALS:
        return
    ORIGINALS["compile_bfcl"] = grammars.compile_bfcl
    ORIGINALS["compile_sudoku"] = grammars.compile_sudoku

    def compile_bfcl(function: Mapping[str, Any], **overrides: Any) -> Any:
        return _get(grammars, automaton_mod, "bfcl", function, overrides)

    def compile_sudoku(puzzle: Sequence[Sequence[int]]) -> Any:
        return _get(grammars, automaton_mod, "sudoku", puzzle, {})

    grammars.compile_bfcl = compile_bfcl
    grammars.compile_sudoku = compile_sudoku


def _limit_memory() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (_WORKER_AS_BYTES, _WORKER_AS_BYTES))


def _worker(job: tuple[str, str, str]) -> str | None:
    """Pool worker (fresh interpreter, memory-capped): compile uncached and write the .npz."""
    import importlib

    kind, arg_json, path = job
    try:
        grammars = importlib.import_module("local_astra.decoding.grammars")
        automaton_mod = importlib.import_module(
            "local_astra.decoding._vendor.dgfa_compile.automaton"
        )
        a = compile_fresh(grammars, kind, json.loads(arg_json), {})
        _save(automaton_mod, a, Path(path))
    except MemoryError:
        return None
    return path


def precompile(grammars: ModuleType, kind: str, args: Iterable[Any]) -> None:
    """Fill the disk cache for ``args`` in a process pool (<= 8 workers); no-op with the cache off."""
    if not disk_enabled():
        return
    jobs = []
    for arg in args:
        key = payload(kind, arg, {})
        if key in _MEMO:
            continue
        path = cache_dir() / f"{disk_key(grammars, key)}.npz"
        if not path.is_file():
            jobs.append((kind, json.dumps(json.loads(key)[1]), str(path)))
    if len(jobs) < 2:
        return
    ctx = multiprocessing.get_context("spawn")
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=_POOL_WORKERS, mp_context=ctx, initializer=_limit_memory
        ) as pool:
            list(pool.map(_worker, jobs, chunksize=1))
    except concurrent.futures.process.BrokenProcessPool:
        # Precompilation is an optimisation only: whatever did not reach the disk compiles in-process.
        pass
