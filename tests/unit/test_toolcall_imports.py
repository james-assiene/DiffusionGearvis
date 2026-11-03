"""T-0080a criterion 6 (negative): the runtime-safe package stays runtime-safe.

- ``local_astra/toolcall/**`` imports neither ``local_astra.decoding``, torch, numpy nor outlines
  (``outlines``/``outlines_core``): an AST check over every module, with relative imports resolved, plus
  ``importlib.import_module``/``__import__`` calls with a literal name;
- it imports in a fresh interpreter (the fast-suite environment) and loading the v1 tools there pulls in
  none of the forbidden modules transitively;
- ``local_astra/conductor/**``, ``local_astra/contracts/llm.py`` and ``plugin/**`` are unchanged over T-0080a's
  closed range ``505ebc4..8ae5700`` (T-0094: two literal commits, never a moving ref).

The two git/AST self-check rows are green by design today; the package rows are red until the package
exists. The diff row is green today and must stay green. T-0084b extends ``MODULES`` and the expected file set
with ``local_astra.toolcall.native`` (criterion 6), so the two package rows are red until ``native.py`` exists.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests import _git_range as gr
from tests.unit._toolcall_support import REPO, TOOLCALL_DIR

FORBIDDEN = ("local_astra.decoding", "torch", "numpy", "outlines", "outlines_core")
MODULES = (
    "local_astra.toolcall",
    "local_astra.toolcall.tools",
    "local_astra.toolcall.router",
    "local_astra.toolcall.score",
    "local_astra.toolcall.native",  # T-0084b criterion 6: the native parser is behind the same fence
)
UNCHANGED = ("local_astra/conductor", "local_astra/contracts/llm.py", "plugin")


def _is_forbidden(module: str) -> bool:
    return any(module == f or module.startswith(f + ".") for f in FORBIDDEN)


def _package_of(path: Path, root: Path) -> list[str]:
    """The package a module's relative imports resolve against (its directory, for ``__init__`` too)."""
    return list(path.relative_to(root).parent.parts)


def imported_names(path: Path, root: Path) -> Iterator[tuple[int, str]]:
    """Every absolute module name ``path`` imports (relative imports resolved against ``root``)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    pkg = _package_of(path, root)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                mod = ".".join([*base, *([node.module] if node.module else [])])
            else:
                mod = node.module or ""
            yield node.lineno, mod
            for alias in node.names:
                yield node.lineno, f"{mod}.{alias.name}"
        elif isinstance(node, ast.Call):
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fname in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield node.lineno, arg.value


def forbidden_imports(root: Path, package_dir: Path) -> list[str]:
    bad = []
    for path in sorted(package_dir.rglob("*.py")):
        for lineno, name in imported_names(path, root):
            if _is_forbidden(name):
                bad.append(f"{path.relative_to(root)}:{lineno}: {name}")
    return bad


# --------------------------------------------------------------------------- self-check of the fence


def test_the_fence_catches_every_import_form(tmp_path: Path) -> None:
    """The checker is not vacuous: each forbidden form is caught, and a permitted import is not."""
    pkg = tmp_path / "local_astra" / "toolcall"
    pkg.mkdir(parents=True)
    (tmp_path / "local_astra" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("from .score import x\n")
    (pkg / "score.py").write_text(
        textwrap.dedent(
            """
            import json
            import jsonschema
            from pydantic import BaseModel
            import numpy as np
            from torch import nn
            import outlines_core.json_schema
            from ..decoding import grammars
            from .. import decoding
            from local_astra.decoding.toolcall import compile_toolcall
            import importlib
            importlib.import_module("torch")

            def f():
                import outlines
            """
        )
    )
    bad = forbidden_imports(tmp_path, pkg)
    names = {b.split(": ", 1)[1] for b in bad}
    assert {
        "numpy",
        "torch",
        "outlines_core.json_schema",
        "local_astra.decoding",
        "local_astra.decoding.grammars",
        "local_astra.decoding.toolcall",
        "outlines",
    } <= names, bad
    assert not any(n.startswith(("json", "pydantic", "local_astra.toolcall")) for n in names), bad


# --------------------------------------------------------------------------- the package


def test_toolcall_package_imports_nothing_forbidden() -> None:
    files = sorted(TOOLCALL_DIR.rglob("*.py"))
    assert files, f"T-0080a not implemented: {TOOLCALL_DIR.relative_to(REPO)} has no modules"
    expected = {"__init__.py", "tools.py", "router.py", "score.py", "native.py"}  # native: T-0084b
    assert expected <= {p.name for p in files}, sorted(p.name for p in files)
    assert forbidden_imports(REPO, TOOLCALL_DIR) == []


def test_toolcall_imports_cleanly_in_a_fresh_interpreter() -> None:
    code = textwrap.dedent(
        f"""
        import importlib, json, sys
        for m in {list(MODULES)!r}:
            importlib.import_module(m)
        from local_astra.toolcall import tools
        ts = tools.load()
        heavy = sorted(m for m in sys.modules
                       if m.split(".")[0] in ("torch", "numpy", "outlines", "outlines_core")
                       or m == "local_astra.decoding" or m.startswith("local_astra.decoding."))
        print(json.dumps({{"heavy": heavy, "version": ts.version}}))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["heavy"] == [], f"importing local_astra.toolcall loaded: {got['heavy']}"
    assert len(got["version"]) == 64


# --------------------------------------------------------------------------- unchanged files


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


#: T-0080a's merge commit 05bc085 (docs/tasks/T-0080a.md); BASE = ``git merge-base 05bc085^1 05bc085^2``, TIP =
#: ``05bc085^2`` (T-0094 criterion 2, ruling L2: closed to T-0080a's own range, not a permanent fence).
T0080A_MERGE = "05bc085"
T0080A_BASE = "505ebc4"
T0080A_TIP = "8ae5700"


def test_conductor_llm_contract_and_plugin_unchanged_over_t0080a_s_range() -> None:
    """Criterion 6 over T-0080a's closed range (T-0094): nothing under ``UNCHANGED`` differs between its BASE and TIP.
    The untracked-file half of the old guard is retired: it only meant something while the branch was open."""
    gr.prove_range(T0080A_BASE, T0080A_TIP, T0080A_MERGE)
    changed = gr.changed(T0080A_BASE, T0080A_TIP, *UNCHANGED)
    assert changed == [], f"T-0080a criterion 6: changed in {T0080A_BASE}..{T0080A_TIP}: {changed}"


def test_the_t0080a_range_guard_has_power() -> None:
    """T-0094 criterion 4 (adversarial): the same predicate over ``BASE..6a8600b`` sees T-0089/T-0092's
    ``executor.py``; a predicate that reports nothing (or lost ``local_astra/conductor``) fails here."""
    gr.require_commits(T0080A_BASE, gr.LATER)
    assert "local_astra/conductor/executor.py" in gr.changed(T0080A_BASE, gr.LATER, *UNCHANGED)


# --------------------------------------------------------------------------- T-0086 criterion 5


def _imports_of(src: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


#: T-0086's merge commit 3b3a0f1; BASE = ``git merge-base 3b3a0f1^1 3b3a0f1^2``, TIP = ``3b3a0f1^2`` (T-0094b
#: criterion 3: the differential compares T-0086's own two blobs of ``score.py``, never BASE with the live file; the
#: declared base 4c7dfdc differs from db7b480 by a docs commit only).
T0086_MERGE = "3b3a0f1"
T0086_BASE = "db7b480"
T0086_TIP = "41df3a4"
T0086_SCORE = "local_astra/toolcall/score.py"
#: The power pair (T-0094b ruling P3): no revision of ``score.py`` in history has a different import set, so the same
#: predicate runs on T-0084c's b1d4c78, whose ``twopass.py`` added ``local_astra.toolcall.native`` (parent db496bf).
IMPORTS_PRIOR_PARENT = "db496bf"
IMPORTS_PRIOR = "b1d4c78"
IMPORTS_PRIOR_FILE = "local_astra/eval/cd/twopass.py"


def _blob(sha: str, rel: str) -> str:
    proc = _git("show", f"{sha}:{rel}")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _new_imports(base: str, tip: str, rel: str) -> set[str]:
    """T-0086 criterion 5's imports predicate: what ``rel`` imports at ``tip`` that it did not import at ``base``."""
    return _imports_of(_blob(tip, rel)) - _imports_of(_blob(base, rel))


def test_t0086_score_adds_no_imports_and_no_io() -> None:
    """T-0086 criterion 5: ``score.py`` at T-0086's TIP imports nothing it did not import at its BASE (stdlib,
    ``jsonschema`` and ``local_astra.toolcall.tools``) -- T-0086's own two blobs (T-0094b) -- and the current file
    calls no ``open``/``print``/``input`` (no I/O; a property of the file, no git reference)."""
    gr.prove_range(T0086_BASE, T0086_TIP, T0086_MERGE)
    base_rev, tip_rev = T0086_BASE, T0086_TIP
    assert _blob(base_rev, T0086_SCORE) != _blob(tip_rev, T0086_SCORE), (
        "premise: two different blobs"
    )
    new = _new_imports(base_rev, tip_rev, T0086_SCORE)
    assert new == set(), sorted(new)
    src = (TOOLCALL_DIR / "score.py").read_text()
    io_calls = [
        n.func.id
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in ("open", "print", "input")
    ]
    assert io_calls == []


def test_the_t0086_imports_predicate_has_power() -> None:
    """T-0094b criterion 5 (adversarial): the same predicate over b1d4c78's parent and b1d4c78 reports the named import
    ``local_astra.toolcall.native``; a predicate that is always empty fails."""
    gr.require_commits(IMPORTS_PRIOR_PARENT, IMPORTS_PRIOR)
    assert gr.is_ancestor(IMPORTS_PRIOR, T0086_BASE)
    assert _new_imports(IMPORTS_PRIOR_PARENT, IMPORTS_PRIOR, IMPORTS_PRIOR_FILE) == {
        "local_astra.toolcall.native"
    }


def test_the_t0086_imports_row_compares_two_different_blobs() -> None:
    """T-0094b too-strict twin: the pair is T-0086's own change, so the stricter "``score.py`` is byte-identical"
    reports it (T-0086 added ``query_problem``); a pair reduced to TIP vs TIP fails here."""
    gr.prove_range(T0086_BASE, T0086_TIP, T0086_MERGE)
    base, tip = _blob(T0086_BASE, T0086_SCORE), _blob(T0086_TIP, T0086_SCORE)
    assert base != tip
    assert "def query_problem" in tip and "def query_problem" not in base


# --------------------------------------------------------------------------- T-0091 criterion 8 (the AST fence)

#: T-0091's three new modules (docs/tasks/T-0091.md criterion 8): ``live.py`` sits under ``toolcall/`` and is already
#: scanned by ``test_toolcall_package_imports_nothing_forbidden``; ``tool_executor.py`` and ``contracts/toolcall.py``
#: are under no fence until these rows. ``tokenizers`` is criterion 8's addition; ``local_astra.eval`` is the
#: implementation note's ("its parity is proven by tests, not by import (the import fence)"): no twopass in the conductor.
T0091_MODULES = {
    "local_astra/toolcall/live.py": "local_astra.toolcall.live",
    "local_astra/conductor/tool_executor.py": "local_astra.conductor.tool_executor",
    "local_astra/contracts/toolcall.py": "local_astra.contracts.toolcall",
}
T0091_FORBIDDEN = (*FORBIDDEN, "tokenizers", "local_astra.eval")


def _t0091_forbidden(module: str) -> bool:
    return any(module == f or module.startswith(f + ".") for f in T0091_FORBIDDEN)


def test_t0091_the_fence_predicate_is_not_vacuous() -> None:
    """Self-check: the T-0091 predicate catches ``tokenizers``, the eval and the cd stack, and lets the reused
    toolcall modules, the contracts and pydantic through."""
    caught = [
        "tokenizers",
        "local_astra.eval.cd.twopass",
        "local_astra.eval",
        "local_astra.decoding.vocab",
        "numpy",
        "torch.nn",
        "outlines_core",
    ]
    passed = [
        "local_astra.toolcall.native",
        "local_astra.toolcall.special_tokens",
        "local_astra.contracts.cd",
        "local_astra.evaluation",
        "pydantic",
        "jsonschema",
    ]
    assert [m for m in caught if not _t0091_forbidden(m)] == []
    assert [m for m in passed if _t0091_forbidden(m)] == []


@pytest.mark.parametrize("rel", sorted(T0091_MODULES))
def test_t0091_the_new_modules_import_nothing_forbidden(rel: str) -> None:
    """Criterion 8: ``live.py``, ``tool_executor.py`` and ``contracts/toolcall.py`` import nothing from ``decoding``,
    torch, numpy, outlines, tokenizers or the eval (AST, relative imports resolved, literal ``import_module`` too)."""
    path = REPO / rel
    assert path.is_file(), f"T-0091: {rel} does not exist yet"
    bad = [f"{rel}:{ln}: {n}" for ln, n in imported_names(path, REPO) if _t0091_forbidden(n)]
    assert bad == []


def test_t0091_the_new_modules_import_cleanly_in_a_fresh_interpreter() -> None:
    """Criterion 8, transitively: a fresh interpreter importing the three modules has none of the forbidden modules in
    ``sys.modules``."""
    code = textwrap.dedent(
        f"""
        import importlib, json, sys
        for m in {sorted(T0091_MODULES.values())!r}:
            importlib.import_module(m)
        bad = {list(T0091_FORBIDDEN)!r}
        heavy = sorted(m for m in sys.modules if any(m == f or m.startswith(f + ".") for f in bad))
        print(json.dumps(heavy))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=120
    )
    assert out.returncode == 0, f"T-0091: the new modules do not import:\n{out.stderr[-2000:]}"
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


# --------------------------------------------------------------------------- T-0113 criterion 13

#: ``loop.py`` may import the stdlib, ``local_astra.contracts`` and these two toolcall modules, nothing else.
T0113_ALLOWED = ("local_astra.toolcall.special_tokens", "local_astra.toolcall.router")


def _t0113_allowed(name: str) -> bool:
    if name.split(".")[0] in sys.stdlib_module_names:
        return True
    return any(
        name == a or name.startswith(a + ".") for a in ("local_astra.contracts", *T0113_ALLOWED)
    )


def _t0113_violations(path: Path, root: Path) -> list[str]:
    """Every import in ``path`` outside the allowance. ``from P import m`` passes when ``P`` or ``P.m`` is allowed
    (``from local_astra.toolcall import router`` is the router, not the package)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    pkg = _package_of(path, root)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [f"{node.lineno}: {a.name}" for a in node.names if not _t0113_allowed(a.name)]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                mod = ".".join([*base, *([node.module] if node.module else [])])
            else:
                mod = node.module or ""
            for a in node.names:
                if not (_t0113_allowed(mod) or _t0113_allowed(f"{mod}.{a.name}")):
                    bad.append(f"{node.lineno}: {mod}.{a.name}")
        elif isinstance(node, ast.Call):
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fname in ("import_module", "__import__"):
                arg = node.args[0] if node.args else None
                if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                    bad.append(f"{node.lineno}: dynamic {fname}")
                elif not _t0113_allowed(arg.value):
                    bad.append(f"{node.lineno}: {arg.value}")
    return bad


def test_t0113_loop_imports_only_stdlib_contracts_special_tokens_and_router(
    tmp_path: Path,
) -> None:
    """Criterion 13. The predicate is checked first on a synthetic module (not vacuous in either direction: each
    forbidden form is caught, each allowed form passes), then on ``local_astra/toolcall/loop.py``."""
    pkg = tmp_path / "local_astra" / "toolcall"
    pkg.mkdir(parents=True)
    probe = pkg / "probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations
            import json
            from collections.abc import Mapping
            from local_astra.contracts.llm import Prompt
            from local_astra.contracts import toolcall
            from local_astra.toolcall import router
            from .special_tokens import drop_special_literals
            from local_astra.toolcall import tools
            from .native import parse
            import pydantic
            import numpy
            from local_astra.conductor import session
            importlib.import_module("local_astra.toolcall.live")
            """
        )
    )
    caught = {v.split(": ", 1)[1] for v in _t0113_violations(probe, tmp_path)}
    assert caught == {
        "local_astra.toolcall.tools",
        "local_astra.toolcall.native.parse",
        "pydantic",
        "numpy",
        "local_astra.conductor.session",
        "local_astra.toolcall.live",
    }, caught
    loop = TOOLCALL_DIR / "loop.py"
    assert loop.is_file(), "T-0113: local_astra/toolcall/loop.py does not exist yet"
    assert _t0113_violations(loop, REPO) == []
