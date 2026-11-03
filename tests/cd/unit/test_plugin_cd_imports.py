"""T-0064 criterion 7 (negative): the plugin's wire modules import only the standard library and numpy.

The vLLM container has no ``local_astra`` and the decoder runs in ``add_request`` on the engine thread, so
``cd/automaton.py``, ``cd/xargs.py`` and ``cd/record_schema.py`` must not import ``torch``,
``local_astra``, ``vllm`` or anything else outside stdlib + numpy. Derived by AST (every ``import``,
``from ... import`` and constant-argument ``import_module``/``__import__``), plus one subprocess that
imports the three modules and inspects ``sys.modules``, which also catches a heavy ``cd/__init__.py``.
``local_astra/contracts/llm.py`` is unchanged over T-0064's closed range ``BASE..TIP`` (T-0094b; the old
working-tree comparison and its ``EXCUSED`` list for later tasks are retired).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap

import pytest

from tests import _git_range as gr
from tests.cd._wire_support import CD_DIR, PLUGIN_SRC

WIRE_MODULES = ("automaton", "xargs", "record_schema")
ALLOWED_TOP = frozenset(sys.stdlib_module_names) | {"numpy", "__future__"}


def _source(name: str) -> str:
    path = CD_DIR / f"{name}.py"
    if not path.is_file():
        pytest.fail(f"T-0064 not implemented: {path} is missing")
    return path.read_text()


def _imports(tree: ast.AST) -> list[str]:
    """Every imported module name; relative imports as ``.name`` / ``..name`` (level dots)."""
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = "." * node.level + (node.module or "")
                if node.module:
                    out.append(base)
                else:
                    out += [base + a.name for a in node.names]
            else:
                out.append(node.module or "")
        elif isinstance(node, ast.Call):
            fn = node.func
            fname = getattr(fn, "attr", getattr(fn, "id", ""))
            if fname in ("import_module", "__import__"):
                for a in node.args[:1]:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        out.append(a.value)
                    else:
                        out.append("<dynamic import>")
    return out


def _allowed(mod: str) -> bool:
    if mod.startswith(".."):
        return False  # reaches the plugin package or above
    if mod.startswith("."):
        return mod[1:].split(".")[0] in WIRE_MODULES
    return mod.split(".")[0] in ALLOWED_TOP


@pytest.mark.parametrize("name", WIRE_MODULES)
def test_wire_module_imports_only_stdlib_and_numpy(name):
    mods = _imports(ast.parse(_source(name)))
    bad = sorted({m for m in mods if not _allowed(m)})
    assert not bad, f"cd/{name}.py imports outside stdlib + numpy + the wire modules: {bad}"


def test_the_ast_check_sees_a_torch_import():
    """Control: the walker is live on the spellings that matter."""
    src = textwrap.dedent(
        """
        import numpy as np
        from torch import nn
        import local_astra.contracts.cd
        from ..sampler import X
        from .emission import build_tables
        importlib.import_module("vllm")
        """
    )
    bad = sorted({m for m in _imports(ast.parse(src)) if not _allowed(m)})
    assert bad == sorted([".emission", "..sampler", "local_astra.contracts.cd", "torch", "vllm"])


def test_importing_the_wire_modules_pulls_no_torch_or_local_astra():
    for name in WIRE_MODULES:
        _source(name)
    code = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(PLUGIN_SRC)!r})
        import astra_vllm_plugin.cd.automaton, astra_vllm_plugin.cd.xargs
        import astra_vllm_plugin.cd.record_schema
        heavy = sorted(m for m in sys.modules
                       if m.split(".")[0] in ("torch", "local_astra", "vllm", "outlines_core",
                                              "tokenizers", "pydantic"))
        print(json.dumps(heavy))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd="/", timeout=120
    )
    assert out.returncode == 0, out.stderr
    heavy = json.loads(out.stdout.strip().splitlines()[-1])
    assert heavy == [], f"importing the wire modules loaded: {heavy}"


#: The paths T-0064 criterion 7 holds byte-identical.
GUARDED_PATHS = ("local_astra/contracts/llm.py",)
#: T-0064's merge commit b170f45; BASE = ``git merge-base b170f45^1 b170f45^2`` = af5f9e9 (not 9669258, the T-0088
#: base the old row used, nor T-0064's declared fork point), TIP = ``b170f45^2`` (T-0094b criterion 2). Over this
#: closed range no later-task excuse is needed: the old ``EXCUSED`` entry (T-0088 criterion 1) covered the guard's only
#: path, so the old row could never fail.
T0064_MERGE = "b170f45"
T0064_BASE = "af5f9e9"
T0064_TIP = "28ba6ed"


def test_llm_contract_is_unchanged_from_the_merge_base():
    """T-0064 criterion 7: nothing under ``GUARDED_PATHS`` differs between T-0064's BASE and TIP."""
    gr.prove_range(T0064_BASE, T0064_TIP, T0064_MERGE)
    changed = gr.changed(T0064_BASE, T0064_TIP, *GUARDED_PATHS)
    assert changed == [], (
        f"T-0064 criterion 7: guarded paths changed in {T0064_BASE}..{T0064_TIP}: {changed}"
    )


def test_the_llm_contract_guard_has_power():
    """T-0094b criterion 5 (adversarial): over ``BASE..6a8600b`` the predicate reports the guarded contract (T-0088 and
    later edited it), reached through the tuple; a predicate that reports nothing fails here."""
    gr.require_commits(T0064_BASE, gr.LATER)
    assert gr.changed(T0064_BASE, gr.LATER, *GUARDED_PATHS) == list(GUARDED_PATHS)
