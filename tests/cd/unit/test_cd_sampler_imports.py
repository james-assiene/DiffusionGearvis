"""T-0067 criterion 10: imports and packaging of the new CD modules.

By AST: ``sampler``, ``requests``, ``record``, ``model_state`` import stdlib, numpy and torch at module scope;
``vllm`` only inside a function or under ``TYPE_CHECKING``; never ``local_astra``. ``make plugin-wheel``'s
build (``uv build --wheel --offline``) contains every ``cd/`` module (the T-0011 criterion 10 mechanism; the
row lives here rather than in ``tests/unit/test_plugin_wheel.py``, whose module-level ``pending(T-0011)``
would turn a T-0067 row into an expected failure).
"""

from __future__ import annotations

import ast
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PLUGIN = REPO / "plugin" / "astra_vllm_plugin"
CD_DIR = PLUGIN / "astra_vllm_plugin" / "cd"
NEW = ("sampler", "requests", "record", "model_state")
ALLOWED_TOP = set(sys.stdlib_module_names) | {"numpy", "torch", "astra_vllm_plugin"}


def _module_scope_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """Imports executed at import time: module body, including if/try blocks, but not
    ``if TYPE_CHECKING:`` bodies and not function/class-method bodies."""
    out: list[tuple[str, int]] = []

    def visit(stmts: list[ast.stmt]) -> None:
        for n in stmts:
            if isinstance(n, ast.Import):
                out.extend((a.name.split(".")[0], n.lineno) for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                if n.level == 0 and n.module:
                    out.append((n.module.split(".")[0], n.lineno))
            elif isinstance(n, ast.If):
                t = n.test
                name = (
                    t.id
                    if isinstance(t, ast.Name)
                    else t.attr
                    if isinstance(t, ast.Attribute)
                    else ""
                )
                if name != "TYPE_CHECKING":
                    visit(n.body)
                visit(n.orelse)
            elif isinstance(n, ast.Try):
                visit(n.body)
                for h in n.handlers:
                    visit(h.body)
                visit(n.orelse)
                visit(n.finalbody)
            elif isinstance(n, ast.ClassDef):
                visit(n.body)

    visit(tree.body)
    return out


def _all_imports(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            roots |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            roots.add(n.module.split(".")[0])
    return roots


@pytest.mark.parametrize("mod", NEW)
def test_module_imports_only_stdlib_numpy_torch(mod: str) -> None:
    path = CD_DIR / f"{mod}.py"
    assert path.exists(), f"T-0067 not implemented: {path.relative_to(REPO)} does not exist"
    tree = ast.parse(path.read_text())
    eager = [(m, ln) for m, ln in _module_scope_imports(tree) if m not in ALLOWED_TOP]
    assert eager == [], f"cd/{mod}.py imports at module scope outside stdlib/numpy/torch: {eager}"
    assert "local_astra" not in _all_imports(tree), (
        f"cd/{mod}.py imports local_astra (absent in the container)"
    )


def test_wheel_contains_every_cd_module(tmp_path: Path) -> None:
    missing_src = [m for m in NEW if not (CD_DIR / f"{m}.py").exists()]
    assert missing_src == [], f"T-0067 not implemented: cd/{missing_src} missing"
    proc = subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(tmp_path)],
        cwd=PLUGIN,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (wheel,) = sorted(tmp_path.glob("astra_vllm_plugin-*.whl"))
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    on_disk = sorted(p.name for p in CD_DIR.glob("*.py"))
    absent = [n for n in on_disk if f"astra_vllm_plugin/cd/{n}" not in names]
    assert absent == [], f"{wheel.name} lacks cd modules {absent}"
