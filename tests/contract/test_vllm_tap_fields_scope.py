"""T-0011 criterion 12 (negative), the git half, on CPU (moved here from ``test_vllm_tap_fields_gpu.py`` by T-0094).

``services/vllm.sh``, the shim test, the shim probe and ``plugin/.../cd/**`` are untouched by T-0011: nothing under
them differs over T-0011's closed range ``BASE..TIP``, two literal commits from its merge commit 4837374 (T-0094
criterion 2). The old row compared ``merge-base(HEAD, main)`` with the working tree, so on main it could never fail;
its uncommitted-changes half is retired (it only meant something while the branch was open).

No torch or vLLM import: only ``REPO``, ``CD_DIR`` and the path list are copied from the gpu module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
CD_DIR = "plugin/astra_vllm_plugin/astra_vllm_plugin/cd"
FENCED = (
    "services/vllm.sh",
    "tests/contract/test_vllm_shim_gpu.py",
    "tests/container_scripts/astra_shim_probe.py",
    CD_DIR,
)
#: T-0011's merge commit 4837374; BASE = ``git merge-base 4837374^1 4837374^2``, TIP = ``4837374^2`` (T-0094
#: criterion 2; T-0011 merged main into its branch, so its declared fork point is not the base).
T0011_MERGE = "4837374"
T0011_BASE = "f21c682"
T0011_TIP = "0bd3346"


@pytest.mark.parametrize("path", FENCED)
def test_fenced_paths_are_untouched_over_t0011_s_range(path: str) -> None:
    """Criterion 12, negative: nothing under ``path`` differs between T-0011's BASE and TIP (T-0094 criterion 3 first:
    the range is the merge's real history; skip if a commit is absent)."""
    gr.prove_range(T0011_BASE, T0011_TIP, T0011_MERGE)
    changed = gr.changed(T0011_BASE, T0011_TIP, path)
    assert changed == [], f"{path} changed in {T0011_BASE}..{T0011_TIP}: {changed}"


def test_the_t0011_range_guard_has_power() -> None:
    """T-0094 criterion 4 (adversarial): over ``BASE..6a8600b`` the predicate over ``FENCED`` sees ``cd/automaton.py``
    (T-0081 and later); a predicate that reports nothing fails here."""
    gr.require_commits(T0011_BASE, gr.LATER)
    assert f"{CD_DIR}/automaton.py" in gr.changed(T0011_BASE, gr.LATER, *FENCED)


def test_the_scope_module_imports_neither_torch_nor_vllm() -> None:
    """T-0094 implementation note: the moved rows run on CPU without the gpu stack."""
    tree = ast.parse(Path(__file__).read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {m for m in names if m.split(".")[0] in ("torch", "vllm")}, names
    assert "tests.vllm_container" not in names, names
