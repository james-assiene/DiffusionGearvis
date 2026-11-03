"""T-0069 criterion 3's fences, the "unedited" half, on CPU (moved from ``test_vllm_cd_wiring_gpu.py`` by T-0094b).

The modules (and their probes) that run as criterion 3's fences in the same launch are unedited by T-0069: nothing
under them differs over T-0069's closed range ``BASE..TIP``, two literal commits from its merge commit f3e3c43
(T-0094b criteria 2 and 9). The old row compared e760cb7 with the working tree inside a gpu module, so it was red
(T-0094's e2fefde legitimately edited ``test_vllm_tap_fields_gpu.py``) but invisible on CPU; its uncommitted-changes
half is retired (it only meant something while the branch was open).

No torch or vLLM import: only the path list is copied from the gpu module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests import _git_range as gr

#: criterion 3's fences, "passing unedited": the modules (and their probes) run in the same launch
FENCED_UNEDITED = (
    "tests/contract/test_vllm_shim_gpu.py",
    "tests/container_scripts/astra_shim_probe.py",
    "tests/contract/test_vllm_tap_fields_gpu.py",
    "tests/container_scripts/astra_tap_probe.py",
    "tests/contract/test_vllm_cd_coupling_gpu.py",
    "tests/container_scripts/astra_cd_probe.py",
    "tests/container_scripts/astra_cd_tier2_runner.py",
    "tests/pytest_summary.py",
    "tests/vllm_container.py",
)
#: T-0069's merge commit f3e3c43; BASE = ``git merge-base f3e3c43^1 f3e3c43^2``, TIP = ``f3e3c43^2`` (T-0094b
#: criterion 2; the gpu module's old ``BASE`` e760cb7 is this merge-base).
T0069_MERGE = "f3e3c43"
T0069_BASE = "e760cb7"
T0069_TIP = "154614d"


@pytest.mark.parametrize("path", FENCED_UNEDITED)
def test_the_fence_modules_are_unedited(path: str) -> None:
    """Criterion 3: ``path`` does not differ between T-0069's BASE and TIP (range proven against ``gr.LATER_B``,
    T-0094b criterion 4)."""
    gr.prove_range(T0069_BASE, T0069_TIP, T0069_MERGE, later=gr.LATER_B)
    changed = gr.changed(T0069_BASE, T0069_TIP, path)
    assert changed == [], (
        f"{path} changed in {T0069_BASE}..{T0069_TIP}: criterion 3's fences must pass unedited"
    )


def test_the_fence_guard_has_power() -> None:
    """T-0094b criterion 5 (adversarial): over ``BASE..2c584d9`` the predicate over ``FENCED_UNEDITED`` reports
    T-0094's edit of the tap-fields gpu module, reached through the tuple; a predicate reporting nothing fails."""
    gr.require_commits(T0069_BASE, gr.LATER_B)
    tap_fields = next(p for p in FENCED_UNEDITED if p.endswith("_tap_fields_gpu.py"))
    assert tap_fields in gr.changed(T0069_BASE, gr.LATER_B, *FENCED_UNEDITED)


def test_the_scope_module_imports_neither_torch_nor_vllm() -> None:
    """T-0094b criterion 9: the moved rows run on CPU without the gpu stack."""
    tree = ast.parse(Path(__file__).read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {m for m in names if m.split(".")[0] in ("torch", "vllm")}, names
    assert "tests.vllm_container" not in names, names
