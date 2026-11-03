"""Sentinel rows for the ``cd`` test environment (T-0063 criterion 1).

Collected only with ``ASTRA_CD_TESTS=1`` (see ``conftest.py``). If the ``cd`` dependency group is not
installed these rows fail -- the suite must never pass empty because its dependencies are missing.
"""

from __future__ import annotations

import importlib

import pytest

#: triton: T-0077b (the host runs the kernels under TRITON_INTERPRET=1)
CD_GROUP_MODULES = ("torch", "outlines_core", "tokenizers", "triton")


@pytest.mark.parametrize("module", CD_GROUP_MODULES)
def test_cd_group_module_imports(module: str) -> None:
    try:
        importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - the failure path is the point
        pytest.fail(
            f"{module!r} is not importable ({exc}). tests/cd needs the `cd` uv dependency group: run "
            "`make cd-test` (ASTRA_CD_TESTS=1 uv run --group cd pytest tests/cd -q -m 'not gpu'). "
            "If the group exists in pyproject.toml but not in uv.lock, the human regenerates uv.lock "
            "(docs/tasks/T-0062.md, human decision 1)."
        )
