"""Collection guard for ``tests/cd`` (T-0063 criterion 1).

``tests/cd`` holds the constrained-decoding tests. They need torch (CPU), outlines-core and tokenizers,
which live in the non-default ``cd`` uv dependency group so that main's environment, the fast suite and
its budget do not change (docs/tasks/T-0062.md, human decision 1). The canonical run is::

    make cd-test   # ASTRA_CD_TESTS=1 uv run --group cd pytest tests/cd -q -m "not gpu"

Without ``ASTRA_CD_TESTS=1`` nothing under ``tests/cd`` is collected -- not even imported -- so a plain
``uv run pytest`` (whose ``testpaths`` is ``tests``) never touches a torch import. With it set, the
sentinel rows in ``test_cd_environment.py`` assert that torch, outlines_core and tokenizers import:
running with the variable but without the group **fails**, it does not pass empty.

This file must stay import-light: no torch, no outlines_core, no tokenizers, no ``local_astra.decoding``
at module level. Shared fixtures for the per-task modules live in their own helper modules, not here,
so that concurrent test-authors on the T-0062 tree do not edit one file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CD_TESTS_ENV = "ASTRA_CD_TESTS"
_HERE = Path(__file__).resolve().parent


def cd_tests_enabled() -> bool:
    return os.environ.get(CD_TESTS_ENV) == "1"


def _under_cd(path: Path) -> bool:
    resolved = Path(path).resolve()
    return resolved == _HERE or _HERE in resolved.parents


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    """Refuse every path under ``tests/cd`` unless ``ASTRA_CD_TESTS=1``.

    ``None`` defers to pytest's defaults. Pytest does not consult this hook for a path named
    explicitly on the command line; ``pytest_pycollect_makemodule`` below covers that case.
    """
    if cd_tests_enabled():
        return None
    return True if _under_cd(collection_path) else None


class _NotCollected(pytest.File):
    """Stands in for a ``tests/cd`` module when the guard is closed: never imports it, yields nothing."""

    def collect(self) -> list[pytest.Item]:
        return []


def pytest_pycollect_makemodule(module_path: Path, parent: pytest.Collector) -> pytest.File | None:
    """An explicitly named ``tests/cd`` file is not imported either (no torch import at collection)."""
    if cd_tests_enabled() or not _under_cd(module_path):
        return None
    return _NotCollected.from_parent(parent, path=module_path)
