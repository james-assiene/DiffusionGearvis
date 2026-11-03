"""Collection gate for ``tests/stock`` (T-0011 tier 1b). The gate only -- keep this file import-light.

``tests/stock`` runs the canvas tap around the *real* vendored v0.29.0 ``DiffusionSampler`` on CPU torch
(criterion 15) and, inside the pinned container, around the image's compiled sampler on CUDA (criterion 16).
torch lives in the non-default ``stock`` uv dependency group, so the fast suite never imports it. The
canonical run is::

    ASTRA_STOCK_TESTS=1 uv run --group stock pytest tests/stock -q -m "not gpu"

Without ``ASTRA_STOCK_TESTS=1`` nothing under ``tests/stock`` is collected -- not even imported. With it set
but without the group, ``tests/stock/world.py`` fails to import torch and the run **fails** with a
collection error naming the group; it never passes empty or skips.

No torch, no plugin, no fixtures here: shared helpers live in ``tests/stock/world.py`` because the
container runs pytest with ``--noconftest`` (Regime, tier 2), where this file is not loaded at all.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

STOCK_TESTS_ENV = "ASTRA_STOCK_TESTS"
_HERE = Path(__file__).resolve().parent


def stock_tests_enabled() -> bool:
    return os.environ.get(STOCK_TESTS_ENV) == "1"


def _under_stock(path: Path) -> bool:
    resolved = Path(path).resolve()
    return resolved == _HERE or _HERE in resolved.parents


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    """Refuse every path under ``tests/stock`` unless ``ASTRA_STOCK_TESTS=1`` (``None`` defers)."""
    if stock_tests_enabled():
        return None
    return True if _under_stock(collection_path) else None


class _NotCollected(pytest.File):
    """Stands in for a ``tests/stock`` module when the gate is closed: never imports it."""

    def collect(self) -> list[pytest.Item]:
        return []


def pytest_pycollect_makemodule(module_path: Path, parent: pytest.Collector) -> pytest.File | None:
    """An explicitly named ``tests/stock`` file is not imported either while the gate is closed."""
    if stock_tests_enabled() or not _under_stock(module_path):
        return None
    return _NotCollected.from_parent(parent, path=module_path)
