"""Contract tests run against the fake by default and the real service under ``-m gpu``.

The ``backend`` fixture is the parametrisation point required by ``.claude/rules/tests.md``:
``fake`` builds the in-memory fake; ``real`` (marked ``gpu``) builds the real client against
the loopback service and is skipped unless the service answers its health check.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.gpu)])
def backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(autouse=True, scope="session")
def brave_key_nowhere(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """T-0163 criterion 10 (the same fixture as ``tests/unit/conftest.py``'s): the Brave Search API key loader's
    default path points at a nonexistent path under a session tmp dir, so no contract row reads the real key file.
    The opt-in live row (``test_brave_api_live.py``) reaches the real file only through the production loader in a
    child process, which does not load this conftest."""
    import importlib
    from types import SimpleNamespace

    try:
        brave_key = importlib.import_module("local_astra.search.brave_key")
    except ModuleNotFoundError as exc:
        if exc.name != "local_astra.search.brave_key":
            raise
        yield SimpleNamespace(path=None, original=None)
        return
    mp, ns = install_brave_key_nowhere(brave_key, lambda: tmp_path_factory.mktemp("t0163_no_key"))
    yield ns
    mp.undo()


def install_brave_key_nowhere(brave_key: Any, make_dir: Any) -> tuple[pytest.MonkeyPatch, Any]:
    """The same function as ``tests/unit/conftest.py``'s (kept identical; a row in ``test_search_egress_scope.py``
    runs both in both orders): the first call in a process installs the marked patch, a later one leaves it."""
    from types import SimpleNamespace

    mp = pytest.MonkeyPatch()
    current = brave_key.default_key_path
    existing = getattr(current, "_t0163_nowhere", None)
    if existing is not None:
        return mp, SimpleNamespace(path=existing, original=current._t0163_original)
    nowhere = make_dir() / "home" / "brave_api_key.absent"

    def nowhere_path() -> Any:
        return nowhere

    nowhere_path._t0163_nowhere = nowhere  # type: ignore[attr-defined]
    nowhere_path._t0163_original = current  # type: ignore[attr-defined]
    mp.setattr(brave_key, "default_key_path", nowhere_path)
    return mp, SimpleNamespace(path=nowhere, original=current)
