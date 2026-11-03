"""Unit-test environment (T-0125 amendment 4 (d)): every proxy variable the web-search backend refuses is unset.

A default ``DdgsSearch()`` refuses a proxy taken from the environment, so on a box with ``HTTPS_PROXY`` (or any other
``PROXY_ENV`` name) set, every rig that does a real live start would take the degrade path and print
``search: off (default: ...)``. No unit result may depend on the box's environment: this one autouse fixture unsets
them all. It is session-scoped so it also precedes the module-scoped captures (the BASE-differential ``now``
fixtures), which a function-scoped fixture would not. A row that wants a proxy variable sets it itself with
``monkeypatch.setenv``, which runs after this and is undone after the row.

The names are the backend's own ``PROXY_ENV``, never retyped. Importing the backend module does not import ``ddgs``
(its client factory imports it lazily); the rows that check what a process loads run in fresh interpreters, which do
not load this conftest.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from local_astra.search.ddgs_backend import PROXY_ENV


@pytest.fixture(autouse=True, scope="session")
def _no_proxy_env() -> Iterator[None]:
    mp = pytest.MonkeyPatch()
    for name in PROXY_ENV:
        mp.delenv(name, raising=False)
    yield
    mp.undo()


@pytest.fixture(autouse=True, scope="session")
def brave_key_nowhere(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """T-0163 criterion 10: no test reads the real Brave Search API key file. The loader's default path
    (``brave_key.default_key_path``) points at a path under a session tmp dir that is never created, for the whole
    session (session-scoped, so it also precedes the module-scoped real-start captures). Yields a namespace with
    ``path`` (that nonexistent path) and ``original`` (the unpatched ``default_key_path``, or ``None`` before
    ``local_astra.search.brave_key`` exists); a row that needs the real default rule calls ``original`` under a tmp
    ``HOME``. Fresh interpreters do not load this conftest, so this patch does not protect them: since T-0175 every
    child inherits the root conftest's session jail (``HOME`` and ``XDG_CONFIG_HOME`` point at a per-session dir with no
    ``diffusion_gearvis`` config), and its spawn guard refuses a child whose env names the real home or has no
    ``HOME``. This patch stays as belt and braces (T-0163 criterion 10 rows pin it)."""
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


@pytest.fixture(autouse=True)
def _join_search_threads() -> Iterator[None]:
    """T-0163 criterion 15 (reviewer B F1): a real web search runs on a daemon thread (``astra-brave-N``,
    ``astra-search-N``) that hands its outcome back to the loop and then unwinds; it can still be alive for
    microseconds when the row ends, and ``tests/conftest.py``'s T-0057 leak guard would blame the row. Give each
    such thread 2 s to finish before the guard looks (this fixture is torn down first: it is set up after the root
    conftest's guard). One still alive after that is a real leak and the guard reports it. Only these two prefixes
    are joined: every other ``astra-`` thread is the guard's business, untouched."""
    import threading

    yield
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.name.startswith(
            ("astra-brave-", "astra-search-")
        ):
            t.join(timeout=2)


def install_brave_key_nowhere(brave_key: Any, make_dir: Any) -> tuple[pytest.MonkeyPatch, Any]:
    """Point ``brave_key.default_key_path`` at a nonexistent path, once per process. The unit and the contract
    conftest both call this (an xdist worker can load both, in either order): the first installs the patch, marked
    with ``_t0163_nowhere`` and ``_t0163_original``; a later call finds the mark and leaves that patch in place, so
    every row of the process sees one path. Returns the ``MonkeyPatch`` to undo (empty for a later call) and a
    namespace with ``path`` and ``original``. ``tests/contract/conftest.py`` holds the same function."""
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
