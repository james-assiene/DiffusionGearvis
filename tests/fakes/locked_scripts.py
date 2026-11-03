"""Load ``scripts/locked/*.py`` by path, turning "it does not exist yet" into an AssertionError.

T-0023's two measurement scripts live in ``scripts/locked/`` (the ``/launch``-gated directory
``.claude/hooks/guard_bash.sh`` recognises), which is not a package: their pure helpers are
reached with ``importlib.util.spec_from_file_location``, never with ``import``.

Everything that can go wrong before the implementation exists — a missing file, a missing helper,
a syntax error, a module-level import of a library the box does not have — is raised as an
``AssertionError`` naming the file. That is deliberate: ``tests/conftest.py`` converts
``pending(task=...)`` tests to ``xfail(strict=True, raises=(NotImplementedError, AssertionError))``,
so an ``ImportError`` from a not-yet-written script would be a *test error*, not a pending test
(``.claude/rules/tests.md``: "a pending test never errors at collection").
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
LOCKED = REPO / "scripts" / "locked"

#: The two scripts T-0023 owns. Their pure helpers are specified in
#: ``tests/unit/test_audio_measure_report.py``'s module docstring.
ACOUSTIC = "audio_acoustic"
SOAK = "audio_soak"

_CACHE: dict[str, ModuleType] = {}


def load_locked_script(name: str) -> ModuleType:
    """Import ``scripts/locked/<name>.py`` under a private module name, or fail by file name."""
    if name in _CACHE:
        return _CACHE[name]
    path = LOCKED / f"{name}.py"
    if not path.exists():
        raise AssertionError(
            f"scripts/locked/{name}.py does not exist (T-0023 owns it; its pure-helper API is "
            f"pinned in tests/unit/test_audio_measure_report.py's module docstring)"
        )
    spec = importlib.util.spec_from_file_location(f"astra_locked_{name}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"scripts/locked/{name}.py is not importable by path: no loader")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - the pending path
        raise AssertionError(
            f"scripts/locked/{name}.py failed to import: {type(exc).__name__}: {exc}. "
            f"Its pure helpers must be importable with no hardware, no device library and no "
            f"argument parsing at module scope (T-0023 criterion 1, group A runs in the fast suite)"
        ) from exc
    _CACHE[name] = module
    return module


def helper(module: ModuleType, name: str, why: str) -> Any:
    """Fetch a pinned helper; its absence is an AssertionError naming the script and the criterion."""
    if not hasattr(module, name):
        raise AssertionError(
            f"{getattr(module, '__file__', module.__name__)} has no {name!r} ({why})"
        )
    return getattr(module, name)
