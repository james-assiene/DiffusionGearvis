"""T-0198 criteria 7 and 8 on the real driver class without a browser: ``PlaywrightDriver.bring_to_front()`` awaits
the active page's ``bring_to_front()`` once and does nothing else; with no page it raises the existing
``RuntimeError``; ``runtime.py``'s imports are BASE's (docs/tasks/T-0198.md).

The driver is built (never launched) and given a stub ``_page`` whose every other attribute is a named failure.
The import guard is closed to T-0198's own range (T-0203 A re-pin, rules/tests.md "pin both ends"): the import sets
of ``runtime.py`` at ``BASE_SHA`` and at ``TIP_SHA`` (T-0198's branch tip, the second parent of merge 8014990b) are
read with ``git show``, both literals; the working tree is never read, so later tasks may change the imports.

Regime: CPU, no browser, no socket.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_astra.computer import runtime as runtime_mod

REPO = Path(__file__).resolve().parents[2]
#: T-0198's base: main at 8e8d3c52 ("T-0198 planned"). A literal.
BASE_SHA = "8e8d3c52"
#: T-0198's branch tip: the second parent of its merge 8014990b. A literal (closes the guard's range, T-0203 A).
TIP_SHA = "bbd3c29f"


class StubPage:
    def __init__(self) -> None:
        self.log: list[str] = []

    async def bring_to_front(self) -> None:
        self.log.append("bring_to_front")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise AssertionError(f"T-0198 C8: bring_to_front() used page.{name}")


def driver() -> Any:
    return runtime_mod.PlaywrightDriver(viewport=(1280, 800), on_egress=lambda e: None)


def state(d: Any) -> dict[str, Any]:
    return {k: v for k, v in vars(d).items() if k != "_page"}


async def test_bring_to_front_awaits_page_once_and_nothing_else() -> None:
    """C8: one awaited ``page.bring_to_front()``; the driver's state (generation, page serial, events, pages,
    navigation bookkeeping) is unchanged."""
    d = driver()
    page = StubPage()
    d._page = page
    before = repr(state(d))
    fn = getattr(type(d), "bring_to_front", None)
    assert fn is not None, "T-0198 C8: PlaywrightDriver.bring_to_front does not exist yet"
    await d.bring_to_front()
    assert page.log == ["bring_to_front"], page.log
    assert repr(state(d)) == before, "T-0198 C8: bring_to_front changed the driver's state"


async def test_bring_to_front_without_page_raises_runtime_error() -> None:
    """C8: no active page: the existing ``RuntimeError`` (the ``page`` property's), nothing else."""
    d = driver()
    assert getattr(type(d), "bring_to_front", None) is not None, (
        "T-0198 C8: PlaywrightDriver.bring_to_front does not exist yet"
    )
    with pytest.raises(RuntimeError, match="no active page"):
        await d.bring_to_front()


def _imports(src: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            out |= {f"import {a.name}" for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out |= {f"from {node.module} import {a.name}" for a in node.names}
    return out


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def runtime_import_drift(base: str, tip: str) -> list[str]:
    """The imports of ``runtime.py`` that differ between commits ``base`` and ``tip`` (both read with ``git show``);
    skips with a reason when either commit is absent or ``base`` is not an ancestor of ``tip``."""
    for rev in (base, tip):
        if _git("cat-file", "-e", f"{rev}^{{commit}}").returncode != 0:
            pytest.skip(f"commit {rev} is not available here (a git archive copy?)")
    if _git("merge-base", "--is-ancestor", base, tip).returncode != 0:
        pytest.skip(f"{base} is not an ancestor of {tip}: not T-0198's range")
    blobs = []
    for rev in (base, tip):
        got = _git("show", f"{rev}:local_astra/computer/runtime.py")
        assert got.returncode == 0, got.stderr
        blobs.append(_imports(got.stdout))
    return sorted(blobs[0] ^ blobs[1])


def test_runtime_imports_unchanged() -> None:
    """C7 (negative, closed range): over T-0198's own range ``BASE_SHA..TIP_SHA``, ``runtime.py`` imports exactly
    what it imported at BASE."""
    drift = runtime_import_drift(BASE_SHA, TIP_SHA)
    assert drift == [], f"T-0198 C7: runtime.py's imports changed in {BASE_SHA}..{TIP_SHA}: {drift}"
