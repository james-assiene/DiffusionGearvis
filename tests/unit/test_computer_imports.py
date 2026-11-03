"""T-0111c criterion 16 (import hygiene) and the task's two non-code deliverables (the ``browser`` marker, ``make
browser``).

- Only ``local_astra/computer/runtime.py`` imports ``playwright``, anywhere in ``local_astra/`` (an AST scan:
  ``import``/``from`` with relative imports resolved, plus ``importlib.import_module``/``__import__`` with a literal).
- ``local_astra.conductor.session``, ``local_astra.cli``, ``local_astra.computer`` and
  ``local_astra.computer.executor`` import in a fresh interpreter whose import system refuses ``playwright``. The
  session and cli rows are green at BASE and must stay green (no marker); the two package rows are pending.
- ``local_astra/computer/**`` imports no socket client and names no ``open_connection``/``create_connection``,
  except ``local_astra/computer/egress_proxy.py`` (T-0160 O-A2 (1), a named spec change).

The scanner and the blocker each have a self-check row, green by design: a scan that sees nothing and a blocker that
blocks nothing would make the package rows vacuous.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "local_astra"
COMPUTER = PKG / "computer"
#: T-0134 (R12, re-scoped in docs/tasks/T-0134.md): the target also runs the T-0134 extent browser module;
#: T-0140 A1 added its overview module and T-0151 A1 (its Makefile scope) adds the overview-frame module;
#: T-0174 B1 (R-B1) adds the navigated module; T-0198 criterion 9 adds the fronting module; T-0200 criterion 10 adds
#: the blank re-capture module; T-0202 B10 adds the wait module (the orchestrator edits the Makefile at merge;
#: ``test_t0138_suite_static`` reads this too)
MAKE_BROWSER = (
    "ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_computer_browser_e2e.py "
    "tests/integration/test_t0134_extent_browser.py tests/integration/test_t0140_overview_browser.py "
    "tests/integration/test_t0151_overview_frame_browser.py "
    "tests/integration/test_t0174_navigated_browser.py "
    "tests/integration/test_t0188_beforeunload_browser.py "
    "tests/integration/test_t0189_type_replaces_browser.py "
    "tests/integration/test_t0194_slow_focus_browser.py "
    "tests/integration/test_t0198_front_browser.py "
    "tests/integration/test_t0200_blank_recapture_browser.py "
    "tests/integration/test_t0202_wait_browser.py "
    "tests/integration/test_t0203_follow_driver_browser.py "
    "tests/integration/test_t0203_follow_browser.py "
    "tests/integration/test_t0204_driver_signals_browser.py "
    "tests/integration/test_t0204_preclick_browser.py "
    "tests/integration/test_t0204_settle_browser.py "
    "tests/integration/test_t0206_submit_browser.py "
    "tests/integration/test_t0210_body_focus_browser.py -q -m browser"
)
SOCKET_CLIENTS = (
    "socket",
    "ssl",
    "http.client",
    "urllib.request",
    "urllib3",
    "requests",
    "httpx",
    "aiohttp",
    "websockets",
    "websocket",
    "smtplib",
    "ftplib",
    "telnetlib",
    "xmlrpc.client",
)
SOCKET_ATTRS = (
    "open_connection",
    "create_connection",
    "open_unix_connection",
    "create_datagram_endpoint",
)


def imported_names(path: Path, root: Path) -> Iterator[tuple[int, str]]:
    """Every absolute module name ``path`` imports (relative imports resolved against its package)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    pkg = list(path.relative_to(root).parent.parts)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                mod = ".".join([*base, *([node.module] if node.module else [])])
            else:
                mod = node.module or ""
            yield node.lineno, mod
            for alias in node.names:
                yield node.lineno, f"{mod}.{alias.name}"
        elif isinstance(node, ast.Call):
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fname in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield node.lineno, arg.value


def _is(name: str, mod: str) -> bool:
    return name == mod or name.startswith(mod + ".")


def playwright_importers(root: Path, pkg: Path) -> set[str]:
    return {
        str(p.relative_to(root))
        for p in sorted(pkg.rglob("*.py"))
        if any(_is(n, "playwright") for _, n in imported_names(p, root))
    }


def socket_uses(root: Path, pkg: Path) -> list[str]:
    bad = []
    for p in sorted(pkg.rglob("*.py")):
        rel = p.relative_to(root)
        for lineno, n in imported_names(p, root):
            if any(_is(n, m) for m in SOCKET_CLIENTS) or n.rsplit(".", 1)[-1] in SOCKET_ATTRS:
                bad.append(f"{rel}:{lineno}: import {n}")
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Attribute) and node.attr in SOCKET_ATTRS:
                bad.append(f"{rel}:{node.lineno}: .{node.attr}")
    return bad


# --------------------------------------------------------------------------------------------- self-checks (green)


def test_the_scan_sees_every_import_form(tmp_path: Path) -> None:
    """The scanner is not vacuous: each form of a playwright import and of a socket client is seen; a module that
    only mentions the names in a string or imports ``urllib.parse`` is not."""
    pkg = tmp_path / "local_astra" / "computer"
    pkg.mkdir(parents=True)
    forms = {
        "a.py": "import playwright\n",
        "b.py": "from playwright.async_api import async_playwright\n",
        "c.py": "import importlib\nimportlib.import_module('playwright.async_api')\n",
        "d.py": "def f():\n    import playwright.sync_api\n",
        "clean.py": "import urllib.parse\nDOC = 'import playwright; import socket'\n",
    }
    for name, src in forms.items():
        (pkg / name).write_text(src)
    assert playwright_importers(tmp_path, pkg) == {
        "local_astra/computer/a.py",
        "local_astra/computer/b.py",
        "local_astra/computer/c.py",
        "local_astra/computer/d.py",
    }
    (pkg / "s1.py").write_text("import socket\n")
    (pkg / "s2.py").write_text(
        "import asyncio\nasync def f():\n    await asyncio.open_connection('x', 1)\n"
    )
    (pkg / "s3.py").write_text("from http.client import HTTPConnection\n")
    bad = socket_uses(tmp_path, pkg)
    assert {b.split(":", 1)[0] for b in bad} == {
        "local_astra/computer/s1.py",
        "local_astra/computer/s2.py",
        "local_astra/computer/s3.py",
    }, bad


def _import_blocking_playwright(module: str) -> subprocess.CompletedProcess[str]:
    code = textwrap.dedent(
        f"""
        import importlib, sys

        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name == "playwright" or name.startswith("playwright."):
                    raise ImportError("blocked by the T-0111c test: " + name)
                return None

        sys.meta_path.insert(0, _Block())
        importlib.import_module({module!r})
        loaded = sorted(m for m in sys.modules if m == "playwright" or m.startswith("playwright."))
        print("PLAYWRIGHT_LOADED", loaded)
        """
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_the_blocker_blocks_playwright() -> None:
    """Self-check: playwright is installed (P1), and the blocker still makes importing it fail."""
    proc = _import_blocking_playwright("playwright.async_api")
    assert proc.returncode != 0 and "blocked by the T-0111c test" in proc.stderr, (
        proc.stdout,
        proc.stderr,
    )


# --------------------------------------------------------------------------------------------- criterion 16


def test_playwright_confined_to_runtime() -> None:
    assert COMPUTER.is_dir(), "T-0111c: local_astra/computer/ does not exist yet"
    assert playwright_importers(REPO, PKG) == {"local_astra/computer/runtime.py"}


@pytest.mark.parametrize(
    "module",
    [
        "local_astra.conductor.session",
        "local_astra.cli",
        "local_astra.computer",
        "local_astra.computer.executor",
    ],
)
def test_conductor_imports_without_playwright(module: str) -> None:
    proc = _import_blocking_playwright(module)
    last = (proc.stderr.strip().splitlines() or [""])[-1]
    assert proc.returncode == 0, (
        f"{module} does not import with playwright blocked: {last}\n{proc.stderr[-2000:]}"
    )
    assert "PLAYWRIGHT_LOADED []" in proc.stdout, proc.stdout


#: T-0160 O-A2 (1), a named spec change: the egress proxy is the one ``computer/**`` file allowed to open sockets
#: (criterion A-5). Exactly this path, relative to the repo root; nothing else under ``computer/`` is exempt.
SOCKET_EXEMPT = frozenset({"local_astra/computer/egress_proxy.py"})


def socket_uses_outside_exempt(root: Path, pkg: Path) -> list[str]:
    return [b for b in socket_uses(root, pkg) if b.split(":", 1)[0] not in SOCKET_EXEMPT]


def test_package_imports_no_socket_client() -> None:
    files = sorted(COMPUTER.rglob("*.py")) if COMPUTER.is_dir() else []
    names = {p.name for p in files}
    assert {"__init__.py", "executor.py", "runtime.py", "fence.py"} <= names, (
        f"T-0111c: local_astra/computer/ has {sorted(names)}"
    )
    assert socket_uses_outside_exempt(REPO, COMPUTER) == []


def test_socket_exemption_is_exactly_the_egress_proxy(tmp_path: Path) -> None:
    """T-0160 O-A2 (1), power row: with the exemption applied, a socket import in ``egress_proxy.py`` passes, and
    the same import in any other ``computer/**`` file (a sibling, a look-alike name, a subpackage) still fails."""
    assert {"local_astra/computer/egress_proxy.py"} == SOCKET_EXEMPT
    pkg = tmp_path / "local_astra" / "computer"
    (pkg / "sub").mkdir(parents=True)
    src = "import asyncio\nimport socket\nasync def f():\n    await asyncio.open_connection('127.0.0.1', 1)\n"
    for rel in ("egress_proxy.py", "executor.py", "egress_proxy_helpers.py", "sub/egress_proxy.py"):
        (pkg / rel).write_text(src)
    bad = socket_uses_outside_exempt(tmp_path, pkg)
    assert {b.split(":", 1)[0] for b in bad} == {
        "local_astra/computer/executor.py",
        "local_astra/computer/egress_proxy_helpers.py",
        "local_astra/computer/sub/egress_proxy.py",
    }, bad


# --------------------------------------------------------------------------------------------- marker and target


def test_browser_marker_registered() -> None:
    """The ``browser`` marker is in ``pyproject.toml`` (the e2e module adds it to ``pytestmark`` once registered)."""
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    markers = data["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.split(":", 1)[0].strip() == "browser" for m in markers), markers


def test_make_browser_target() -> None:
    """``make browser`` runs exactly the task's command (read with ``make -n``, which runs nothing). Re-pinned for
    T-0151 A1 (its Makefile scope adds the overview-frame browser module). Re-pinned for T-0189 (the target also
    runs the T-0188 beforeunload and T-0189 type-replaces browser modules). Re-pinned for T-0203 B (both T-0203
    browser modules, A's driver rows then B's executor rows; the orchestrator edits the Makefile at merge). Re-pinned
    for T-0204 B (A's driver-signal browser module, added at B's merge per T-0204's Owned files, then B's pre-click
    browser module; the orchestrator edits the Makefile). Re-pinned for T-0204 C (the settle browser module last; the
    orchestrator edits the Makefile at C's merge). Re-pinned for T-0206 A (the submit-stop browser module last; the
    orchestrator edits the Makefile at A's merge). Re-pinned for T-0210 A (the body-focus browser module last; the
    orchestrator edits the Makefile at A's merge)."""
    proc = subprocess.run(
        ["make", "-n", "-s", "browser"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [ln.split("#", 1)[0].strip() for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines == [MAKE_BROWSER], proc.stdout
