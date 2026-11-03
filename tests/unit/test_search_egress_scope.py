"""T-0111b criteria 7-10: the egress fence around the web-search backend (H-T111c, the scoped CLAUDE.md exception).

Static scans over ``local_astra/**/*.py`` and ``scripts/**/*.py`` (AST for imports, raw text for URL literals, so
docstrings and comments count), plus a fresh interpreter for the default ``--tools dry-run`` kit. Each scanner is a
function of a root directory, and ``test_fence_has_power`` runs it over tmp trees seeded with the violations the task
lists (and a clean control), so a scanner that sees nothing, or flags everything, is caught.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
BACKEND_REL = "local_astra/search/ddgs_backend.py"
INIT_REL = "local_astra/search/__init__.py"
BACKEND_DOTTED = "local_astra.search.ddgs_backend"
TASK = "T-0111b"

SEARCH_CLIENTS = frozenset({"ddgs", "primp"})
#: criterion 7b: what the backend may not import (it reaches the network through ddgs only)
TRANSPORTS = frozenset(
    {"socket", "ssl", "urllib", "httpx", "requests", "aiohttp", "websockets", "primp"}
)
URL_RE = re.compile(r"(?i)\b(?:https?|wss?)://")
#: a bare DNS hostname: labels of [a-z0-9-], no leading/trailing hyphen, at least one dot; nothing else
HOSTNAME_RE = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


def _py_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for top in ("local_astra", "scripts"):
        base = root / top
        if base.is_dir():
            out.extend(sorted(base.rglob("*.py")))
    return out


def _imported_modules(tree: ast.AST) -> set[str]:
    """Every module name a file imports in any form: ``import``, ``from ... import`` (incl. ``from x import y`` as
    ``x.y``), and string arguments to ``importlib.import_module`` / ``import_module`` / ``__import__``."""
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module)
            mods.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Call):
            fn = node.func
            name = (
                fn.id
                if isinstance(fn, ast.Name)
                else fn.attr
                if isinstance(fn, ast.Attribute)
                else ""
            )
            if name in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    mods.add(arg.value)
    return mods


def _top(mods: set[str]) -> set[str]:
    return {m.split(".")[0] for m in mods}


def search_client_importers(root: Path) -> set[str]:
    """Repo-relative paths of files that import ``ddgs`` or ``primp`` in any form."""
    hits: set[str] = set()
    for path in _py_files(root):
        mods = _imported_modules(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if _top(mods) & SEARCH_CLIENTS:
            hits.add(path.relative_to(root).as_posix())
    return hits


def transport_imports(path: Path) -> set[str]:
    mods = _imported_modules(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    bad = {m for m in mods if m.split(".")[0] in TRANSPORTS}
    bad |= {m for m in mods if m == "http.client" or m.startswith("http.client.")}
    return bad


def url_literals(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if URL_RE.search(line)
    ]


_NOT_A_LITERAL = object()


def engines_literal(path: Path) -> object:
    """Ruling (4a): the module-level ``ENGINES = (<literal tuple>)`` (annotated or not). ``None`` if absent;
    ``_NOT_A_LITERAL`` if it is computed (a call, a name, a comprehension), i.e. not a fixed tuple."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        target, value = None, None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == "ENGINES" and value is not None:
            if not isinstance(value, ast.Tuple):
                return _NOT_A_LITERAL
            try:
                return ast.literal_eval(value)
            except ValueError:
                return _NOT_A_LITERAL
    return None


def text_engine_registry() -> dict[str, type]:
    """The installed ddgs's concrete text engines (``ddgs.engines.ENGINES["text"]``), the oracle for 7c (4a)."""
    pytest.importorskip("ddgs", reason="the `search` extra (uv sync --all-extras)")
    import importlib

    return dict(importlib.import_module("ddgs.engines").ENGINES["text"])


def engines_problems(engines: object, registry: dict[str, type]) -> list[str]:
    """Every way ``engines`` breaks ruling (4a)'s fence; empty when it holds."""
    if engines is None:
        return ["no module-level ENGINES"]
    if engines is _NOT_A_LITERAL or not isinstance(engines, tuple):
        return ["ENGINES is not a literal tuple"]
    problems: list[str] = []
    if not 2 <= len(engines) <= 3:
        problems.append(f"{len(engines)} engines, not 2-3")
    if not engines or engines[0] != "duckduckgo":
        problems.append("ENGINES[0] is not 'duckduckgo'")
    if len(set(engines)) != len(engines):
        problems.append("an engine repeats")
    for e in engines:
        if not isinstance(e, str) or e not in registry:
            problems.append(
                f"{e!r} is not a concrete ddgs text engine (ddgs would fall back to 'auto')"
            )
    return problems


def backend_namers(root: Path) -> set[str]:
    """Files (outside the backend itself) that name ``DdgsSearch`` or the backend's dotted module, or import it
    relatively (``from .ddgs_backend import ...``, ``from . import ddgs_backend``)."""
    hits: set[str] = set()
    for path in _py_files(root):
        rel = path.relative_to(root).as_posix()
        if rel == BACKEND_REL:
            continue
        text = path.read_text(encoding="utf-8")
        if "DdgsSearch" in text or BACKEND_DOTTED in text:
            hits.add(rel)
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                names = {node.module or ""} | {a.name for a in node.names}
                if any(
                    n.split(".")[-1] == "ddgs_backend" or n.startswith("ddgs_backend")
                    for n in names
                ):
                    hits.add(rel)
    return hits


def _backend() -> Path:
    path = REPO / BACKEND_REL
    if not path.exists():
        raise NotImplementedError(f"{TASK}: {BACKEND_REL} does not exist yet")
    return path


# --- criterion 7 over the repo ----------------------------------------------------------------------------------


def test_only_the_backend_imports_a_web_search_client() -> None:
    _backend()
    # exactly the backend: the fence holds, and the backend's lazy default client_factory really imports ddgs
    assert search_client_importers(REPO) == {BACKEND_REL}


def test_backend_names_no_url_and_no_transport_library() -> None:
    path = _backend()
    assert transport_imports(path) == set()
    assert url_literals(path) == []


def test_allowed_hosts_are_bare_hostnames() -> None:
    import importlib

    path = _backend()
    m = importlib.import_module(BACKEND_DOTTED)
    hosts = m.ALLOWED_HOSTS
    assert type(hosts) is frozenset and hosts, hosts
    for h in hosts:
        assert isinstance(h, str) and HOSTNAME_RE.match(h), (
            h
        )  # no scheme, path, port, wildcard, upper case
    # ruling (4a): the "DuckDuckGo hosts only" assertion (H-T111b-1) is superseded; the hosts are now pinned to the
    # engines' own hosts in test_allowed_hosts_cover_every_engine
    # module level: a top-level assignment, not something built inside a function
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = {
        t.id
        for n in tree.body
        for t in (n.targets if isinstance(n, ast.Assign) else [getattr(n, "target", None)])
        if isinstance(t, ast.Name)
    }
    assert "ALLOWED_HOSTS" in top
    # ruling (4a): the ``m.ENGINE == "duckduckgo"`` assertion moved to test_engines_are_pinned


def test_engines_are_pinned() -> None:
    """Ruling (4a), 7c: ``ENGINES`` is a fixed literal tuple, ``duckduckgo`` first, 2-3 entries, each a concrete
    engine in the installed ddgs text registry (never ``auto``/``all``/a comma list), and the module's runtime value
    is that literal."""
    import importlib

    path = _backend()
    m = importlib.import_module(BACKEND_DOTTED)
    literal = engines_literal(path)
    assert engines_problems(literal, text_engine_registry()) == []
    assert type(m.ENGINES) is tuple and literal == m.ENGINES
    # (4a) review: the exact tuple recorded under Rulings, in order (a swap or a reorder is a ruling change)
    assert m.ENGINES == ("duckduckgo", "brave", "mojeek")


def test_allowed_hosts_are_exactly_the_engines_search_hosts() -> None:
    """(4a) review: the exact allowlist recorded under Rulings; a host added beside them (``duckduckgo.com``) is a
    widening of the egress exception and needs a ruling."""
    import importlib

    _backend()
    m = importlib.import_module(BACKEND_DOTTED)
    want = {"html.duckduckgo.com", "search.brave.com", "www.mojeek.com"}
    assert type(m.ALLOWED_HOSTS) is frozenset
    assert set(m.ALLOWED_HOSTS) == want


def test_allowed_hosts_cover_every_engine() -> None:
    """Ruling (4a): ``ALLOWED_HOSTS`` holds the host each engine in ``ENGINES`` searches (read from the installed
    engine class's ``search_url``), and nothing from an unrelated site (every host shares its last two labels with
    one engine's host)."""
    import importlib
    from urllib.parse import urlsplit

    _backend()
    m = importlib.import_module(BACKEND_DOTTED)
    registry = text_engine_registry()
    engines = getattr(m, "ENGINES", None)
    if engines is None:
        raise NotImplementedError(f"{TASK} (4a): {BACKEND_DOTTED}.ENGINES does not exist yet")
    assert engines_problems(engines, registry) == []
    hosts = m.ALLOWED_HOSTS
    engine_hosts = {}
    for e in engines:
        host = urlsplit(registry[e].search_url).hostname
        assert host and "{" not in host, (e, registry[e].search_url)
        engine_hosts[e] = host
    missing = {e: h for e, h in engine_hosts.items() if h not in hosts}
    assert missing == {}, f"engines whose search host is not allowlisted: {missing}"
    domains = {".".join(h.split(".")[-2:]) for h in engine_hosts.values()}
    stray = {h for h in hosts if ".".join(h.split(".")[-2:]) not in domains}
    assert stray == set(), f"allowlisted hosts belonging to no engine in ENGINES: {stray}"
    for h in hosts:
        assert HOSTNAME_RE.match(h), h


CLI_REL = "local_astra/cli.py"
SEARCH_HELPER = "_search_backend"


def backend_mentions_outside(path: Path, function: str) -> list[int]:
    """Line numbers in ``path`` that name ``DdgsSearch``/the backend's dotted module/``ddgs_backend`` outside the
    body of the module-level ``def <function>`` (comments and docstrings count: raw text). Every line when the
    function is absent."""
    text = path.read_text(encoding="utf-8")
    span: tuple[int, int] | None = None
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function:
            span = (node.lineno, node.end_lineno or node.lineno)
    out: list[int] = []
    for i, line in enumerate(text.splitlines(), start=1):
        named = "DdgsSearch" in line or BACKEND_DOTTED in line or "ddgs_backend" in line
        if named and (span is None or not span[0] <= i <= span[1]):
            out.append(i)
    return out


def function_imports(path: Path, function: str) -> set[str]:
    """The modules imported inside the body of the module-level ``def <function>``."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function:
            return _imported_modules(node)
    return set()


def test_nothing_outside_tests_names_ddgs_search() -> None:
    """(d), widened by T-0125 criterion 3 (the named spec change): outside ``tests/`` only ``cli.py`` names the
    backend, and only inside the body of ``cli._search_backend``, which imports it lazily (no module-level import,
    so ``load_tool_kit`` and a search-off session never load it)."""
    assert backend_namers(REPO) == {CLI_REL}, (
        f"T-0125 criterion 3: exactly {CLI_REL} names the backend; got {sorted(backend_namers(REPO))}"
    )
    cli_path = REPO / CLI_REL
    assert function_imports(cli_path, SEARCH_HELPER) & {
        BACKEND_DOTTED,
        f"{BACKEND_DOTTED}.DdgsSearch",
    }, f"T-0125 criterion 3: cli.{SEARCH_HELPER} imports {BACKEND_DOTTED} inside its body"
    outside = backend_mentions_outside(cli_path, SEARCH_HELPER)
    assert outside == [], (
        f"T-0125 criterion 3: {CLI_REL} names the backend outside {SEARCH_HELPER}'s body, lines {outside}"
    )


def test_widened_fence_has_power(tmp_path: Path) -> None:
    """T-0125: the widened scan flags a module-level import and a mention in another function; a body-only
    mention passes."""
    ok = (
        "def _search_backend(name):\n"
        "    from local_astra.search.ddgs_backend import DdgsSearch\n"
        "    return DdgsSearch(), 'x'\n"
    )
    top = "from local_astra.search.ddgs_backend import DdgsSearch\n" + ok
    other = ok + "\ndef load_tool_kit():\n    return DdgsSearch()\n"
    absent = "def build():\n    from local_astra.search.ddgs_backend import DdgsSearch\n"
    for name, text, want_clean in (
        ("ok", ok, True),
        ("top", top, False),
        ("other", other, False),
        ("absent", absent, False),
    ):
        p = tmp_path / f"{name}.py"
        p.write_text(text, encoding="utf-8")
        assert (backend_mentions_outside(p, SEARCH_HELPER) == []) is want_clean, name
    assert BACKEND_DOTTED in function_imports(tmp_path / "ok.py", SEARCH_HELPER)
    assert function_imports(tmp_path / "absent.py", SEARCH_HELPER) == set()


def test_search_package_init_imports_nothing_from_the_backend() -> None:
    init = REPO / INIT_REL
    if not init.exists():
        raise NotImplementedError(f"{TASK}: {INIT_REL} does not exist yet")
    tree = ast.parse(init.read_text(encoding="utf-8"))
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == [], "local_astra/search/__init__.py is a docstring only"
    assert INIT_REL not in backend_namers(REPO)


# --- criterion 7e: each scan has power ------------------------------------------------------------------------------

_CLEAN_BACKEND = '''"""A search backend that reaches DuckDuckGo through ddgs."""
from __future__ import annotations

ENGINES = ("duckduckgo", "mojeek")
ALLOWED_HOSTS = frozenset({"duckduckgo.com"})


def _default_client_factory(**kw):
    import ddgs

    return ddgs.DDGS(**kw)
'''


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in {BACKEND_REL: _CLEAN_BACKEND, INIT_REL: '"""Search."""\n', **files}.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def test_fence_clean_control(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"local_astra/other.py": "import json\nx = 'duckduckgo'\n"})
    assert search_client_importers(root) == {BACKEND_REL}
    assert transport_imports(root / BACKEND_REL) == set()
    assert url_literals(root / BACKEND_REL) == []
    assert engines_problems(engines_literal(root / BACKEND_REL), text_engine_registry()) == []
    assert backend_namers(root) == set()


_IMPORT_VIOLATIONS = {
    "from_ddgs_import": "from ddgs import DDGS\n",
    "import_ddgs_submodule": "import ddgs.exceptions as e\n",
    "importlib_import_module": "import importlib\nm = importlib.import_module('ddgs')\n",
    "bare_import_module": "from importlib import import_module\nm = import_module('ddgs.engines')\n",
    "dunder_import_primp": "m = __import__('primp')\n",
    "in_a_function": "def f():\n    from primp import Client\n    return Client\n",
}


@pytest.mark.parametrize("case", list(_IMPORT_VIOLATIONS))
def test_fence_has_power_imports(tmp_path: Path, case: str) -> None:
    for where in ("local_astra/other.py", "scripts/tool.py"):
        root = _tree(tmp_path / where.replace("/", "_"), {where: _IMPORT_VIOLATIONS[case]})
        assert search_client_importers(root) == {BACKEND_REL, where}


@pytest.mark.parametrize(
    "literal",
    [
        "https://html.duckduckgo.com/html/",
        "http://duckduckgo.com",
        "wss://x.example",
        "HTTPS://A.B",
    ],
)
def test_fence_has_power_url_in_docstring(tmp_path: Path, literal: str) -> None:
    root = _tree(tmp_path, {BACKEND_REL: f'"""Posts to {literal}."""\n' + _CLEAN_BACKEND})
    assert url_literals(root / BACKEND_REL) != []


@pytest.mark.parametrize(
    "line",
    [
        "import socket\n",
        "from urllib.parse import urlsplit\n",
        "import http.client\n",
        "from http import client\n",
        "import primp\n",
        "import httpx\n",
    ],
)
def test_fence_has_power_transport(tmp_path: Path, line: str) -> None:
    root = _tree(tmp_path, {BACKEND_REL: _CLEAN_BACKEND + line})
    assert transport_imports(root / BACKEND_REL) != set()


#: ruling (4a): the single-engine power rows (ENGINE = "auto" / "bing" / "duckduckgo,bing" / "") became ENGINES rows
_BAD_ENGINES = {
    "misspelled": 'ENGINES = ("duckduckgo", "mojeeek")',
    "unknown": 'ENGINES = ("duckduckgo", "bing")',  # bing is not a ddgs 9.16.1 *text* engine
    "auto": 'ENGINES = ("duckduckgo", "auto")',
    "comma_list": 'ENGINES = ("duckduckgo", "brave,mojeek")',
    "empty_name": 'ENGINES = ("duckduckgo", "")',
    "not_first": 'ENGINES = ("mojeek", "duckduckgo")',
    "one_only": 'ENGINES = ("duckduckgo",)',
    "four": 'ENGINES = ("duckduckgo", "brave", "mojeek", "yahoo")',
    "repeat": 'ENGINES = ("duckduckgo", "duckduckgo")',
    "computed": 'ENGINES = tuple(["duckduckgo", "mojeek"])',
    "a_list": 'ENGINES = ["duckduckgo", "mojeek"]',
    "absent": "SOMETHING = 1",
}


@pytest.mark.parametrize("case", list(_BAD_ENGINES))
def test_fence_has_power_engines(tmp_path: Path, case: str) -> None:
    text = _CLEAN_BACKEND.replace('ENGINES = ("duckduckgo", "mojeek")', _BAD_ENGINES[case])
    assert text != _CLEAN_BACKEND
    root = _tree(tmp_path, {BACKEND_REL: text})
    assert engines_problems(engines_literal(root / BACKEND_REL), text_engine_registry()) != []


@pytest.mark.parametrize(
    "text",
    [
        "from local_astra.search.ddgs_backend import DdgsSearch\n",
        "import importlib\nm = importlib.import_module('local_astra.search.ddgs_backend')\n",
        "# DdgsSearch is built here\n",
    ],
)
def test_fence_has_power_backend_namers(tmp_path: Path, text: str) -> None:
    root = _tree(tmp_path, {"local_astra/cli.py": text})
    assert backend_namers(root) == {"local_astra/cli.py"}
    root2 = _tree(tmp_path / "init", {INIT_REL: "from .ddgs_backend import DdgsSearch\n"})
    assert INIT_REL in backend_namers(root2)
    root3 = _tree(tmp_path / "init2", {INIT_REL: "from . import ddgs_backend\n"})
    assert INIT_REL in backend_namers(root3)


# --- criterion 8: the default kit never loads the backend ---------------------------------------------------------

_DRY_RUN_SCRIPT = r"""
import importlib.util, sys
from local_astra import cli
from local_astra.config import Profile
from local_astra.toolcall.detok import HostDetokenizer
HostDetokenizer.load = classmethod(lambda cls: object())  # stub detokeniser
kit, _ = cli.load_tool_kit(Profile(name="lean"))
loaded = sorted(m for m in sys.modules
                if m.split(".")[0] in ("ddgs", "primp") or m == "local_astra.search.ddgs_backend")
print("loaded", loaded)
print("search", kit.search)
try:
    findable = importlib.util.find_spec("local_astra.search.ddgs_backend") is not None
except ModuleNotFoundError:
    findable = False
print("findable", findable)
"""


def test_default_dry_run_kit_never_imports_ddgs() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _DRY_RUN_SCRIPT],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.splitlines()
    assert "loaded []" in out, proc.stdout
    assert "search None" in out, proc.stdout
    # the backend exists (so "not loaded" is not vacuous) -- checked last: find_spec imports local_astra.search
    if "findable False" in out:
        raise NotImplementedError(f"{TASK}: {BACKEND_DOTTED} does not exist yet")
    assert "findable True" in out, proc.stdout


# --- criteria 9 and 10: the marker, the make target, the pin ------------------------------------------------------


def _pyproject() -> dict[str, object]:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def test_net_marker_registered_and_search_live_target() -> None:
    ini = _pyproject()["tool"]["pytest"]["ini_options"]  # type: ignore[index]
    markers = [m.split(":")[0].strip() for m in ini["markers"]]
    assert "net" in markers
    assert "-m 'not gpu'" in ini["addopts"]
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    m = re.search(r"^search-live:(.*)$", makefile, re.M)
    if m is None:
        raise NotImplementedError(f"{TASK}: no `search-live` target in the Makefile yet")
    body = m.group(1)
    for part in (
        "ASTRA_LIVE_SEARCH=1",
        "-m net",
        "tests/contract/test_ddgs_live.py",
        "tests/contract/test_search_backend_protocol.py",
    ):
        assert part in body, (part, body)
    check = re.search(r"^check:(.*)$", makefile, re.M)
    assert check is None or "search-live" not in check.group(1)


def test_search_extra_pins_ddgs_exactly() -> None:
    proj = _pyproject()["project"]  # type: ignore[index]
    assert proj["optional-dependencies"]["search"] == ["ddgs==9.16.1"]
    assert not any(
        d.split("=")[0].split("<")[0].split(">")[0].strip() == "ddgs" for d in proj["dependencies"]
    )
    lock = (REPO / "uv.lock").read_text(encoding="utf-8")
    assert re.search(r'^name = "ddgs"\nversion = "9\.16\.1"$', lock, re.M)


# =====================================================================================================================
# T-0163 criterion 9 (additive rows): the fence around the Brave Search API fallback. Same style as above: each scan
# is a function of a root, run over the repo and over tmp trees seeded with violations (power) and a clean control.
# Raw text for the literals (docstrings and comments count, as in the ddgs fence; prose can trip these, by design:
# the host, header and key-dir strings belong in their one module only). Active since T-0163's implementation.
# =====================================================================================================================

BRAVE_API_REL = "local_astra/search/brave_api.py"
BRAVE_KEY_REL = "local_astra/search/brave_key.py"
FALLBACK_REL = "local_astra/search/fallback.py"
BRAVE_HOST = "api.search.brave.com"
TOKEN_HEADER = "X-Subscription-Token"
KEY_DIR = ".config/diffusion_gearvis"
#: criterion 9: what ``brave_api.py`` may not import (it reaches the network through httpx only)
BRAVE_API_FORBIDDEN = frozenset(
    {"socket", "ssl", "urllib", "requests", "aiohttp", "websockets", "primp", "ddgs"}
)
#: criterion 9: what ``brave_key.py`` and ``fallback.py`` may not import (no transport at all)
NO_TRANSPORT = frozenset(BRAVE_API_FORBIDDEN | {"httpx", "httpcore", "h11", "h2", "anyio"})
#: name -> the one module that may contain it outside ``cli._search_backend``'s body
BRAVE_NAMES = {
    "BraveApiSearch": BRAVE_API_REL,
    "FallbackSearch": FALLBACK_REL,
    "local_astra.search.brave_api": BRAVE_API_REL,
    "local_astra.search.fallback": FALLBACK_REL,
    "local_astra.search.brave_key": BRAVE_KEY_REL,
}
_RELATIVE = {"brave_api": BRAVE_API_REL, "fallback": FALLBACK_REL, "brave_key": BRAVE_KEY_REL}


def literal_holders(root: Path, needle: str) -> set[str]:
    """Files under ``local_astra/`` and ``scripts/`` whose raw text contains ``needle`` (case-insensitive: a header
    name or a host spelled in another case is the same literal)."""
    low = needle.lower()
    return {
        p.relative_to(root).as_posix()
        for p in _py_files(root)
        if low in p.read_text(encoding="utf-8").lower()
    }


def brave_api_import_problems(path: Path) -> list[str]:
    mods = _imported_modules(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    problems = [f"imports {m}" for m in sorted(mods) if m.split(".")[0] in BRAVE_API_FORBIDDEN]
    problems += [
        f"imports {m}" for m in sorted(mods) if m == "http.client" or m.startswith("http.client.")
    ]
    if "httpx" not in _top(mods):
        problems.append("does not import httpx")
    if "://" in path.read_text(encoding="utf-8"):
        problems.append("contains a '://' literal")
    return problems


def no_transport_problems(path: Path) -> list[str]:
    mods = _imported_modules(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    bad = [m for m in sorted(mods) if m.split(".")[0] in NO_TRANSPORT]
    bad += [m for m in sorted(mods) if m == "http.client" or m.startswith("http.client.")]
    return [f"imports {m}" for m in bad]


def _span(text: str, function: str) -> tuple[int, int] | None:
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function:
            return node.lineno, node.end_lineno or node.lineno
    return None


def brave_name_problems(root: Path) -> list[str]:
    """Every place a Brave class or module name appears outside its own module, other than inside
    ``cli._search_backend``'s body; and every relative import of a Brave module from another module."""
    problems: list[str] = []
    for path in _py_files(root):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        span = _span(text, SEARCH_HELPER) if rel == CLI_REL else None
        for i, line in enumerate(text.splitlines(), start=1):
            for name, own in BRAVE_NAMES.items():
                if name in line and rel != own and not (span and span[0] <= i <= span[1]):
                    problems.append(f"{rel}:{i} names {name}")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                names = {(node.module or "").split(".")[-1]} | {a.name for a in node.names}
                for short, own in _RELATIVE.items():
                    if short in names and rel != own:
                        problems.append(f"{rel}:{node.lineno} imports .{short} relatively")
    return problems


def _brave_mod(rel: str) -> Path:
    path = REPO / rel
    if not path.exists():
        raise NotImplementedError(f"T-0163: {rel} does not exist yet")
    return path


@pytest.mark.parametrize(
    ("needle", "owner"),
    [(BRAVE_HOST, BRAVE_API_REL), (TOKEN_HEADER, BRAVE_API_REL), (KEY_DIR, BRAVE_KEY_REL)],
    ids=["host", "token_header", "key_dir"],
)
def test_brave_literals_only_in_their_module(needle: str, owner: str) -> None:
    _brave_mod(owner)
    assert literal_holders(REPO, needle) == {owner}, (
        f"T-0163 criterion 9: only {owner} contains {needle!r}; got {sorted(literal_holders(REPO, needle))}"
    )


def test_brave_api_imports_httpx_only_and_no_url_literal() -> None:
    assert brave_api_import_problems(_brave_mod(BRAVE_API_REL)) == []


@pytest.mark.parametrize("rel", [BRAVE_KEY_REL, FALLBACK_REL])
def test_brave_key_and_fallback_import_no_transport(rel: str) -> None:
    assert no_transport_problems(_brave_mod(rel)) == []


def test_brave_names_only_in_search_backend_body() -> None:
    for rel in (BRAVE_API_REL, BRAVE_KEY_REL, FALLBACK_REL):
        _brave_mod(rel)
    assert brave_name_problems(REPO) == []
    imported = function_imports(REPO / CLI_REL, SEARCH_HELPER)
    for dotted in (
        "local_astra.search.brave_api",
        "local_astra.search.fallback",
        "local_astra.search.brave_key",
    ):
        assert any(m == dotted or m.startswith(dotted + ".") for m in imported), (
            f"T-0163 criterion 8: cli.{SEARCH_HELPER} imports {dotted} inside its body; {sorted(imported)}"
        )


def test_brave_allowed_hosts_exactly_the_api_host() -> None:
    import importlib

    _brave_mod(BRAVE_API_REL)
    m = importlib.import_module("local_astra.search.brave_api")
    assert type(m.ALLOWED_HOSTS) is frozenset
    assert frozenset({BRAVE_HOST}) == m.ALLOWED_HOSTS
    tree = ast.parse((REPO / BRAVE_API_REL).read_text(encoding="utf-8"))
    top = {
        t.id
        for n in tree.body
        for t in (n.targets if isinstance(n, ast.Assign) else [getattr(n, "target", None)])
        if isinstance(t, ast.Name)
    }
    assert "ALLOWED_HOSTS" in top, "a module-level constant"


def test_ddgs_engines_and_hosts_unchanged_by_t0163() -> None:
    """Criterion 9: ``ddgs_backend.ENGINES`` and ``ALLOWED_HOSTS`` stay pinned (``api.search.brave.com`` is not added
    to the keyless allowlist)."""
    import importlib

    m = importlib.import_module(BACKEND_DOTTED)
    assert m.ENGINES == ("duckduckgo", "brave", "mojeek")
    assert (
        frozenset({"html.duckduckgo.com", "search.brave.com", "www.mojeek.com"}) == m.ALLOWED_HOSTS
    )


def test_search_live_target_runs_the_brave_live_row() -> None:
    """Criterion 11: ``make search-live`` gains the Brave live module, still opt-in and still not in ``check``."""
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    m = re.search(r"^search-live:(.*)$", makefile, re.M)
    assert m is not None
    body = m.group(1)
    for part in (
        "ASTRA_LIVE_SEARCH=1",
        "-m net",
        "tests/contract/test_brave_api_live.py",
        "tests/contract/test_ddgs_live.py",
    ):
        assert part in body, (part, body)
    check = re.search(r"^check:(.*)$", makefile, re.M)
    assert check is None or "search-live" not in check.group(1)


# --- criterion 10: the fixture is in force ------------------------------------------------------------------------------


@pytest.mark.parametrize("first", ["unit", "contract"])
def test_key_path_fixture_holds_whichever_conftest_runs_first(
    tmp_path: Path, brave_key_nowhere: Any, first: str
) -> None:
    """Criterion 10, mixed xdist workers: a worker that runs a unit row and a contract row instantiates both
    conftests' session fixtures, in either order. Run both installers on a stand-in module in each order: the first
    one's nonexistent path stays the default path (the second leaves the patch and reports the same path and the true
    original), undoing the second changes nothing, undoing the first restores the original. Both conftests hold the
    same installer (compared as AST). And in this process the live default path carries the mark and does not exist."""
    import importlib
    import types

    unit = importlib.import_module("tests.unit.conftest")
    contract = importlib.import_module("tests.contract.conftest")
    srcs = []
    for mod in (unit, contract):
        fn = next(
            n
            for n in ast.parse(Path(mod.__file__).read_text(encoding="utf-8")).body
            if isinstance(n, ast.FunctionDef) and n.name == "install_brave_key_nowhere"
        )
        fn.body = [
            s
            for s in fn.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        ]
        srcs.append(ast.dump(fn))
    assert srcs[0] == srcs[1], "the two conftests' installers differ"

    original = brave_key_nowhere.original
    assert original is not None
    order = (unit, contract) if first == "unit" else (contract, unit)
    stub = types.ModuleType("brave_key_stub")
    stub.default_key_path = original  # type: ignore[attr-defined]
    dirs = iter([tmp_path / "t0163_no_key_a", tmp_path / "t0163_no_key_b"])
    mp1, ns1 = order[0].install_brave_key_nowhere(stub, lambda: next(dirs))
    mp2, ns2 = order[1].install_brave_key_nowhere(stub, lambda: next(dirs))
    want = tmp_path / "t0163_no_key_a" / "home" / "brave_api_key.absent"
    assert stub.default_key_path() == ns1.path == ns2.path == want, (
        stub.default_key_path(),
        ns1.path,
        ns2.path,
    )
    assert not want.exists()
    assert ns1.original is original and ns2.original is original
    mp2.undo()
    assert stub.default_key_path() == want, "undoing the second installer left the first patch"
    mp1.undo()
    assert stub.default_key_path is original

    m = importlib.import_module("local_astra.search.brave_key")
    live = m.default_key_path()
    assert getattr(m.default_key_path, "_t0163_nowhere", None) == live == brave_key_nowhere.path
    assert (
        any(p.name.startswith("t0163_no_key") for p in Path(live).parents)
        and not Path(live).exists()
    )


def test_key_path_fixture_in_force(brave_key_nowhere: Any) -> None:
    """Criterion 10: in this process the loader's default path is the conftest's nonexistent tmp path (not under the
    user's real home), and loading the default says "not configured" without a refusal. The real home is computed from
    the password database as a string only; nothing under it is opened or statted."""
    import importlib
    import os
    import pwd

    path = brave_key_nowhere.path
    if not (REPO / BRAVE_KEY_REL).exists():
        raise NotImplementedError(f"T-0163: {BRAVE_KEY_REL} does not exist yet")
    m = importlib.import_module("local_astra.search.brave_key")
    got = m.default_key_path()
    assert got == path, f"the default path is the fixture's; got {got}"
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert not Path(got).is_relative_to(real_home / ".config"), got
    assert not Path(got).exists()
    result = m.load_key()
    assert result.status == "not configured" and result.reason is None, (
        result.status,
        result.reason,
    )
    contract = ast.parse((REPO / "tests/contract/conftest.py").read_text(encoding="utf-8"))
    fixtures = [
        n
        for n in contract.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "brave_key_nowhere"
        and any(
            "autouse=True" in ast.unparse(d)
            and re.search(r"scope=['\"]session['\"]", ast.unparse(d)) is not None
            for d in n.decorator_list
        )
    ]
    assert len(fixtures) == 1, (
        "tests/contract/conftest.py has the same session-scoped autouse fixture"
    )


# --- each scan has power, and a clean control -----------------------------------------------------------------------

_CLEAN_BRAVE_API = '''"""The Brave Search API backend."""
from __future__ import annotations

import httpx

ALLOWED_HOSTS = frozenset({"api.search.brave.com"})
_SCHEME = "https"
_HEADER = "X-Subscription-Token"


class BraveApiSearch:
    pass
'''
_CLEAN_BRAVE_KEY = '''"""The key file loader."""
from __future__ import annotations

import os
from pathlib import Path


def default_key_path() -> Path:
    return Path.home() / ".config/diffusion_gearvis/brave_api_key"
'''
_CLEAN_FALLBACK = '''"""A primary, then a fallback."""
from __future__ import annotations


class FallbackSearch:
    pass
'''
_CLEAN_CLI = '''"""The CLI."""


def _search_backend(name):
    from local_astra.search import brave_key
    from local_astra.search.brave_api import BraveApiSearch
    from local_astra.search.fallback import FallbackSearch
    from local_astra.search.ddgs_backend import DdgsSearch

    loaded = brave_key.load_key()
    return FallbackSearch(DdgsSearch(), BraveApiSearch(loaded.key)), "x"


def run_session(log):
    log.record("note", {"what": "web search fallback", "brave_api": "configured"})
'''


def _brave_tree(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    base = {
        BRAVE_API_REL: _CLEAN_BRAVE_API,
        BRAVE_KEY_REL: _CLEAN_BRAVE_KEY,
        FALLBACK_REL: _CLEAN_FALLBACK,
        CLI_REL: _CLEAN_CLI,
        "local_astra/other.py": "import json\nx = 'brave search'\n",
        "scripts/tool.py": "print('fallback')\n",
    }
    return _tree(tmp_path, {**base, **(files or {})})


def test_brave_fence_clean_control(tmp_path: Path) -> None:
    root = _brave_tree(tmp_path)
    assert literal_holders(root, BRAVE_HOST) == {BRAVE_API_REL}
    assert literal_holders(root, TOKEN_HEADER) == {BRAVE_API_REL}
    assert literal_holders(root, KEY_DIR) == {BRAVE_KEY_REL}
    assert brave_api_import_problems(root / BRAVE_API_REL) == []
    assert no_transport_problems(root / BRAVE_KEY_REL) == []
    assert no_transport_problems(root / FALLBACK_REL) == []
    assert brave_name_problems(root) == []
    assert {"local_astra.search.brave_api", "local_astra.search.fallback"} <= function_imports(
        root / CLI_REL, SEARCH_HELPER
    )


@pytest.mark.parametrize(
    ("needle", "where", "text"),
    [
        (
            BRAVE_HOST,
            "local_astra/conductor/session.py",
            '"""Asks https://api.search.brave.com."""\n',
        ),
        (BRAVE_HOST, "scripts/probe.py", "HOST = 'API.SEARCH.BRAVE.COM'\n"),
        (TOKEN_HEADER, CLI_REL, _CLEAN_CLI + "# sends x-subscription-token\n"),
        (TOKEN_HEADER, FALLBACK_REL, _CLEAN_FALLBACK + "H = 'X-Subscription-Token'\n"),
        (KEY_DIR, CLI_REL, _CLEAN_CLI + "P = '~/.config/diffusion_gearvis/brave_api_key'\n"),
        (KEY_DIR, BRAVE_API_REL, _CLEAN_BRAVE_API + "# key in .config/diffusion_gearvis\n"),
    ],
    ids=[
        "host_docstring",
        "host_upper_in_script",
        "header_comment_cli",
        "header_fallback",
        "keydir_cli",
        "keydir_api",
    ],
)
def test_fence_has_power_brave_literals(tmp_path: Path, needle: str, where: str, text: str) -> None:
    root = _brave_tree(tmp_path, {where: text})
    assert where in literal_holders(root, needle)


@pytest.mark.parametrize(
    "line",
    [
        "import socket\n",
        "import ssl\n",
        "from urllib.parse import urlencode\n",
        "import requests\n",
        "import aiohttp\n",
        "import websockets\n",
        "import primp\n",
        "import http.client\n",
        "from http import client\n",
        "from ddgs import DDGS\n",
        "def f():\n    import socket\n",
        "import importlib\nm = importlib.import_module('ssl')\n",
    ],
)
def test_fence_has_power_brave_api_transport(tmp_path: Path, line: str) -> None:
    root = _brave_tree(tmp_path, {BRAVE_API_REL: _CLEAN_BRAVE_API + line})
    assert brave_api_import_problems(root / BRAVE_API_REL) != []


def test_fence_has_power_brave_api_needs_httpx_and_no_scheme_literal(tmp_path: Path) -> None:
    root = _brave_tree(
        tmp_path / "a", {BRAVE_API_REL: _CLEAN_BRAVE_API.replace("import httpx\n", "")}
    )
    assert brave_api_import_problems(root / BRAVE_API_REL) == ["does not import httpx"]
    root = _brave_tree(
        tmp_path / "b",
        {BRAVE_API_REL: _CLEAN_BRAVE_API + 'URL = "https://api.search.brave.com/res"\n'},
    )
    assert brave_api_import_problems(root / BRAVE_API_REL) == ["contains a '://' literal"]


@pytest.mark.parametrize("rel", [BRAVE_KEY_REL, FALLBACK_REL])
@pytest.mark.parametrize(
    "line",
    [
        "import httpx\n",
        "import socket\n",
        "from urllib import request\n",
        "import ddgs\n",
        "import ssl\n",
    ],
)
def test_fence_has_power_no_transport(tmp_path: Path, rel: str, line: str) -> None:
    clean = _CLEAN_BRAVE_KEY if rel == BRAVE_KEY_REL else _CLEAN_FALLBACK
    root = _brave_tree(tmp_path, {rel: clean + line})
    assert no_transport_problems(root / rel) != []


@pytest.mark.parametrize(
    ("where", "text"),
    [
        (
            "local_astra/conductor/session.py",
            "from local_astra.search.brave_api import BraveApiSearch\n",
        ),
        (CLI_REL, "from local_astra.search.fallback import FallbackSearch\n" + _CLEAN_CLI),
        (CLI_REL, _CLEAN_CLI + "\ndef load_tool_kit():\n    return FallbackSearch\n"),
        (
            "scripts/tool.py",
            "import importlib\nm = importlib.import_module('local_astra.search.brave_key')\n",
        ),
        (FALLBACK_REL, _CLEAN_FALLBACK + "# wraps DdgsSearch and BraveApiSearch\n"),
        (BRAVE_API_REL, _CLEAN_BRAVE_API + "from local_astra.search.brave_key import load_key\n"),
        ("local_astra/search/__init__.py", "from .brave_api import BraveApiSearch\n"),
        ("local_astra/search/__init__.py", "from . import fallback\n"),
    ],
    ids=[
        "session_imports",
        "cli_top_level",
        "cli_other_function",
        "script_import_module",
        "fallback_names_brave",
        "api_imports_key_module",
        "init_relative_from",
        "init_relative_module",
    ],
)
def test_fence_has_power_brave_names(tmp_path: Path, where: str, text: str) -> None:
    root = _brave_tree(tmp_path, {where: text})
    problems = brave_name_problems(root)
    assert any(p.startswith(where + ":") for p in problems), problems
