"""T-0210 group B, criteria B0, B1, B2 (the files half), B3 (the shim's static bans), B12 and B14: the pinned three.js
editor copy read as files, its provenance, its own page root and the frozen paths (docs/tasks/T-0210.md, "Tests to
write", ``tests/unit/test_t0210_threejs_static.py``; rulings B-R1..B-R6 in the task's Status line).

**What these rows read.** ``threejs_tasks.THREEJS_DIR`` (through :func:`threejs_dir`, a named ``AssertionError``
"does not exist yet" while the module or the directory is absent), the files under it, ``PROVENANCE.md``,
``pages.server.resolve`` and real loopback ``StartPageServer``s, ``make -n -s`` and ``git diff`` over literal commits.
One row also reads the upstream r179 tree (below). No browser, no network.

**The upstream tree.** H-T210-6 alternative 2: a sparse clone at ``~/threejs-r179`` of commit
``54ac263593c81b669ca9a089491ddd9e240427d2``. The test session's ``HOME`` is jailed (``tests/_home_jail.py``), so the
row finds the real home through ``pwd`` (or ``ASTRA_THREEJS_UPSTREAM``) and skips, naming the path, when the tree is
absent. The tree is untrusted data: it is only read and hashed; nothing in it is run, imported or handed to ``git``
(the commit is read from ``.git/HEAD`` / ``packed-refs`` as text).

**Readings this module pins where the task is silent or superseded** (reported to the orchestrator):

- ``PROVENANCE.md``'s file table: one Markdown row ``| <path> | <sha256 vendored> | unmodified|patched|ours |
  <sha256 upstream> |`` per file under ``THREEJS_DIR`` except ``PROVENANCE.md`` itself (it cannot hold its own sha).
  The path is relative to ``THREEJS_DIR`` (optionally in backticks) and equals the upstream path for ``unmodified``
  and ``patched`` rows; an ``ours`` row's upstream cell is not a sha (``—``, ``-`` or ``n/a``). ``unmodified`` means
  vendored sha == upstream sha; ``patched`` means they differ.
- ``ours`` is exactly ``editor/astra_state.js``, ``editor/astra_seeds.js`` and the stub files (any file with
  ``astra_stubs`` in its path); ``editor/index.html`` is ``patched`` (P1-P3, P5) and ``LICENSE`` ``unmodified``.
- B-R3: the vendored ``editor/`` is upstream's ``editor/**`` with an allowed suffix, minus ``sw.js`` (so
  ``images/icon.xcf`` and ``docs/*.md`` are absent), plus the ``ours`` files. ``build/three.module.js`` and
  ``build/three.core.js`` are present. No symlink anywhere in the copy.
- B-R6 / B1: no URL attribute of an ``index.html`` tag and no importmap value is absolute (``scheme:`` or ``//``);
  ``index.html`` carries no ``serviceWorker`` registration and names no ``sw.js``; ``PROVENANCE.md`` names each
  removed off-origin reference (``three-gpu-pathtracer``, ``three-mesh-bvh``, ``ffmpeg``, ``favicon``) and the
  service worker. P5: exactly one ``<script type="module" src="astra_state.js">``.
- P1 stubs: an importmap entry for ``three-gpu-pathtracer`` (it has a consumer, ``Viewport.Pathtracer.js``) resolves
  to a file named ``three-gpu-pathtracer.js`` under a directory named ``astra_stubs`` in the copy. B-R8: the
  ``three-mesh-bvh`` entry is removed (nothing imports it at r179), so it is not mapped and has no stub; only the path
  tracer has one. Every name a vendored module imports from a stubbed specifier is exported by the stub
  (``WebGLPathTracer`` at r179).
- B-R7 (amended): the patched ``index.html`` names the favicon as the vendored local ``images/icon.png`` (one
  ``rel="icon"`` link, the file present and ``unmodified``), so Chromium requests no ``/favicon.ico``; the session
  module's zero-404 rows hold that intent.
- The static import closure: from ``editor/index.html``'s classic ``<script src>``, ``<link href>``, importmap and
  inline module scripts, every static ``import``/``export ... from`` resolves to a file in the copy and none is
  off-origin. Dynamic ``import()`` is not followed (the loaders behind File › Import are not vendored by B0), except
  ``Sidebar.Geometry.BoxGeometry.js`` and ``Sidebar.Geometry.SphereGeometry.js``, which the golden paths reach
  through ``Sidebar.Geometry.js``'s templated import.
- B2 layout files (:data:`LAYOUT`) are ``unmodified``: B2's checks are about the upstream editor, so a patch to them
  would make those checks say nothing. ``Menubar.Help.js`` is excluded (P4 patches it).
- Determinism and network bans (B14 last sentence, B3 "never fetches, no timers of its own") apply to the ``ours``
  files only: no ``Date``, ``Math.random``, ``setTimeout``/``setInterval``/``requestAnimationFrame``/
  ``requestIdleCallback``, ``fetch``, XHR, WebSocket, EventSource, ``sendBeacon``, ``Worker``, ``import()``,
  ``://``. ``requestAnimationFrame`` counts as a timer (reading). The rule is textual: prose in a comment trips it.
- B1 404s: the threejs root answers 404 for every path the default, scenarios, suite and rtx roots serve (``/``
  included: the copy has no root ``index.html``), and each of those roots answers 404 for every threejs path. The
  bulk is checked through ``server.resolve`` (the server's own 404 decision), with real-server spot checks.
- B-R4: the copy's total size (sum of file sizes) is at most 12 x 10^6 bytes.
- B14's BASE is the orchestrator's literal 26196a88. ``TIP`` is B's implementation commit d81727e9 (round 3;
  2b9e7e91 at round 2, B-R21; c81f1ebf at first activation) and the range starts at ``RANGE_BASE`` =
  ``merge-base(main, TIP)`` = dc7c6591 (175a29b0 for 2b9e7e91, 968e47d4 for c81f1ebf; the A8 pattern; part 1's branch forked
  at c907ef93, before 26196a88, and the branches merged main mid-way), with premise checks that BASE precedes it and
  that B's first commit lies inside the range.

BASE = 26196a88 (main when B's rulings were written). Written red ahead of the copy (pending ``T-0210``); activated
against B's implementation at c81f1ebf (rulings B-R7 amended, B-R8, B-R9); the Makefile row stays red until the
orchestrator adds ``browser-threejs`` at merge.

Regime: CPU, files and loopback sockets only, n = 1.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import posixpath
import pwd
import re
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.unit.test_start_pages import new_server, port_of, request
from tests.unit.test_t0205_scenario_static import (
    ABSOLUTE,
    SCRIPT_BLOCK,
    TAG,
    URL_ATTRS,
    _make_lines,
    attrs,
    url_paths,
)

REPO = Path(__file__).resolve().parents[2]
PAGES = REPO / "local_astra" / "pages"
THREEJS_PATH = PAGES / "threejs"
MODULE = "local_astra.pages.threejs_tasks"
#: B1: the suffixes ``pages/server.py._TYPES`` serves, plus exactly ``LICENSE`` and ``PROVENANCE.md`` (B-R3).
ALLOWED_SUFFIXES = frozenset({".html", ".css", ".js", ".json", ".svg", ".png", ".ico"})
ALLOWED_NAMES = frozenset({"LICENSE", "PROVENANCE.md"})
#: H-T210-7 / the Status line: the r179 tree James fetched on 2026-10-30.
UPSTREAM_COMMIT = "54ac263593c81b669ca9a089491ddd9e240427d2"
UPSTREAM_DIRNAME = "threejs-r179"
#: Typed from the upstream LICENSE's first line and copyright line (read 2026-10-30 at the commit above).
LICENSE_FIRST = "The MIT License"
LICENSE_COPYRIGHT = "Copyright © 2010-2025 three.js authors"
OURS = frozenset({"editor/astra_state.js", "editor/astra_seeds.js"})
STATUSES = frozenset({"unmodified", "patched", "ours"})
#: B2's layout, as files (Menubar.Help.js excluded: P4 patches it).
LAYOUT = (
    "editor/js/Editor.js",
    "editor/js/Menubar.js",
    "editor/js/Menubar.File.js",
    "editor/js/Menubar.Edit.js",
    "editor/js/Menubar.Add.js",
    "editor/js/Menubar.View.js",
    "editor/js/Viewport.js",
    "editor/js/Toolbar.js",
    "editor/js/Sidebar.js",
    "editor/js/Sidebar.Scene.js",
    "editor/js/Sidebar.Properties.js",
    "editor/js/Sidebar.Object.js",
    "editor/js/Sidebar.Geometry.js",
    "editor/js/Sidebar.Material.js",
    "editor/js/Sidebar.Material.NumberProperty.js",
    "editor/js/libs/ui.js",
    "editor/js/libs/ui.three.js",
)
#: B-R6: the off-origin references of upstream's index.html, each named in PROVENANCE.md.
OFF_ORIGIN_NAMES = ("three-gpu-pathtracer", "three-mesh-bvh", "ffmpeg", "favicon")
STUBBED = ("three-gpu-pathtracer",)
#: B-R8: removed from the importmap, no stub.
REMOVED_SPECIFIERS = ("three-mesh-bvh",)
#: The golden paths' dynamic geometry panels (Sidebar.Geometry.js imports `./Sidebar.Geometry.${type}.js`).
DYNAMIC_NEEDED = (
    "editor/js/Sidebar.Geometry.BoxGeometry.js",
    "editor/js/Sidebar.Geometry.SphereGeometry.js",
)
#: B-R4.
SIZE_CAP = 12 * 10**6
MAKE_BROWSER_THREEJS = "ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0210_threejs_session.py -q -m browser"

#: T-0210 group B's BASE: main at 26196a88 ("T-0208 C merged; status"), the orchestrator's literal.
BASE = "26196a880f424f6d1de4b412c4ff5c4c4dcc76af"
#: B's implementation tip (T-0210-b-impl, set by the test-author at activation).
TIP = "d81727e959857449e89fa5cc0f449d1b4090f590"
#: B14's range base: B's branches merged main mid-way (b4f9c42f, 3996b0c2), so the range starts at
#: merge-base(main, TIP) = 175a29b0 (.claude/rules/tests.md: BASE = merge-base M^1 M^2, not the declared fork
#: point), as A8 did. B-impl2 merged main at 37942460, so the base moved from 968e47d4 (TIP c81f1ebf) to 175a29b0
#: (TIP 2b9e7e91); round 3 (TIP d81727e9) moves it to dc7c6591. RANGE_BASE..TIP must still hold every B commit
#: (3366b27e first; the premise checks below say so).
RANGE_BASE = "dc7c65917b72f12150287e996e312338a546251f"
#: B's first branch commit (the judge rows), made before the mid-way merges of main.
B_FIRST = "3366b27e755871def95247e9e4fffbd6d9c11c03"
#: B14's list, written out from the task.
FROZEN = (
    "local_astra/computer",
    "local_astra/conductor",
    "local_astra/toolcall",
    "local_astra/contracts",
    "local_astra/cli.py",
    "local_astra/config.py",
    "local_astra/conductor/prompt_builder.py",  # B14's `prompt_builder.py` (inside conductor/ already)
    "local_astra/pages/server.py",
    "local_astra/pages/static",
    "local_astra/pages/scenarios",
    "local_astra/pages/suite",
    "local_astra/pages/handoff",
    "local_astra/pages/scenario_tasks.py",
    "tests/fakes",
)
TESTS_OWNED = frozenset(
    {
        "tests/unit/test_t0210_threejs_judge.py",
        "tests/unit/test_t0210_threejs_static.py",
        "tests/integration/test_t0210_threejs_session.py",
    }
)
#: The power range: T-0205 group B (a4ba3cae..d4ad7597) changed ``local_astra/pages/scenarios``.
POWER_BASE = "a4ba3cae01da9bb4731124e5ebb068d87140b41a"
POWER_TIP = "d4ad7597f1ee7ec2d587d37bd8977792e2dbe338"


# =====================================================================================================
# helpers
# =====================================================================================================


def tt() -> Any:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in (MODULE, "local_astra.pages"):
            raise AssertionError(f"T-0210 B1: {MODULE} does not exist yet") from exc
        raise


def threejs_dir() -> Path:
    """``threejs_tasks.THREEJS_DIR``, checked to be the real path of ``local_astra/pages/threejs/`` (B1)."""
    m = tt()
    assert hasattr(m, "THREEJS_DIR"), f"T-0210 B1: {MODULE} has no THREEJS_DIR"
    d = Path(m.THREEJS_DIR)
    want = Path(os.path.realpath(THREEJS_PATH))
    assert d == want, f"T-0210 B1: THREEJS_DIR is {d}, not the real path {want}"
    assert d.is_dir(), f"T-0210 B0: {d} does not exist yet"
    return d


def rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def copy_files(root: Path) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())
    assert files, f"T-0210 B0: {root} holds no file (does not exist yet)"
    return files


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


HEX64 = re.compile(r"^[0-9a-f]{64}$")


def provenance_rows(text: str) -> tuple[dict[str, tuple[str, str, str]], list[str]]:
    """``{path: (vendored, status, upstream)}`` from ``PROVENANCE.md``'s file table, and the malformed lines (a table
    line naming a status or a sha that does not parse, or a path listed twice)."""
    rows: dict[str, tuple[str, str, str]] = {}
    bad: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("|") or not s.endswith("|"):
            continue
        cells = [c.strip() for c in s[1:-1].split("|")]
        if len(cells) != 4:
            continue
        path, vend, status, up = cells
        path = path.strip("`").strip()
        status = status.strip("`").strip()
        vend = vend.strip("`").strip()
        up = up.strip("`").strip()
        if status not in STATUSES:
            if HEX64.match(vend):
                bad.append(line)
            continue
        if not HEX64.match(vend):
            bad.append(line)
            continue
        if status in ("unmodified", "patched") and not HEX64.match(up):
            bad.append(line)
            continue
        if status == "ours" and HEX64.match(up):
            bad.append(line)
            continue
        if path in rows:
            bad.append(line)
            continue
        rows[path] = (vend, status, up)
    return rows, bad


def provenance(root: Path) -> dict[str, tuple[str, str, str]]:
    p = root / "PROVENANCE.md"
    assert p.is_file(), "T-0210 B0: PROVENANCE.md does not exist yet"
    rows, bad = provenance_rows(p.read_text(encoding="utf-8"))
    assert bad == [], f"T-0210 B0: malformed PROVENANCE.md rows: {bad}"
    assert rows, "T-0210 B0: PROVENANCE.md has no file row"
    return rows


def is_ours(path: str) -> bool:
    return path in OURS or "astra_stubs" in path.split("/")


def upstream_dir() -> Path:
    """The r179 tree outside the repo; skips (never passes) when absent."""
    env = os.environ.get("ASTRA_THREEJS_UPSTREAM")
    d = Path(env) if env else Path(pwd.getpwuid(os.getuid()).pw_dir) / UPSTREAM_DIRNAME
    if not (d / "editor" / "index.html").is_file():
        pytest.skip(f"the upstream r179 tree is absent at {d}; the upstream row cannot run")
    return d


def upstream_head(d: Path) -> str:
    """The commit of a clone, read as text from ``.git`` (git is never run in the untrusted tree)."""
    head = (d / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:].strip()
    loose = d / ".git" / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    for line in (d / ".git" / "packed-refs").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    raise AssertionError(f"cannot read the upstream commit from {d}/.git")


# --------------------------------------------------------------------------------- index.html parsing

IMPORTMAP = re.compile(r'(?is)<script\b[^>]*type\s*=\s*["\']importmap["\'][^>]*>(.*?)</script\s*>')
MODULE_INLINE = re.compile(
    r'(?is)<script\b(?=[^>]*type\s*=\s*["\']module["\'])(?![^>]*\bsrc\s*=)[^>]*>(.*?)</script\s*>'
)


def strip_script_bodies(html: str) -> str:
    return SCRIPT_BLOCK.sub(lambda m: m.group(0)[: m.group(0).index(">") + 1] + "</script>", html)


def html_tags(html: str) -> list[str]:
    return [str(t) for t in TAG.findall(strip_script_bodies(html))]


def importmap(html: str) -> dict[str, str]:
    m = IMPORTMAP.search(html)
    if m is None:
        return {}
    data = json.loads(m.group(1))
    imports = data.get("imports", {})
    assert isinstance(imports, dict), imports
    return {str(k): str(v) for k, v in imports.items()}


def index_findings(html: str) -> list[str]:
    """B1/B-R6: absolute URL attributes on tags, absolute importmap values, a service-worker registration."""
    out: list[str] = []
    for tag in html_tags(html):
        for k, v in attrs(tag):
            if k in URL_ATTRS and ABSOLUTE.match(v):
                out.append(f"tag {k}={v!r}")
    try:
        imap = importmap(html)
    except (json.JSONDecodeError, AssertionError) as exc:
        return [*out, f"importmap does not parse: {exc}"]
    for k, v in imap.items():
        if ABSOLUTE.match(v) or v.startswith("//"):
            out.append(f"importmap {k}={v!r}")
    code = "\n".join(m.group(1) for m in SCRIPT_BLOCK.finditer(html))
    if re.search(r"\bserviceWorker\b", code):
        out.append("serviceWorker registration")
    if re.search(r"""["'/]sw\.js\b""", html):
        out.append("sw.js named")
    return out


# --------------------------------------------------------------------------------- the import closure

IMPORT = re.compile(
    r"""(?:^|[;\s}])(?:import|export)\s*(?:[\w*{}\s,$]+?\s*from\s*)?["']([^"']+)["']""", re.M
)
NAMED_IMPORT = re.compile(r"""\bimport\s*\{([^}]*)\}\s*from\s*["']([^"']+)["']""")
COMMENT = re.compile(r"/\*.*?\*/|(?<![:\\'\"])//[^\n]*", re.S)


def _resolve(spec: str, base: str, imap: dict[str, str]) -> str | None:
    """A URL path (``/editor/js/x.js``), an off-origin URL, or ``None`` for an unmapped bare specifier."""
    if ABSOLUTE.match(spec) or spec.startswith("//"):
        return spec
    if spec.startswith(("./", "../", "/")):
        return posixpath.normpath(posixpath.join(posixpath.dirname(base), spec))
    best = None
    for k in imap:
        if (spec == k or (k.endswith("/") and spec.startswith(k))) and (
            best is None or len(k) > len(best)
        ):
            best = k
    if best is None:
        return None
    v = imap[best]
    target = v + spec[len(best) :] if best.endswith("/") else v
    if ABSOLUTE.match(target) or target.startswith("//"):
        return target
    return posixpath.normpath(posixpath.join("/editor/", target))


def _url(v: str) -> str:
    if v.startswith(("/", "./", "../", "//")) or ABSOLUTE.match(v):
        return v
    return "./" + v


def closure(root: Path) -> tuple[set[str], list[str]]:
    """The URL paths ``editor/index.html`` reaches statically (classic scripts, stylesheet and icon links, the
    importmap, inline modules and every static ``import``/``export from``), and the findings: off-origin, unresolved
    or missing targets."""
    html = (root / "editor" / "index.html").read_text(encoding="utf-8")
    imap = importmap(html)
    base = "/editor/index.html"
    seen: set[str] = set()
    findings: list[str] = []
    todo: list[tuple[str, str]] = []

    def visit(target: str | None, origin: str, spec: str, follow: bool) -> None:
        if target is None:
            findings.append(f"unresolved {spec!r} in {origin}")
            return
        if not target.startswith("/"):
            findings.append(f"off-origin {spec!r} in {origin}")
            return
        p = root / target.lstrip("/")
        if not p.is_file():
            findings.append(f"missing {target} (from {origin})")
            return
        if target in seen:
            return
        seen.add(target)
        if follow and target.endswith(".js"):
            todo.append((target, p.read_text(encoding="utf-8", errors="replace")))

    for tag in html_tags(html):
        name = tag[1:].split(None, 1)[0].lower().rstrip(">")
        a = dict(attrs(tag))
        # a URL attribute is never a bare specifier: "js/x.js" is "./js/x.js"
        if name == "script" and "src" in a:
            module = a.get("type", "").lower() == "module"
            visit(_resolve(_url(a["src"]), base, imap), base, a["src"], module)
        elif name == "link" and "href" in a:
            visit(_resolve(_url(a["href"]), base, imap), base, a["href"], False)
    for k, v in imap.items():
        if not k.endswith("/"):
            visit(_resolve(k, base, imap), "importmap", f"{k} -> {v}", True)
    todo += [(base, m.group(1)) for m in MODULE_INLINE.finditer(html)]
    while todo:
        origin, text = todo.pop()
        code = COMMENT.sub(" ", text)
        for m in IMPORT.finditer(code):
            visit(_resolve(m.group(1), origin, imap), origin, m.group(1), True)
    return seen, findings


def named_imports(text: str) -> list[tuple[str, list[str]]]:
    out = []
    for m in NAMED_IMPORT.finditer(COMMENT.sub(" ", text)):
        names = [n.strip().split(" as ")[0].strip() for n in m.group(1).split(",") if n.strip()]
        out.append((m.group(2), names))
    return out


def exports(text: str) -> set[str]:
    code = COMMENT.sub(" ", text)
    names = set(
        re.findall(
            r"\bexport\s+(?:default\s+)?(?:async\s+)?(?:class|function\*?|const|let|var)\s+([\w$]+)",
            code,
        )
    )
    for m in re.finditer(r"\bexport\s*\{([^}]*)\}", code):
        for part in m.group(1).split(","):
            part = part.strip()
            if part:
                names.add(part.split(" as ")[-1].strip())
    return names


# =====================================================================================================
# B0 / B1: contents, licence, suffixes
# =====================================================================================================


def test_root_contents_and_licence() -> None:
    """B0/B1/B-R3: ``THREEJS_DIR`` is the real path of ``pages/threejs/``; the required files exist; no symlink; no
    ``sw.js`` anywhere; ``LICENSE`` is the MIT licence with upstream's copyright line; ``PROVENANCE.md`` names the
    tag, the commit, MIT, the copyright line, the patches P1-P5, B-R6's off-origin references and B-R1's ``confirm``
    finding."""
    root = threejs_dir()
    files = copy_files(root)
    names = {rel(p, root) for p in files}
    required = {
        "LICENSE",
        "PROVENANCE.md",
        "editor/index.html",
        "editor/astra_state.js",
        "editor/astra_seeds.js",
        "build/three.module.js",
        "build/three.core.js",
        *DYNAMIC_NEEDED,
    }
    assert required - names == set(), f"T-0210 B0: missing {sorted(required - names)}"
    links = [rel(p, root) for p in files if p.is_symlink()]
    assert links == [], f"T-0210 B0: symlinks in the copy: {links}"
    sw = [n for n in names if n.rsplit("/", 1)[-1] == "sw.js"]
    assert sw == [], f"T-0210 B0: sw.js is vendored: {sw}"
    lic = (root / "LICENSE").read_text(encoding="utf-8")
    assert lic.lstrip().startswith(LICENSE_FIRST) and LICENSE_COPYRIGHT in lic, (
        "T-0210 B0: LICENSE is not upstream's MIT text"
    )
    prov = (root / "PROVENANCE.md").read_text(encoding="utf-8")
    for needle in ("r179", UPSTREAM_COMMIT, "MIT", LICENSE_COPYRIGHT, "P1", "P2", "P3", "P4", "P5"):
        assert needle in prov, f"T-0210 B0: PROVENANCE.md does not name {needle!r}"
    for needle in (*OFF_ORIGIN_NAMES, "sw.js"):
        assert needle in prov, f"T-0210 B-R6: PROVENANCE.md does not list the removed {needle!r}"
    assert "confirm" in prov, "T-0210 B-R1: PROVENANCE.md does not record the confirm() finding"


def test_suffix_allow_list() -> None:
    """B1/B-R3: every file's suffix is one ``pages/server.py._TYPES`` serves, or the name is exactly ``LICENSE`` or
    ``PROVENANCE.md`` (at the root). Power: the predicate refuses ``icon.xcf``, a ``.md`` doc and ``.wasm``."""
    from local_astra.pages import server

    assert set(server._TYPES) == ALLOWED_SUFFIXES, (
        "premise: B1's suffix list is pages/server.py._TYPES"
    )

    def allowed(r: str) -> bool:
        return r in ALLOWED_NAMES or Path(r).suffix in ALLOWED_SUFFIXES

    for bad in (
        "editor/images/icon.xcf",
        "editor/docs/Writing unit tests.md",
        "a/b.wasm",
        "editor/LICENSE",
    ):
        assert not allowed(bad), bad
    root = threejs_dir()
    off = [rel(p, root) for p in copy_files(root) if not allowed(rel(p, root))]
    assert off == [], f"T-0210 B1: files outside the suffix allow-list: {off}"


# =====================================================================================================
# B0: PROVENANCE.md
# =====================================================================================================


def test_provenance_parser_has_power() -> None:
    """The row parser (not the copy): it reads a well-formed table and flags a short sha, an unknown status, an
    ``ours`` row with an upstream sha and a duplicate path."""
    a, b = "a" * 64, "b" * 64
    good = f"| file | sha | status | up |\n|---|---|---|---|\n| `x/y.js` | {a} | unmodified | {a} |\n| z.js | {b} | ours | — |\n"
    rows, bad = provenance_rows(good)
    assert rows == {"x/y.js": (a, "unmodified", a), "z.js": (b, "ours", "—")} and bad == []
    for line in (
        f"| p.js | {a[:-1]} | patched | {b} |",
        f"| p.js | {a} | changed | {b} |",
        f"| p.js | {a} | ours | {b} |",
        f"| p.js | {a} | patched | — |",
    ):
        assert provenance_rows(line)[1] == [line], line
    dup = f"| p.js | {a} | ours | — |\n| p.js | {a} | ours | — |"
    assert len(provenance_rows(dup)[1]) == 1


def test_provenance_rows_recompute() -> None:
    """B0: one row per file under the root but ``PROVENANCE.md``; each vendored sha recomputes; ``unmodified`` rows
    carry the same sha twice and ``patched`` rows two different ones; ``ours`` is exactly the shim, the seeds and the
    stubs; ``editor/index.html`` is ``patched``; ``LICENSE`` is ``unmodified``."""
    root = threejs_dir()
    rows = provenance(root)
    on_disk = {rel(p, root) for p in copy_files(root)} - {"PROVENANCE.md"}
    assert on_disk - set(rows) == set(), (
        f"T-0210 B0: files with no row: {sorted(on_disk - set(rows))}"
    )
    assert set(rows) - on_disk == set(), (
        f"T-0210 B0: rows with no file: {sorted(set(rows) - on_disk)}"
    )
    drift = [p for p, (v, _s, _u) in rows.items() if sha256(root / p) != v]
    assert drift == [], f"T-0210 B0: vendored sha256 does not recompute: {drift}"
    wrong = [
        p
        for p, (v, s, u) in rows.items()
        if (s == "unmodified" and v != u) or (s == "patched" and v == u)
    ]
    assert wrong == [], f"T-0210 B0: status disagrees with the two shas: {wrong}"
    ours = {p for p, (_v, s, _u) in rows.items() if s == "ours"}
    want_ours = {p for p in rows if is_ours(p)}
    assert ours == want_ours, f"T-0210 B0: ours is {sorted(ours)}, want {sorted(want_ours)}"
    assert ours >= OURS, f"T-0210 B3: shim/seeds not rows marked ours: {sorted(OURS - ours)}"
    assert any("astra_stubs" in p for p in ours), "T-0210 B0 P1: no astra_stubs file"
    assert rows["editor/index.html"][1] == "patched", rows["editor/index.html"]
    assert rows["LICENSE"][1] == "unmodified", rows["LICENSE"]


def test_layout_files_unmodified() -> None:
    """B2: the layout files B2's checks are about are vendored ``unmodified`` (the vendored sha recomputes and equals
    the upstream column, checked against the upstream tree by the next row)."""
    root = threejs_dir()
    rows = provenance(root)
    missing = [p for p in LAYOUT if p not in rows]
    assert missing == [], f"T-0210 B2: layout files not vendored: {missing}"
    changed = [p for p in LAYOUT if rows[p][1] != "unmodified" or rows[p][0] != rows[p][2]]
    assert changed == [], f"T-0210 B2: layout files patched: {changed}"
    drift = [p for p in LAYOUT if sha256(root / p) != rows[p][0]]
    assert drift == [], drift


def test_upstream_shas_and_coverage() -> None:
    """B0 against the r179 tree (read-only; skipped, naming the path, when absent): the tree is at the pinned commit;
    every ``unmodified``/``patched`` row's upstream sha recomputes from the same path upstream and ``unmodified``
    files are byte-equal; no ``ours`` path exists upstream; every upstream ``editor/**`` file with an allowed suffix
    but ``sw.js`` is vendored (B-R3); ``LICENSE`` is byte-equal to upstream's."""
    up = upstream_dir()
    assert upstream_head(up) == UPSTREAM_COMMIT, f"premise: {up} is not at {UPSTREAM_COMMIT}"
    root = threejs_dir()
    rows = provenance(root)
    bad: list[str] = []
    for p, (_v, s, u) in sorted(rows.items()):
        target = up / p
        if s == "ours":
            if target.exists():
                bad.append(f"{p}: ours but exists upstream")
            continue
        if not target.is_file():
            bad.append(f"{p}: {s} but absent upstream")
            continue
        if sha256(target) != u:
            bad.append(f"{p}: upstream sha column is not the upstream file's")
        if s == "unmodified" and target.read_bytes() != (root / p).read_bytes():
            bad.append(f"{p}: unmodified but bytes differ")
    assert bad == [], f"T-0210 B0: {bad}"
    want = {
        rel(p, up)
        for p in (up / "editor").rglob("*")
        if p.is_file() and p.suffix in ALLOWED_SUFFIXES and p.name != "sw.js"
    }
    missing = sorted(want - set(rows))
    assert missing == [], f"T-0210 B0/B-R3: upstream editor files not vendored: {missing}"
    assert (root / "LICENSE").read_bytes() == (up / "LICENSE").read_bytes()


# =====================================================================================================
# B1 / B-R6: index.html
# =====================================================================================================

INDEX_MUTANTS = (
    (
        '<script src="js/libs/esprima.js"></script>',
        '<script src="https://cdn.example/x.js"></script>',
    ),
    (
        '<link rel="manifest" href="manifest.json">',
        '<link rel="manifest" href="//cdn.example/m.json">',
    ),
    ('"imports": {', '"imports": { "three-mesh-bvh": "https://cdn.jsdelivr.net/npm/x.js",'),
    ("</body>", "<script>navigator.serviceWorker.register( 'sw.js' );</script></body>"),
)


def test_index_html_has_no_absolute_reference() -> None:
    """B1/B-R6 (P1-P3, P5): no ``index.html`` tag URL attribute and no importmap value is absolute; no service-worker
    registration and no ``sw.js``; exactly one ``<script type="module" src="astra_state.js">``. Power: each of
    ``INDEX_MUTANTS`` (an off-origin script, a protocol-relative link, an https importmap entry, a service-worker
    registration) seeded into the vendored file is caught."""
    root = threejs_dir()
    html = (root / "editor" / "index.html").read_text(encoding="utf-8")
    assert importmap(html), "T-0210 B1: index.html has no importmap"
    found = index_findings(html)
    assert found == [], f"T-0210 B1/B-R6: absolute references in index.html: {found}"
    shim = [
        t
        for t in html_tags(html)
        if t.lower().startswith("<script")
        and dict(attrs(t)).get("src") in ("astra_state.js", "./astra_state.js")
        and dict(attrs(t)).get("type", "").lower() == "module"
    ]
    assert len(shim) == 1, f"T-0210 B0 P5: astra_state.js module tags: {shim}"
    for anchor, mutant in INDEX_MUTANTS:
        if anchor in html:
            assert index_findings(html.replace(anchor, mutant, 1)), mutant
    assert index_findings(html.replace("</body>", INDEX_MUTANTS[3][1], 1)), "power: sw"
    seeded = html.replace(
        "</head>", '<link rel="icon" href="https://threejs.org/files/favicon.ico"></head>', 1
    )
    assert index_findings(seeded), "power: an absolute favicon"


def test_stubs_export_consumed_names() -> None:
    """B0 P1 (H-T210-3): ``three-gpu-pathtracer`` maps to ``astra_stubs/three-gpu-pathtracer.js`` in the copy;
    every name a vendored module imports from a stubbed specifier is exported by the stub; ``WebGLPathTracer`` is
    among them (``Viewport.Pathtracer.js``'s import at r179). B-R8: ``three-mesh-bvh`` is not mapped, no stub file
    carries its name, and no vendored module imports it."""
    root = threejs_dir()
    imap = importmap((root / "editor" / "index.html").read_text(encoding="utf-8"))
    assert "three-gpu-pathtracer" in imap, "T-0210 B0 P1: three-gpu-pathtracer is not mapped"
    for name in REMOVED_SPECIFIERS:
        assert name not in imap, f"T-0210 B-R8: {name} is still mapped to {imap[name]!r}"
        stray = [rel(p, root) for p in root.rglob(f"{name}*") if "astra_stubs" in p.parts]
        assert stray == [], f"T-0210 B-R8: a stub for the removed {name}: {stray}"
        importers = [
            rel(f, root)
            for f in root.rglob("*.js")
            if any(
                spec == name
                for spec, _ in named_imports(f.read_text(encoding="utf-8", errors="replace"))
            )
        ]
        assert importers == [], f"T-0210 B-R8: {name} is imported by {importers}"
    stubs: dict[str, Path] = {}
    for name in STUBBED:
        if name not in imap:
            continue
        target = _resolve(name, "/editor/index.html", imap)
        assert target is not None and target.startswith("/"), (name, target)
        parts = target.split("/")
        assert parts[-1] == f"{name}.js" and "astra_stubs" in parts, (
            f"T-0210 B0 P1: {name} maps to {target}, not astra_stubs/{name}.js"
        )
        p = root / target.lstrip("/")
        assert p.is_file(), f"T-0210 B0 P1: {target} does not exist yet"
        stubs[name] = p
    consumed: dict[str, set[str]] = {n: set() for n in stubs}
    for f in root.rglob("*.js"):
        for spec, names in named_imports(f.read_text(encoding="utf-8", errors="replace")):
            if spec in consumed:
                consumed[spec].update(names)
    assert "WebGLPathTracer" in consumed["three-gpu-pathtracer"], (
        "premise: Viewport.Pathtracer.js imports WebGLPathTracer"
    )
    for name, p in stubs.items():
        have = exports(p.read_text(encoding="utf-8"))
        missing = consumed[name] - have
        assert missing == set(), f"T-0210 B0 P1: {p.name} does not export {sorted(missing)}"


def icon_links(html: str) -> list[str]:
    """The ``href`` of every ``<link>`` whose ``rel`` has the token ``icon`` (``apple-touch-icon`` is another
    token)."""
    out: list[str] = []
    for t in html_tags(html):
        if not t.lower().startswith("<link"):
            continue
        a = dict(attrs(t))
        if "icon" in a.get("rel", "").lower().split():
            out.append(a.get("href", ""))
    return out


def test_favicon_is_the_local_icon() -> None:
    """B-R7 (amended): the patched ``index.html`` has exactly one ``rel="icon"`` link, ``href="images/icon.png"``,
    and that file is in the copy and ``unmodified`` in ``PROVENANCE.md`` (the vendored upstream icon, not a file of
    ours). No ``favicon`` and no ``data:`` URL is named. Power: upstream's ``../files/favicon.ico`` link and the old
    ``data:,`` literal are both told apart from the local icon."""
    root = threejs_dir()
    html = (root / "editor" / "index.html").read_text(encoding="utf-8")
    links = icon_links(html)
    assert links == ["images/icon.png"], f"T-0210 B-R7: rel=icon links in index.html: {links}"
    assert (root / "editor" / "images" / "icon.png").is_file(), (
        "T-0210 B-R7: images/icon.png is absent"
    )
    status = provenance(root).get("editor/images/icon.png")
    assert status is not None and status[1] == "unmodified", f"T-0210 B-R7: icon.png row {status}"
    assert "favicon" not in html.lower(), "T-0210 B-R7: index.html still names a favicon"
    assert 'href="data:' not in html, "T-0210 B-R7: a data: favicon (B1 refuses scheme URLs)"
    for bad in ('<link rel="icon" href="data:,">', '<link rel="icon" href="../files/favicon.ico">'):
        assert icon_links(bad) != ["images/icon.png"], f"power: {bad}"
    assert icon_links('<link rel="apple-touch-icon" href="images/icon.png">') == [], "width"


def test_export_reader_has_power() -> None:
    """The export reader (not the copy): class, function, const and ``export { a as b }`` forms; a comment does not
    count."""
    src = (
        "export class WebGLPathTracer {}\nexport function f() {}\nexport const C = 1;\n"
        "const x = 1; export { x as Y };\n// export class Ghost {}\n"
    )
    assert exports(src) == {"WebGLPathTracer", "f", "C", "Y"}


# =====================================================================================================
# B3 / B14: the shim, seeds and stubs are deterministic and never touch the network
# =====================================================================================================

OURS_BANS = re.compile(
    r"\bDate\b|\bMath\s*\.\s*random\b|\bsetTimeout\b|\bsetInterval\b|\brequestAnimationFrame\b"
    r"|\brequestIdleCallback\b|\bfetch\s*\(|\bXMLHttpRequest\b|\bWebSocket\b|\bEventSource\b"
    r"|\bsendBeacon\b|\bWorker\s*\(|\bimport\s*\(|://"
)
OURS_MUTANTS = (
    "const t = Date.now();",
    "new Date();",
    "const r = Math.random();",
    "setTimeout(f, 10);",
    "setInterval(f, 10);",
    "requestAnimationFrame(f);",
    "requestIdleCallback(f);",
    "fetch('seeds.json');",
    "window.fetch ('x');",
    "new XMLHttpRequest();",
    "new WebSocket('ws://127.0.0.1');",
    "new EventSource('/e');",
    "navigator.sendBeacon('/b');",
    "new Worker('w.js');",
    "await import('./x.js');",
    "const u = 'https://threejs.org';",
)
OURS_BENIGN = (
    "editor.signals.sceneGraphChanged.add(update);",
    "const seed = new URLSearchParams(location.search).get('seed');",
    "await editor.fromJSON(json);",
    "export const SEEDS = { empty: null };",
    "import { SEEDS } from './astra_seeds.js';",
)


def test_ours_determinism_and_network_bans() -> None:
    """B3/B14: the shim, the seeds and the stubs contain no ``Date``, ``Math.random``, timers, ``fetch``, XHR,
    WebSocket, EventSource, beacons, workers, ``import()`` or ``://`` (a text rule; prose in a comment trips it).
    Power: each ``OURS_MUTANTS`` appended to each file is caught; width: ``OURS_BENIGN`` is not."""
    root = threejs_dir()
    ours = [p for p in copy_files(root) if is_ours(rel(p, root))]
    assert {rel(p, root) for p in ours} >= OURS, "T-0210 B3: the shim or the seeds do not exist yet"
    dirty = {
        rel(p, root): [m.group(0) for m in OURS_BANS.finditer(p.read_text(encoding="utf-8"))]
        for p in ours
    }
    dirty = {k: v for k, v in dirty.items() if v}
    assert dirty == {}, f"T-0210 B3/B14 (a text rule; prose can trip it): {dirty}"
    for p in ours:
        text = p.read_text(encoding="utf-8")
        for snippet in OURS_MUTANTS:
            assert OURS_BANS.search(text + "\n" + snippet), (p.name, snippet)
        for snippet in OURS_BENIGN:
            assert not OURS_BANS.search(text + "\n" + snippet), (p.name, snippet)


def test_shim_reads_the_seed_and_sets_ready() -> None:
    """B3 (static half; the browser rows check behaviour): the shim reads ``seed`` from the query, defines
    ``__state`` on ``window`` as a getter and sets ``ready``; the seeds module names ``empty`` and ``crate``."""
    root = threejs_dir()
    shim = (root / "editor" / "astra_state.js").read_text(encoding="utf-8")
    seeds = (root / "editor" / "astra_seeds.js").read_text(encoding="utf-8")
    assert "__state" in shim and re.search(r"\bget\b", shim), "T-0210 B3: no __state getter"
    assert "seed" in shim and "ready" in shim, "T-0210 B3: the shim names no seed/ready"
    assert re.search(r"\bempty\b", seeds) and re.search(r"\bcrate\b", seeds), "T-0210 B3: seeds"
    assert "Crate" in seeds, "T-0210 B3: the crate seed's object is not named Crate"


# =====================================================================================================
# B0 / B-R6: the static import closure
# =====================================================================================================


def test_closure_reader_has_power(tmp_path: Path) -> None:
    """The closure reader (not the copy), on a hand-made miniature: it follows a classic script, a link, the
    importmap, an inline module and a chain of static imports, and reports an off-origin importmap entry, an
    off-origin script, a missing module and an unmapped bare specifier."""
    (tmp_path / "editor" / "js").mkdir(parents=True)
    (tmp_path / "build").mkdir()
    (tmp_path / "editor" / "index.html").write_text(
        '<link rel="stylesheet" href="css/main.css">\n'
        '<script type="importmap">{"imports": {"three": "../build/three.module.js",'
        ' "bvh": "https://cdn.example/bvh.js"}}</script>\n'
        '<script src="js/classic.js"></script>\n'
        '<script src="https://cdn.example/ffmpeg.js"></script>\n'
        "<script type=\"module\">\nimport { Editor } from './js/Editor.js';\n</script>\n",
        encoding="utf-8",
    )
    (tmp_path / "editor" / "css").mkdir()
    (tmp_path / "editor" / "css" / "main.css").write_text("", encoding="utf-8")
    (tmp_path / "editor" / "js" / "classic.js").write_text("var a = 1;", encoding="utf-8")
    (tmp_path / "editor" / "js" / "Editor.js").write_text(
        "import * as THREE from 'three';\nimport { X } from './Gone.js';\nimport 'lonely';\n"
        "// import { Y } from './Commented.js';\n",
        encoding="utf-8",
    )
    (tmp_path / "build" / "three.module.js").write_text("export const R = 1;", encoding="utf-8")
    seen, findings = closure(tmp_path)
    assert seen == {
        "/editor/css/main.css",
        "/editor/js/classic.js",
        "/editor/js/Editor.js",
        "/build/three.module.js",
    }, seen
    joined = "\n".join(findings)
    assert "off-origin 'bvh -> https://cdn.example/bvh.js'" in joined, findings
    assert "off-origin 'https://cdn.example/ffmpeg.js'" in joined, findings
    assert "missing /editor/js/Gone.js" in joined, findings
    assert "unresolved 'lonely'" in joined, findings
    assert "Commented" not in joined, findings
    assert len(findings) == 4, findings


def test_readers_see_upstream_off_origin_references() -> None:
    """Power on the real input (skipped when the r179 tree is absent): over upstream's untouched ``editor/``, the
    closure and ``index.html`` readers report exactly B-R6's list: the two ``../files`` favicons (missing), the
    ffmpeg script, the ``three-gpu-pathtracer`` and ``three-mesh-bvh`` importmap entries (and the pathtracer's
    consumer), the service-worker registration and ``sw.js``. Expected findings typed from the upstream file."""
    up = upstream_dir()
    assert upstream_head(up) == UPSTREAM_COMMIT
    seen, findings = closure(up)
    joined = "\n".join(findings)
    for needle in (
        "missing /files/favicon_white.ico",
        "missing /files/favicon.ico",
        "ffmpeg.min.js",
        "three-gpu-pathtracer -> https:",
        "three-mesh-bvh -> https:",
        "'three-gpu-pathtracer' in /editor/js/Viewport.Pathtracer.js",
    ):
        assert needle in joined, (needle, findings)
    assert len(findings) == 6, findings
    assert "/build/three.core.js" in seen and "/examples/jsm/controls/TransformControls.js" in seen
    found = index_findings((up / "editor" / "index.html").read_text(encoding="utf-8"))
    assert len(found) == 5 and "serviceWorker registration" in found and "sw.js named" in found, (
        found
    )


def test_import_closure_resolves_locally() -> None:
    """B0/B-R6: everything ``editor/index.html`` reaches statically resolves to a file in the copy and nothing is
    off-origin; the closure includes ``build/three.module.js``, ``build/three.core.js``, ``Editor.js`` and the shim;
    the golden paths' dynamic geometry panels exist."""
    root = threejs_dir()
    seen, findings = closure(root)
    assert findings == [], f"T-0210 B0/B-R6: the import closure leaves the copy: {findings}"
    for need in (
        "/build/three.module.js",
        "/build/three.core.js",
        "/editor/js/Editor.js",
        "/editor/astra_state.js",
        "/editor/js/Viewport.Pathtracer.js",
    ):
        assert need in seen, f"T-0210 B0: {need} is not reached from index.html"
    missing = [p for p in DYNAMIC_NEEDED if not (root / p).is_file()]
    assert missing == [], missing
    print(f"T-0210 B0: static closure {len(seen)} files")


# =====================================================================================================
# B1: separate root, 404s both ways
# =====================================================================================================


def _other_roots() -> dict[str, Path]:
    from local_astra.pages.rtx_tasks import RTX_DIR
    from local_astra.pages.scenario_tasks import SCENARIO_DIR
    from local_astra.pages.server import STATIC_DIR
    from local_astra.pages.suite_tasks import SUITE_DIR

    return {
        "default": Path(os.path.realpath(STATIC_DIR)),
        "scenarios": Path(SCENARIO_DIR),
        "suite": Path(SUITE_DIR),
        "rtx": Path(RTX_DIR),
    }


def test_other_roots_404_threejs_paths() -> None:
    """B1: the default, scenarios, suite and rtx roots answer 404 for every threejs path (``server.resolve`` for all,
    real servers for a spot set), and the threejs root serves each of its files with their bytes (spot set)."""
    from local_astra.pages.server import resolve

    root = threejs_dir()
    paths = url_paths(root)
    spot = [
        p
        for p in (
            "/editor/index.html",
            "/editor/",
            "/LICENSE",
            "/build/three.module.js",
            "/editor/astra_state.js",
        )
        if p in paths
    ]
    assert "/editor/index.html" in paths, paths[:5]
    for name, other in _other_roots().items():
        leaked = [p for p in paths if resolve(other, p) is not None]
        assert leaked == [], f"T-0210 B1: the {name} root resolves threejs paths: {leaked[:10]}"
        srv = new_server(other)
        srv.start()
        try:
            port = port_of(srv.origin)
            got = {p: request(port, "GET", p)[0] for p in spot}
        finally:
            srv.stop()
        assert set(got.values()) == {404}, (name, got)
    srv = new_server(root)
    srv.start()
    try:
        port = port_of(srv.origin)
        for p in spot:
            status, body, _ = request(port, "GET", p)
            target = root / (p.lstrip("/") + ("index.html" if p.endswith("/") else ""))
            assert (status, body) == (200, target.read_bytes()), (p, status)
    finally:
        srv.stop()


def test_threejs_root_404_other_paths() -> None:
    """B1: the threejs root answers 404 for every path the other roots serve (``/`` included: the copy has no root
    ``index.html``), through ``server.resolve`` and a real server for a spot set; traversal spellings never answer
    200."""
    from local_astra.pages.server import resolve

    root = threejs_dir()
    every: list[str] = []
    for other in _other_roots().values():
        every += url_paths(other)
    leaked = sorted({p for p in every if resolve(root, p) is not None})
    assert leaked == [], f"T-0210 B1: the threejs root resolves other roots' paths: {leaked[:10]}"
    spot = ["/", "/index.html", "/web.html", "/search.html"]
    traversal = [
        "/../static/index.html",
        "/%2e%2e/static/index.html",
        "/..%2fserver.py",
        "/editor/../../server.py",
    ]
    srv = new_server(root)
    srv.start()
    try:
        port = port_of(srv.origin)
        for p in spot:
            assert request(port, "GET", p)[0] == 404, p
        for p in traversal:
            assert request(port, "GET", p)[0] != 200, p
    finally:
        srv.stop()


def test_no_other_root_references_threejs() -> None:
    """B1: no file of the default, scenarios, suite or rtx roots names ``threejs`` or ``three.js``' editor path."""
    threejs_dir()
    hits = []
    for name, other in _other_roots().items():
        for p in other.rglob("*"):
            if p.is_file() and re.search(
                r"threejs|/editor/index\.html|astra_state",
                p.read_text(encoding="utf-8", errors="replace"),
            ):
                hits.append(f"{name}:{rel(p, other)}")
    assert hits == [], f"T-0210 B1: other roots reference the threejs copy: {hits}"


# =====================================================================================================
# B-R4, B12
# =====================================================================================================


def test_size_cap() -> None:
    """B0/B-R4/H-T210-8: the copy's total size is at most 12 x 10^6 bytes (decimal); the size is printed."""
    root = threejs_dir()
    total = sum(p.stat().st_size for p in copy_files(root))
    print(f"T-0210 B0: threejs root {total} bytes")
    assert 0 < total <= SIZE_CAP, f"T-0210 B-R4: the copy is {total} bytes, over {SIZE_CAP}"


def test_make_browser_threejs_target() -> None:
    """B12: ``make -n -s browser-threejs`` prints exactly B12's command and the target is ``.PHONY`` (written by the
    orchestrator at merge; red until then)."""
    code, lines, err = _make_lines("browser-threejs")
    assert code == 0, f"T-0210 B12: no browser-threejs target (does not exist yet): {err}"
    assert lines == [MAKE_BROWSER_THREEJS], lines
    phony = [ln for ln in (REPO / "Makefile").read_text().splitlines() if ln.startswith(".PHONY:")]
    assert any("browser-threejs" in ln.split() for ln in phony), phony


# =====================================================================================================
# B14 (green at BASE)
# =====================================================================================================


def _makefile_at(rev: str) -> list[str]:
    proc = gr._git("show", f"{rev}:Makefile")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


def test_frozen_paths_over_closed_range() -> None:
    """B14: B14's paths are byte-identical over ``RANGE_BASE..TIP`` (two literals, never the working tree), each
    exists at RANGE_BASE; under ``tests/`` only B's three modules change; the Makefile changes only by lines naming
    ``browser-threejs``. ``RANGE_BASE`` is ``merge-base(main, TIP)`` because B's branches merged main mid-way.
    Premises: BASE precedes RANGE_BASE, and B's first commit lies in the range. Power: the same predicate sees
    T-0205 B's change to ``local_astra/pages/scenarios``."""
    gr.require_commits(BASE, RANGE_BASE, B_FIRST, TIP)
    assert gr.is_ancestor(BASE, RANGE_BASE), f"{BASE[:7]} is not an ancestor of {RANGE_BASE[:7]}"
    assert gr.is_ancestor(RANGE_BASE, TIP), f"{RANGE_BASE[:7]} is not an ancestor of {TIP[:7]}"
    assert gr.is_ancestor(B_FIRST, TIP) and not gr.is_ancestor(B_FIRST, RANGE_BASE), (
        f"premise: {B_FIRST} lies in {RANGE_BASE[:7]}..{TIP[:7]}"
    )
    listed = gr._git("ls-tree", "-r", "--name-only", RANGE_BASE, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    changed = gr.changed(RANGE_BASE, TIP, *FROZEN)
    assert changed == [], (
        f"T-0210 B14: frozen paths differ in {RANGE_BASE[:7]}..{TIP[:7]}: {changed}"
    )
    tests_changed = [p for p in gr.changed(RANGE_BASE, TIP, "tests") if p not in TESTS_OWNED]
    assert tests_changed == [], f"T-0210 B14: other test files edited: {tests_changed}"
    assert set(gr.changed(RANGE_BASE, TIP, "tests")) == TESTS_OWNED, (
        "premise: B's three modules are in range"
    )
    before, after = _makefile_at(RANGE_BASE), _makefile_at(TIP)
    removed = [ln for ln in before if ln not in after]
    added = [ln for ln in after if ln not in before]
    assert all(ln.startswith(".PHONY:") for ln in removed), removed
    assert all("browser-threejs" in ln for ln in added), added
    gr.require_commits(POWER_BASE, POWER_TIP)
    assert gr.changed(POWER_BASE, POWER_TIP, "local_astra/pages/scenarios"), (
        "power: the predicate cannot see a change"
    )


# =====================================================================================================
# B-R14 (patch P4b): tern.js opens nothing off-origin; the unpatched sites are recorded
# =====================================================================================================

TERN_FILES = ("editor/js/libs/codemirror/addon/tern.js", "editor/js/libs/ternjs/tern.js")
#: ``window.open(data.url)`` (any spacing) and the ``[docs]`` anchor's ``href = data.url``, in code (comments dropped).
TERN_OPEN = re.compile(r"window\s*\.\s*open\s*\(\s*data\s*\.\s*url\b")
TERN_HREF = re.compile(r"\.\s*href\s*=\s*data\s*\.\s*url\b")


def _not_patched_section(text: str) -> str:
    """The text from the line starting "Not patched, recorded" to the next ``## `` heading."""
    i = text.find("Not patched, recorded")
    assert i >= 0, "T-0210 B-R14: PROVENANCE.md has no 'Not patched, recorded' passage"
    j = text.find("\n## ", i)
    return text[i : j if j >= 0 else len(text)]


def test_r14_tern_opens_nothing_and_unpatched_sites_recorded() -> None:
    """B-R14: the copy's tern.js holds no ``window.open(data.url)`` call and no ``[docs]`` anchor whose ``href`` is
    ``data.url`` (comments are stripped first, so a quoted line in a comment does not count); the CodeMirror addon
    ``tern.js`` is ``patched`` in PROVENANCE.md's table and the patch is named P4b; the "Not patched, recorded"
    passage names XRControllerModelFactory's jsdelivr profiles path and ``Sidebar.Project.Image``'s
    ``window.open('')``. Power: the regexes see upstream's literal line."""
    assert TERN_OPEN.search("if (!data.file && data.url) { window.open(data.url); return; }"), (
        "power"
    )
    assert TERN_HREF.search("child.href = data.url;"), "power"
    root = threejs_dir()
    for name in TERN_FILES:
        code = COMMENT.sub("", (root / name).read_text(encoding="utf-8"))
        assert not TERN_OPEN.search(code), f"T-0210 B-R14: {name} still calls window.open(data.url)"
        assert not TERN_HREF.search(code), f"T-0210 B-R14: {name} still links [docs] to data.url"
    rows = provenance(root)
    assert rows[TERN_FILES[0]][1] == "patched", rows[TERN_FILES[0]]
    text = (root / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "P4b" in text, "T-0210 B-R14: PROVENANCE.md names no patch P4b"
    rec = _not_patched_section(text)
    assert "XRControllerModelFactory" in rec and "jsdelivr" in rec, rec
    assert "Sidebar.Project.Image" in rec and "window.open" in rec, rec
