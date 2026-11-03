"""T-0209 group A1, criteria A1.1-A1.5 (static half), A1.8, A1.15, A1.16 (``cli``) and A1.17: the rtx page root read
as files, the three roots' 404s, ``ROUTES`` parity, the R1 image, the ``browser-rtx`` Make target and the frozen paths
(docs/tasks/T-0209.md, "Tests to write", ``tests/unit/test_t0209_rtx_static.py``). A2's ``R2_FILES`` row is group A2's.

**What these rows read.** ``rtx_tasks.RTX_DIR`` (through :func:`rtx_dir`, a named ``AssertionError`` while the module
or directory is absent), the files under it, ``rtx_shared/``, ``pages.server.resolve``, three real loopback
``StartPageServer``s, Pillow's decoder, ``make -n -s`` and ``git diff`` over literal commits. No browser.

**The static egress and determinism rules.** A1.3's bans are T-0205 A3/A8's rules with A1.3's changes:

- ``http:``/``https:``/``://`` (and ``javascript:``/``data:``/``blob:``) anywhere in a file, protocol-relative URLs,
  absolute URL attributes and CSS URLs, network APIs (``fetch``, XHR, WebSocket, beacons, ``window.open``,
  ``import()``, any ``location``/``history`` navigation), embeds and link hints, as T-0205;
- ``location.assign(`` only in ``web.html`` (A1.2); anywhere else is a finding;
- forms: a ``method`` other than ``get`` (on ``<form>`` or as ``formmethod``) or an absolute ``action`` is a finding
  (T-0205 banned every form);
- host names (T-0205's ``HOSTNAME`` rule) are allowed in visible text, in ``placeholder`` attributes and in
  ``web.html``'s script (its route table and the live page's hint strings); in any other attribute, in any other
  script (``assets/rtx.js`` included) or in style code they are findings;
- no ``contenteditable``, ``<textarea>`` or ``<canvas>`` (A1.3 / "What does not change": every text field is a real
  ``<input>``);
- determinism: T-0205's ``determinism_findings`` (``Date``, ``Math.random``, timers, dialogs, transition, animation,
  ``@font-face``, ``<select>``, date inputs, prefilled text inputs), imported.

Power: each mutant seeded in memory into every real page is caught by its own rule (the three the task names: an
``https://`` href, a ``location.assign`` outside ``web.html``, ``<form method="post">``); width: benign snippets
add no finding.

**Readings this module pins where the task is silent** (reported):

- A1's required files are ``web.html``, ``assets/rtx.css``, ``assets/rtx.js`` and the ten ``ROUTES`` targets of A1.2;
  ``forum-login`` is any page whose ``data-page-id`` is ``forum-login``. Each ``.html`` file carries exactly one
  ``data-page-id`` value: ``web`` for ``web.html``, the A1.4 ids for the ``ROUTES`` targets (``PAGE_OF_PATH``).
- ``web.html``'s in-page table is written in its script as string pairs ``"<key>": "/<path>"`` or
  ``["<key>", "/<path>"]`` (a key holds a dot, a path starts with ``/``); the row parses every such pair.
- ``ROUTES`` holds A1.2's pinned keys with their paths (A2 adds keys); every key is already normalised (A1.2's rule
  is the identity on it); every value resolves to a file in the root.
- The footer names "Blender" on every ``blender-*`` page.
- The image's generator has a top-level ``r1()``; its string constants include A1.8's texts and name no GPU
  (``rtx``, ``geforce``, ``quadro``, ``radeon``, ``blackwell``, ``5090``).
- "The rtx root returns 404 for every ``pages/scenarios/`` path" excludes ``/web.html`` and ``/`` (both roots may
  serve their own); the default and scenarios roots answer 404 for every rtx path but ``/web.html``, which they serve
  with different bytes.
- A1.17's TIP is a literal: BASE until A1's implementation lands, then bumped to its last implementation commit.
  Under ``tests/`` only A1's five modules and ``test_guards_use_fixed_refs.py`` may change; the Makefile may change
  only by lines naming ``browser-rtx``.

BASE = 76b98997 (main at branch time), TIP = c529fc2f (A1's implementation). Activated at A1's ``/implement``: no row
carries ``pending`` any more.

Regime: CPU, files and loopback sockets only, n = 1.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import io
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any

from tests import _git_range as gr
from tests.unit.test_start_pages import new_server, port_of, request
from tests.unit.test_t0205_scenario_static import (
    ABSOLUTE,
    CSS_IMPORT,
    CSS_URL,
    DET_BENIGN,
    DET_MUTANTS,
    EMBEDS,
    HOSTNAME,
    LINK_HINT,
    PROTO_RELATIVE,
    SCRIPT_BLOCK,
    STYLE_BLOCK,
    TAG,
    URL_ATTRS,
    _make_lines,
    _seed,
    attrs,
    determinism_findings,
    script_code,
    style_code,
    url_paths,
)
from tests.unit.test_t0208_handoff_static import sof_size

REPO = Path(__file__).resolve().parents[2]
PAGES = REPO / "local_astra" / "pages"
RTX_PATH = PAGES / "rtx"
RTX_SHARED = PAGES / "rtx_shared"
STATIC = PAGES / "static"
SUITE = PAGES / "suite"
SCENARIOS = PAGES / "scenarios"
HANDOFF = PAGES / "handoff"
SCRIPT = REPO / "scripts" / "make_t0209_shared_images.py"
MODULE = "local_astra.pages.rtx_tasks"
SEARCH_MODULE = "local_astra.pages.rtx_search"
ALLOWED_SUFFIXES = frozenset({".html", ".css", ".js"})
#: A1.2, typed from the task
PINNED_ROUTES = {
    "docs.blender.org": "/blender-manual/",
    "docs.blender.org/manual/en/4.2/render/cycles/gpu_rendering.html": "/blender-manual/gpu-rendering.html",
    "docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html": "/blender-manual/gpu-rendering.html",
    "projects.blender.org": "/blender-projects/",
    "projects.blender.org/blender/blender/issues/134001": "/blender-projects/issue-134001.html",
    "developer.blender.org": "/blender-notes/",
    "developer.blender.org/docs/release_notes/4.4/cycles": "/blender-notes/4.4-cycles.html",
    "blender.org": "/blender-notes/download.html",
    "blender.org/download": "/blender-notes/download.html",
    "blenderartists.org": "/blender-artists/",
    "blenderartists.org/t/rtx-5090-not-showing-in-cycles-render-devices/1580001": "/blender-artists/thread-rtx-5090.html",
    "blenderartists.org/t/cycles-cuda-error-out-of-memory/1570002": "/blender-artists/thread-oom.html",
}
#: A1.2 paths -> A1.4 page ids
PAGE_OF_PATH = {
    "/web.html": "web",
    "/blender-manual/": "manual-home",
    "/blender-manual/gpu-rendering.html": "manual-gpu",
    "/blender-projects/": "projects-home",
    "/blender-projects/issue-134001.html": "issue-blackwell",
    "/blender-notes/": "notes-home",
    "/blender-notes/4.4-cycles.html": "notes-4-4",
    "/blender-notes/download.html": "download",
    "/blender-artists/": "forum-home",
    "/blender-artists/thread-rtx-5090.html": "forum-thread",
    "/blender-artists/thread-oom.html": "forum-oom",
}
REQUIRED_FILES = (
    "web.html",
    "assets/rtx.css",
    "assets/rtx.js",
    "blender-manual/index.html",
    "blender-manual/gpu-rendering.html",
    "blender-projects/index.html",
    "blender-projects/issue-134001.html",
    "blender-notes/index.html",
    "blender-notes/4.4-cycles.html",
    "blender-notes/download.html",
    "blender-artists/index.html",
    "blender-artists/thread-rtx-5090.html",
    "blender-artists/thread-oom.html",
)
FOOTER = re.compile(
    r"Demo copy for testing; not affiliated with ([^.<]+)\. Facts on this page are invented\."
)
R1_IMAGE = "r1_blender_error.jpg"
W, H = 1280, 800
QUALITY = 85
#: A1.8's texts, typed from the task
R1_TEXTS = (
    "Blender 4.2.3 LTS",
    "Blender Preferences",
    "System",
    "Cycles Render Devices",
    "No compatible GPUs found for Cycles",
    "CUDA binary kernel for this graphics card compute capability (12.0) not found.",
)
R1_TABS = ("None", "CUDA", "OptiX", "HIP", "oneAPI")
GPU_NAME = re.compile(r"(?i)\brtx\b|geforce|quadro|radeon|blackwell|5090")
MAKE_BROWSER_RTX = (
    "ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0209_rtx_browser.py "
    "tests/integration/test_t0209_rtx_session.py -q -m browser"
)
NETWORK_MODULES = frozenset({"socket", "urllib", "http", "httpx", "requests", "ddgs", "aiohttp"})

#: T-0209 group A1's BASE: main at 76b98997 ("T-0208 B merged; status"), the commit the branch was reset onto.
BASE = "76b989972da51d4675f222feaeaff552c7ea6aad"
#: A1's implementation tip: c529fc2f, A1's last implementation commit (BASE..TIP touches only A1's owned files and
#: the five T-0209 test modules; checked with ``git diff --stat`` at activation).
TIP = "c529fc2f7f5557119e07576ff1ffe1b34bdc86a9"
#: A1.17's list, written out from the task.
FROZEN = (
    "local_astra/pages/server.py",
    "local_astra/pages/__init__.py",
    "local_astra/pages/static",
    "local_astra/pages/suite",
    "local_astra/pages/scenarios",
    "local_astra/pages/scenario_tasks.py",
    "local_astra/pages/handoff",
    "local_astra/pages/handoff_tasks.py",
    "local_astra/conductor",
    "local_astra/computer",
    "local_astra/toolcall",
    "local_astra/search",
    "local_astra/cli.py",
    "local_astra/config.py",
    "local_astra/contracts",
    "tests/fakes",
)
TESTS_OWNED = frozenset(
    {
        "tests/unit/test_t0209_rtx_judge.py",
        "tests/unit/test_t0209_rtx_static.py",
        "tests/unit/test_t0209_rtx_search.py",
        "tests/integration/test_t0209_rtx_browser.py",
        "tests/integration/test_t0209_rtx_session.py",
        "tests/unit/test_guards_use_fixed_refs.py",
    }
)
#: The power range: T-0205 group B (a4ba3cae..d4ad7597) changed ``local_astra/pages/scenarios``.
POWER_BASE = "a4ba3cae01da9bb4731124e5ebb068d87140b41a"
POWER_TIP = "d4ad7597f1ee7ec2d587d37bd8977792e2dbe338"


# =====================================================================================================
# helpers
# =====================================================================================================


def rt() -> Any:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in (MODULE, "local_astra.pages"):
            raise AssertionError(f"T-0209 A1.1: {MODULE} does not exist yet") from exc
        raise


def rtx_dir() -> Path:
    """``rtx_tasks.RTX_DIR``, checked to be the real path of ``local_astra/pages/rtx/`` (A1.1)."""
    m = rt()
    assert hasattr(m, "RTX_DIR"), f"T-0209 A1.1: {MODULE} has no RTX_DIR"
    d = Path(m.RTX_DIR)
    want = Path(os.path.realpath(RTX_PATH))
    assert d == want, f"T-0209 A1.1: RTX_DIR is {d}, not the real path {want}"
    assert d.is_dir(), f"T-0209 A1.1: {d} is not a directory"
    return d


def rtx_files() -> list[Path]:
    root = rtx_dir()
    files = sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())
    assert files, f"T-0209 A1.1: {root} holds no file"
    return files


def rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def routes() -> dict[str, str]:
    m = rt()
    assert hasattr(m, "ROUTES"), f"T-0209 A1.2: {MODULE} has no ROUTES"
    return dict(m.ROUTES)


def resolve_fn() -> Any:
    from local_astra.pages import server

    return server.resolve


def normalise(text: str) -> str:
    """A1.2's normalisation, written from the task."""
    t = text.strip()
    t = re.sub(r"^(?i:https?://)", "", t)
    t = t.casefold()
    if t.startswith("www."):
        t = t[4:]
    t = re.split(r"[?#]", t, maxsplit=1)[0]
    if t.endswith("/"):
        t = t[:-1]
    return t


def plain(path: Path) -> str:
    t = path.read_text(encoding="utf-8")
    t = SCRIPT_BLOCK.sub(" ", STYLE_BLOCK.sub(" ", t))
    return re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", t)).replace("&amp;", "&")


PAGE_ID = re.compile(r"""\bdata-page-id\s*=\s*["']([^"']+)["']""")


def page_ids() -> dict[str, list[str]]:
    root = rtx_dir()
    return {
        rel(p, root): PAGE_ID.findall(p.read_text(encoding="utf-8"))
        for p in rtx_files()
        if p.suffix == ".html"
    }


# ------------------------------------------------------------------------------- A1.3's static rules

SCHEME_RTX = re.compile(r"(?i)\bhttps?:|://|\bjavascript:|\bdata:|\bblob:")
API_RTX = re.compile(
    r"\b(?:fetch|XMLHttpRequest|sendBeacon|WebSocket|EventSource|serviceWorker)\b"
    r"|\bwindow\s*\.\s*open\b"
    r"|\bimport\s*\("
    r"|\blocation(?:\s*\.\s*(?:href|pathname|search|hash|host|hostname|port|protocol|origin))?\s*=(?!=)"
    r"|\blocation\s*\.\s*(?:replace|reload)\s*\("
    r"|\bhistory\s*\.\s*(?:back|forward|go|pushState|replaceState)\s*\("
)
ASSIGN = re.compile(r"\blocation\s*\.\s*assign\s*\(")
FORM_TAG = re.compile(r"(?is)<form\b[^>]*>")
TEXT_FIELD = re.compile(r"(?i)\bcontenteditable\b|<textarea\b|<canvas\b")


def rtx_findings(name: str, text: str) -> list[tuple[str, str]]:
    """A1.3's static rule over one file's text: ``(rule, match)`` per hit. ``name`` is root-relative."""
    out: list[tuple[str, str]] = []
    web = name == "web.html"
    out += [("scheme", m.group(0)) for m in SCHEME_RTX.finditer(text)]
    out += [("protocol-relative", m.group(0)) for m in PROTO_RELATIVE.finditer(text)]
    for tag in TAG.findall(text):
        for k, v in attrs(tag):
            if k in URL_ATTRS:
                values = [c.strip().split(" ")[0] for c in v.split(",")] if k == "srcset" else [v]
                out += [("absolute-attr", f"{k}={x}") for x in values if ABSOLUTE.match(x)]
            if k == "formmethod" and v.strip().lower() != "get":
                out.append(("form-method", f"{k}={v}"))
            if k != "placeholder" and HOSTNAME.search(v):
                out.append(("hostname-in-attribute", f"{k}={v}"))
    for tag in FORM_TAG.findall(text):
        a = dict(attrs(tag))
        if "method" in a and a["method"].strip().lower() != "get":
            out.append(("form-method", tag))
        if "action" in a and (ABSOLUTE.match(a["action"]) or a["action"].strip().startswith("//")):
            out.append(("form-action", tag))
    css = (
        style_code(name, text)
        if name.endswith(".html")
        else (text if name.endswith(".css") else "")
    )
    out += [
        ("absolute-css-url", m.group(1))
        for m in CSS_URL.finditer(css)
        if ABSOLUTE.match(m.group(1))
    ]
    out += [("import", m.group(1)) for m in CSS_IMPORT.finditer(css) if ABSOLUTE.match(m.group(1))]
    out += [("network-api", m.group(0)) for m in API_RTX.finditer(text)]
    if not web:
        out += [("assign-outside-web", m.group(0)) for m in ASSIGN.finditer(text)]
    out += [("embed", m.group(0)) for m in EMBEDS.finditer(text)]
    out += [("link-hint", m.group(0)) for m in LINK_HINT.finditer(text)]
    out += [("text-field", m.group(0)) for m in TEXT_FIELD.finditer(text)]
    code = style_code(name, text) if name.endswith(".html") else ""
    if not web:
        code += "\n" + script_code(name, text)
    if name.endswith(".css"):
        code += "\n" + text
    out += [("hostname-in-code", m.group(0)) for m in HOSTNAME.finditer(code)]
    return out


#: (mutant, rule that must fire, seeded into web.html too)
MUTANTS: list[tuple[str, str, bool]] = [
    ('<a href="https://docs.blender.org/">Manual</a>', "scheme", True),
    ('<img src="https://198.51.100.7/x.png" alt="">', "scheme", True),
    ('<a href="http://x/">x</a>', "scheme", True),
    ('<link rel="stylesheet" href="//cdn/s.css">', "protocol-relative", True),
    ('<script>fetch("/x")</script>', "network-api", True),
    ("<script>new XMLHttpRequest()</script>", "network-api", True),
    ('<script>new WebSocket("ws")</script>', "network-api", True),
    ('<script>navigator.sendBeacon("/x", "1")</script>', "network-api", True),
    ('<script>location = "/x";</script>', "network-api", True),
    ('<script>window.location.href = "/x";</script>', "network-api", True),
    ('<script>window.location.replace("/x")</script>', "network-api", True),
    ('<script>history.pushState({}, "", "/x")</script>', "network-api", True),
    ('<script>window.open("/x")</script>', "network-api", True),
    ('<script>import("/x.js")</script>', "network-api", True),
    ('<script>location.assign("/blender-manual/")</script>', "assign-outside-web", False),
    ('<form method="post"></form>', "form-method", True),
    ('<form method="get" action="//evil/x"></form>', "form-action", True),
    ('<button formmethod="post">x</button>', "form-method", True),
    ('<iframe src="/x"></iframe>', "embed", True),
    ('<object data="/x"></object>', "embed", True),
    ('<meta http-equiv="refresh" content="0">', "embed", True),
    ('<link rel="prefetch" href="/x">', "link-hint", True),
    ('<a href="javascript:void(0)">x</a>', "scheme", True),
    ('<img src="data:image/png;base64,AAAA" alt="">', "scheme", True),
    ('<a title="docs.blender.org" href="/x">x</a>', "hostname-in-attribute", True),
    ('<script>const h = "docs.blender.org";</script>', "hostname-in-code", False),
    ('<div contenteditable="true"></div>', "text-field", True),
    ('<textarea name="t"></textarea>', "text-field", True),
    ("<canvas></canvas>", "text-field", True),
]
BENIGN = [
    '<form method="get" action="/blender-projects/"><input name="q" placeholder="Filter issues"></form>',
    '<form method="GET" action="index.html"></form>',
    "<form></form>",
    "<p>See docs.blender.org and blenderartists.org for more.</p>",
    '<a href="/blender-notes/4.4-cycles.html">Release notes</a>',
    "<script>// a comment\nconst n = 1;</script>",
    '<button data-scn="download-windows" type="button">Windows</button>',
    '<input name="address" placeholder="en.wikipedia.org">',
]


# =====================================================================================================
# A1.1: the root
# =====================================================================================================


def _targets(p: Path, root: Path) -> list[str]:
    text = p.read_text(encoding="utf-8", errors="replace")
    targets = [v for tag in TAG.findall(text) for k, v in attrs(tag) if k in URL_ATTRS]
    targets += CSS_URL.findall(text) + re.findall(
        r"""["'](/[^"'\s]*)["']""", script_code(p.name, text)
    )
    return [urllib.parse.urljoin("/" + rel(p, root), t) for t in targets]


def test_root_contents() -> None:
    """A1.1: ``RTX_DIR`` is the real path of ``local_astra/pages/rtx/``, inside no other root; it holds only
    ``.html``/``.css``/``.js`` regular files including A1's required files; each page has one ``data-page-id`` (the
    A1.4 id for each ``ROUTES`` target, ``web`` for ``web.html``, and one page ``forum-login``); and no
    ``static/``, ``suite/`` or ``scenarios/`` page links to a path only the rtx root serves."""
    root = rtx_dir()
    for other in (STATIC, SUITE, SCENARIOS):
        assert not root.is_relative_to(Path(os.path.realpath(other))), (root, other)
    files = rtx_files()
    bad = [
        rel(p, root) for p in files if p.is_symlink() or p.suffix.lower() not in ALLOWED_SUFFIXES
    ]
    assert bad == [], f"T-0209 A1.1: files other than .html/.css/.js regular files: {bad}"
    names = {rel(p, root) for p in files}
    missing = [f for f in REQUIRED_FILES if f not in names]
    assert missing == [], f"T-0209 A1.1: missing {missing}"
    ids = page_ids()
    multi = {n: v for n, v in ids.items() if len(set(v)) != 1}
    assert multi == {}, f"T-0209 A1.5: pages without exactly one data-page-id: {multi}"
    resolve = resolve_fn()
    for path, pid in PAGE_OF_PATH.items():
        target = resolve(root, path)
        assert target is not None, path
        got = ids[rel(Path(target), root)][0]
        assert got == pid, f"T-0209 A1.4: {path} has data-page-id {got!r}, not {pid!r}"
    assert sum(1 for v in ids.values() if v[0] == "forum-login") == 1, (
        "T-0209 A1.4: one forum-login page"
    )
    into: list[str] = []
    for other in (STATIC, SUITE, SCENARIOS):
        oroot = Path(os.path.realpath(other))
        for p in sorted(oroot.rglob("*")):
            if not p.is_file():
                continue
            for path in _targets(p, oroot):
                if (
                    path.startswith("/")
                    and resolve(oroot, path) is None
                    and resolve(root, path) is not None
                ):
                    into.append(f"{rel(p, oroot)} -> {path}")
    assert into == [], f"T-0209 A1.1: other roots link into rtx: {into}"


def test_links_resolve() -> None:
    """A1.1: every URL attribute and CSS ``url()`` target in the root (fragments aside), resolved against its page,
    resolves through ``pages.server.resolve(RTX_DIR, ...)`` to a file inside the root; no ``target=_blank``."""
    root = rtx_dir()
    resolve = resolve_fn()
    seen, broken = 0, []
    for p in rtx_files():
        name = rel(p, root)
        text = p.read_text(encoding="utf-8")
        targets = [
            v
            for tag in TAG.findall(text)
            for k, v in attrs(tag)
            if k in URL_ATTRS and not v.startswith("#")
        ]
        css = style_code(name, text) if name.endswith(".html") else text
        targets += [u for u in CSS_URL.findall(css) if u and not u.startswith("#")]
        for t in targets:
            seen += 1
            path = urllib.parse.urljoin("/" + name, t).split("#")[0].split("?")[0]
            hit = resolve(root, path) if path.startswith("/") else None
            if hit is None or not Path(hit).is_relative_to(root):
                broken.append(f"{name} -> {t}")
        assert not re.search(r"""(?i)\btarget\s*=\s*["']?_blank""", text), name
    assert seen > 0, "premise: the pages reference at least one file"
    assert broken == [], f"T-0209 A1.1: targets that do not resolve inside RTX_DIR: {broken}"


def test_home_pages_link_to_their_pages() -> None:
    """A1.4: ``manual-home`` links to ``manual-gpu``, ``projects-home`` to the issue, ``notes-home`` to the 4.4
    notes and the download page, ``forum-home`` to both threads; ``projects-home`` has a text input (the filter);
    ``forum-thread`` links to the ``forum-login`` page."""
    root = rtx_dir()
    resolve = resolve_fn()
    ids = page_ids()
    login = next(n for n, v in ids.items() if v and v[0] == "forum-login")

    def hrefs(path: str) -> set[str]:
        p = Path(resolve(root, path))
        text = p.read_text(encoding="utf-8")
        out = set()
        for tag in TAG.findall(text):
            for k, v in attrs(tag):
                if k in ("href", "action", "formaction") and not v.startswith("#"):
                    t = urllib.parse.urljoin("/" + rel(p, root), v).split("#")[0].split("?")[0]
                    hit = resolve(root, t)
                    if hit is not None:
                        out.add(rel(Path(hit), root))
        return out

    want = {
        "/blender-manual/": ["blender-manual/gpu-rendering.html"],
        "/blender-projects/": ["blender-projects/issue-134001.html"],
        "/blender-notes/": ["blender-notes/4.4-cycles.html", "blender-notes/download.html"],
        "/blender-artists/": [
            "blender-artists/thread-rtx-5090.html",
            "blender-artists/thread-oom.html",
        ],
        "/blender-artists/thread-rtx-5090.html": [login],
    }
    for home, targets in want.items():
        got = hrefs(home)
        missing = [t for t in targets if t not in got]
        assert missing == [], f"T-0209 A1.4: {home} does not link to {missing}; links {sorted(got)}"
    text = Path(resolve(root, "/blender-projects/")).read_text(encoding="utf-8")
    inputs = [t for t in re.findall(r"(?is)<input\b[^>]*>", text)]
    assert any(dict(attrs(t)).get("type", "text").lower() in ("text", "search") for t in inputs), (
        "T-0209 A1.4: projects-home has no issue filter box"
    )


# =====================================================================================================
# A1.2: ROUTES parity
# =====================================================================================================

PAIR = re.compile(r"""(["'])([^"'\s]+\.[^"'\s]+)\1\s*[:,]\s*(["'])(/[^"'\s]*)\3""")


def test_routes_parity_and_pins() -> None:
    """A1.2: ``rtx_tasks.ROUTES`` (a ``Mapping[str, str]``) equals ``web.html``'s in-page table (every string pair in
    its script); it holds A1.2's pinned keys with their paths; every key is already normalised; every path resolves
    to a file in the root; ``location.assign`` appears in ``web.html`` and in no other file."""
    from collections.abc import Mapping

    m = rt()
    assert isinstance(m.ROUTES, Mapping), type(m.ROUTES)
    table = routes()
    for k, v in PINNED_ROUTES.items():
        assert table.get(k) == v, f"T-0209 A1.2: ROUTES[{k!r}] is {table.get(k)!r}, not {v!r}"
    for k in table:
        assert normalise(k) == k, f"T-0209 A1.2: ROUTES key {k!r} is not normalised"
    root = rtx_dir()
    resolve = resolve_fn()
    for k, v in table.items():
        hit = resolve(root, v)
        assert hit is not None and Path(hit).is_file(), (
            f"T-0209 A1.2: {k} -> {v} resolves to no file"
        )
    web = (root / "web.html").read_text(encoding="utf-8")
    in_page = {mm.group(2): mm.group(4) for mm in PAIR.finditer(script_code("web.html", web))}
    assert in_page == table, (
        "T-0209 A1.2: web.html's table and rtx_tasks.ROUTES differ: "
        f"only in page {sorted(set(in_page) - set(table))}, only in ROUTES {sorted(set(table) - set(in_page))}, "
        f"different {sorted(k for k in set(in_page) & set(table) if in_page[k] != table[k])}"
    )
    assert ASSIGN.search(script_code("web.html", web)), (
        "T-0209 A1.2: web.html never calls location.assign"
    )
    others = [
        rel(p, root)
        for p in rtx_files()
        if rel(p, root) != "web.html" and ASSIGN.search(p.read_text())
    ]
    assert others == [], f"T-0209 A1.2: location.assign outside web.html: {others}"


def test_routes_parser_has_power() -> None:
    """The ``PAIR`` parser (own, green now): it reads object and array pairs, single or double quotes, and skips
    strings that are not a host key and a path."""
    js = (
        "const ROUTES = {\"docs.blender.org\": \"/blender-manual/\", 'blender.org/download': '/x.html'};\n"
        'const B = [["blenderartists.org", "/blender-artists/"]]; const HINT = "Type a web address like '
        'en.wikipedia.org"; const n = {"a": "/b"};'
    )
    got = {m.group(2): m.group(4) for m in PAIR.finditer(js)}
    assert got == {
        "docs.blender.org": "/blender-manual/",
        "blender.org/download": "/x.html",
        "blenderartists.org": "/blender-artists/",
    }, got


# =====================================================================================================
# A1.3 (static half)
# =====================================================================================================


def test_no_egress_strings() -> None:
    """A1.3 (adversarial): the static rule over every rtx file is clean. Power: each ``MUTANTS`` entry seeded into
    every page is caught by its own rule (the ``location.assign`` and host-in-script mutants on pages other than
    ``web.html``, where they are allowed); width: ``BENIGN`` adds no finding; a host name in ``assets/rtx.js`` or
    ``assets/rtx.css`` is a finding."""
    root = rtx_dir()
    files = [(rel(p, root), p.read_text(encoding="utf-8")) for p in rtx_files()]
    dirty = {n: f for n, t in files if (f := rtx_findings(n, t))}
    assert dirty == {}, f"T-0209 A1.3 (a text rule; prose can trip it): {dirty}"
    for name, text in files:
        if not name.endswith(".html"):
            continue
        for snippet, rule, on_web in MUTANTS:
            got = {r for r, _ in rtx_findings(name, _seed(name, text, snippet))}
            if name == "web.html" and not on_web:
                assert rule not in got, f"A1.2/A1.3 allow {snippet!r} in web.html; got {got}"
                continue
            assert rule in got, f"the scanner missed {snippet!r} in {name} (rule {rule}); got {got}"
        for snippet in BENIGN:
            assert rtx_findings(name, _seed(name, text, snippet)) == [], (name, snippet)
    assert {r for r, _ in rtx_findings("assets/rtx.js", 'const h = "docs.blender.org";')} == {
        "hostname-in-code"
    }
    assert {r for r, _ in rtx_findings("assets/rtx.css", "/* blenderartists.org */")} == {
        "hostname-in-code"
    }


def test_determinism_bans() -> None:
    """A1.3: T-0205 A3's determinism rule (imported) over every rtx file is clean. Power: each ``DET_MUTANTS``
    seeded into every page is caught; width: ``DET_BENIGN`` is not."""
    root = rtx_dir()
    files = [(rel(p, root), p.read_text(encoding="utf-8")) for p in rtx_files()]
    dirty = {n: f for n, t in files if (f := determinism_findings(n, t))}
    assert dirty == {}, f"T-0209 A1.3 (a text rule; prose in code can trip it): {dirty}"
    for name, text in files:
        if not name.endswith(".html"):
            continue
        for snippet in DET_MUTANTS:
            assert determinism_findings(name, _seed(name, text, snippet)), (name, snippet)
        for snippet in DET_BENIGN:
            assert determinism_findings(name, _seed(name, text, snippet)) == [], (name, snippet)


def test_footer_on_every_page() -> None:
    """A1.3: every page but ``web.html`` carries "Demo copy for testing; not affiliated with <name>. Facts on this
    page are invented."; on ``blender-*`` pages the name holds "Blender"; ``web.html`` carries no footer."""
    root = rtx_dir()
    missing, wrong = [], []
    for p in rtx_files():
        name = rel(p, root)
        if not name.endswith(".html"):
            continue
        m = FOOTER.search(plain(p))
        if name == "web.html":
            assert m is None, "T-0209 A1.3: web.html is the start-page stand-in, without the footer"
            continue
        if m is None:
            missing.append(name)
        elif name.startswith("blender-") and "Blender" not in m.group(1):
            wrong.append((name, m.group(1)))
    assert missing == [] and wrong == [], f"T-0209 A1.3 footers: missing {missing}, wrong {wrong}"


def test_web_html_words_match_the_live_start_page() -> None:
    """A1.2 "visible words: the same as ``pages/static/web.html``": the title "Open a website", "Type a web address
    and press Go", the label "Web address", the button "Go" and the live notice sentence appear in the stand-in's text;
    its one text input is not prefilled; the live page's hint strings appear verbatim in its script."""
    root = rtx_dir()
    live = (STATIC / "web.html").read_text(encoding="utf-8")
    notice = re.search(r'id="notice">([^<]+)<', live)
    assert notice is not None, "premise: the live page's notice"
    mine = (root / "web.html").read_text(encoding="utf-8")
    text = plain(root / "web.html")
    for words in (
        "Open a website",
        "Type a web address and press Go",
        "Web address",
        "Go",
        notice.group(1),
    ):
        assert words in text, f"T-0209 A1.2: web.html lacks {words!r}"
    for hint in ("Type a web address like en.wikipedia.org", "A web address has no spaces. "):
        assert hint in mine, f"T-0209 A1.2: web.html lacks the live hint {hint!r}"
    assert "This demo copy cannot open that site." in mine, "T-0209 A1.2: the refusal text"


# =====================================================================================================
# A1.1 / A1.8: the roots' 404s
# =====================================================================================================


def test_other_roots_do_not_serve_rtx() -> None:
    """A1.1: ``StartPageServer()`` and ``StartPageServer(static_dir=SCENARIO_DIR)`` answer 404 for every rtx path but
    ``/web.html`` (served with other bytes); ``StartPageServer(static_dir=RTX_DIR)`` answers 200 with the file's bytes
    for each."""
    from local_astra.pages.scenario_tasks import SCENARIO_DIR

    root = rtx_dir()
    paths = url_paths(root)
    assert "/assets/rtx.js" in paths and "/blender-artists/thread-rtx-5090.html" in paths, paths
    for other in (None, SCENARIO_DIR):
        srv = new_server(other)
        srv.start()
        try:
            port = port_of(srv.origin)
            served = {p: request(port, "GET", p) for p in paths}
        finally:
            srv.stop()
        leaked = [(p, r[0]) for p, r in served.items() if p != "/web.html" and r[0] != 404]
        assert leaked == [], f"T-0209 A1.1: root {other} serves rtx paths: {leaked}"
        assert served["/web.html"][1] != (root / "web.html").read_bytes(), other
    srv = new_server(root)
    srv.start()
    try:
        port = port_of(srv.origin)
        for p in paths:
            status, body, _ = request(port, "GET", p)
            target = root / (p.lstrip("/") + ("index.html" if p.endswith("/") else ""))
            assert (status, body) == (200, target.read_bytes()), (p, status)
    finally:
        srv.stop()


def test_rtx_root_serves_nothing_else() -> None:
    """A1.1/A1.8: the rtx root answers 404 for ``rtx_shared/`` (every spelling of the image and the manifest),
    ``handoff/`` and every ``pages/scenarios/`` path but ``/web.html`` and ``/``; traversal spellings never answer 200
    or the image's bytes."""
    from local_astra.pages.scenario_tasks import SCENARIO_DIR

    root = rtx_dir()
    image = (
        (RTX_SHARED / R1_IMAGE).read_bytes() if (RTX_SHARED / R1_IMAGE).is_file() else b"\x00" * 64
    )
    plain_paths = [
        f"/rtx_shared/{R1_IMAGE}",
        f"/{R1_IMAGE}",
        f"/pages/rtx_shared/{R1_IMAGE}",
        "/rtx_shared/MANIFEST.md",
        "/rtx_shared/",
    ]
    for img in sorted(p.name for p in HANDOFF.glob("*")):
        plain_paths += [f"/handoff/{img}", f"/{img}"]
    plain_paths += [p for p in url_paths(Path(SCENARIO_DIR)) if p not in ("/web.html", "/")]
    plain_paths += [f"/scenarios{p}" for p in url_paths(Path(SCENARIO_DIR))[:5]]
    traversal = [
        f"/../rtx_shared/{R1_IMAGE}",
        f"/%2e%2e/rtx_shared/{R1_IMAGE}",
        f"/..%2frtx_shared/{R1_IMAGE}",
    ]
    srv = new_server(root)
    srv.start()
    try:
        port = port_of(srv.origin)
        for p in plain_paths:
            status, body, _ = request(port, "GET", p)
            assert status == 404, f"T-0209 A1.1: the rtx root answers {status} for {p}"
        for p in traversal:
            status, body, _ = request(port, "GET", p)
            assert status != 200 and image not in body, (p, status)
    finally:
        srv.stop()


def test_no_root_serves_rtx_shared() -> None:
    """A1.8: neither ``StartPageServer()`` nor the scenarios root serves ``rtx_shared/`` (404 for every spelling)."""
    from local_astra.pages.scenario_tasks import SCENARIO_DIR

    assert (RTX_SHARED / R1_IMAGE).is_file(), f"T-0209 A1.8: {R1_IMAGE} does not exist yet"
    for other in (None, SCENARIO_DIR):
        srv = new_server(other)
        srv.start()
        try:
            port = port_of(srv.origin)
            for p in (f"/rtx_shared/{R1_IMAGE}", f"/{R1_IMAGE}", f"/pages/rtx_shared/{R1_IMAGE}"):
                status, _b, _h = request(port, "GET", p)
                assert status == 404, (other, p, status)
        finally:
            srv.stop()


# =====================================================================================================
# A1.8: the image
# =====================================================================================================


def table_row() -> Any:
    m = rt()
    assert hasattr(m, "RTX"), f"T-0209 A1.6: {MODULE} has no RTX"
    rows = {r.rtx_id: r for r in m.RTX}
    assert "R1" in rows, "T-0209 A1.6: no R1 row"
    return rows["R1"]


def r1_bytes() -> bytes:
    p = RTX_SHARED / R1_IMAGE
    assert p.is_file(), f"T-0209 A1.8: {p.relative_to(REPO)} does not exist yet"
    return p.read_bytes()


def test_image_sha_matches_the_table() -> None:
    """A1.8: the committed image's sha256 is R1's ``image_sha256`` and the row names that file."""
    row = table_row()
    assert row.image == R1_IMAGE, row.image
    got = hashlib.sha256(r1_bytes()).hexdigest()
    assert got == row.image_sha256, f"T-0209 A1.8: file sha {got} != table {row.image_sha256}"


def test_image_is_a_1280x800_q85_jpeg_in_the_bucket() -> None:
    """A1.8: SOI first, EOI last, SOF says 1280x800, Pillow reads an RGB JPEG of that size; its quantisation tables are
    Pillow's at ``quality=85`` (power: quality 75's differ); the HUD screen bucket would send it unscaled."""
    from PIL import Image

    from local_astra.contracts.perception import Source
    from local_astra.perception import policy

    data = r1_bytes()
    assert data[:3] == b"\xff\xd8\xff" and data[-2:] == b"\xff\xd9", (data[:4], data[-4:])
    _m, w, h = sof_size(data)
    assert (w, h) == (W, H), f"T-0209 A1.8: SOF says {w}x{h}"
    with Image.open(io.BytesIO(data)) as im:
        assert (im.format, im.size, im.mode) == ("JPEG", (W, H), "RGB"), (
            im.format,
            im.size,
            im.mode,
        )
        im.load()

    def tables(raw: bytes) -> dict[int, list[int]]:
        with Image.open(io.BytesIO(raw)) as im:
            q = getattr(im, "quantization", None)
            assert q, "no quantisation tables"
            return {k: list(v) for k, v in q.items()}

    def encode(quality: int) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (W, H), (200, 120, 40)).save(buf, "JPEG", quality=quality)
        return buf.getvalue()

    want = tables(encode(QUALITY))
    assert tables(encode(75)) != want, "power"
    assert tables(data) == want, "T-0209 A1.8: not quality 85's tables"
    bucket = policy.bucket_for(Source.SCREEN, w, h)
    assert policy.target_size(w, h, bucket) == (w, h), bucket


def test_shared_directory_and_manifest() -> None:
    """A1.8: ``rtx_shared/`` holds only JPEGs and ``MANIFEST.md``, R1's image among them; the manifest names the image,
    says original work and Apache-2.0 and names the generator script."""
    assert RTX_SHARED.is_dir(), f"T-0209 A1.8: {RTX_SHARED.relative_to(REPO)} does not exist yet"
    names = sorted(
        p.relative_to(RTX_SHARED).as_posix() for p in RTX_SHARED.rglob("*") if p.is_file()
    )
    assert R1_IMAGE in names and "MANIFEST.md" in names, names
    assert all(n == "MANIFEST.md" or (n.endswith(".jpg") and "/" not in n) for n in names), names
    text = (RTX_SHARED / "MANIFEST.md").read_text(encoding="utf-8")
    assert R1_IMAGE in text and re.search(r"(?i)\boriginal\b", text) and "Apache-2.0" in text, text[
        :400
    ]
    assert "scripts/make_t0209_shared_images.py" in text, "MANIFEST.md does not name the generator"


def test_generator_script() -> None:
    """A1.8: ``scripts/make_t0209_shared_images.py`` imports Pillow and no network module and names
    ``pages/rtx_shared`` as where it writes; its top-level ``r1()`` holds A1.8's texts (the five tab names each) as
    string constants and none naming a GPU."""
    assert SCRIPT.is_file(), f"T-0209 A1.8: {SCRIPT.relative_to(REPO)} does not exist yet"
    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    tops: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            tops.add(node.module.split(".")[0])
    assert "PIL" in tops and not (tops & NETWORK_MODULES), sorted(tops)
    assert re.search(r"pages[\"'/, ]+rtx_shared", src), "the script does not name pages/rtx_shared"
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "r1"]
    assert len(fns) == 1, "T-0209 A1.8: the generator has no single top-level r1()"
    consts = [
        n.value
        for n in ast.walk(fns[0])
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    joined = "\n".join(consts)
    for t in R1_TEXTS:
        assert any(t in c for c in consts), f"T-0209 A1.8: r1() draws no {t!r}"
    for tab in R1_TABS:
        assert re.search(rf"\b{re.escape(tab)}\b", joined), f"T-0209 A1.8: r1() has no tab {tab!r}"
    named = [c for c in consts if GPU_NAME.search(c)]
    assert named == [], f"T-0209 A1.8: r1() names a GPU: {named}"


# =====================================================================================================
# A1.15, A1.16
# =====================================================================================================


def test_make_browser_rtx_target() -> None:
    """A1.15: ``make -n -s browser-rtx`` prints exactly A1.15's command and the target is ``.PHONY`` (written by the
    orchestrator at Merge T-0209-a1; red until then)."""
    code, lines, err = _make_lines("browser-rtx")
    assert code == 0, f"T-0209 A1.15: no browser-rtx target: {err}"
    assert lines == [MAKE_BROWSER_RTX], lines
    phony = [ln for ln in (REPO / "Makefile").read_text().splitlines() if ln.startswith(".PHONY:")]
    assert any("browser-rtx" in ln.split() for ln in phony), phony


def test_cli_imports_neither_rtx_module() -> None:
    """A1.16 (static): ``cli.py`` imports neither ``rtx_tasks`` nor ``rtx_search`` (nor ``pages.rtx*`` by any spelling,
    nor names them in a string). Premise: both modules exist."""
    rt()
    assert importlib.util.find_spec(SEARCH_MODULE) is not None, (
        f"T-0209 A1.7: {SEARCH_MODULE} does not exist yet"
    )
    src = (REPO / "local_astra" / "cli.py").read_text(encoding="utf-8")
    bad: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if "rtx" in a.name]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            bad += [
                f"{mod}.{a.name}" for a in node.names if "rtx" in mod or a.name.startswith("rtx")
            ]
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.search(r"\brtx_(?:tasks|search)\b", node.value)
        ):
            bad.append(node.value)
    assert bad == [], f"T-0209 A1.16: cli.py reaches an rtx module: {bad}"


# =====================================================================================================
# A1.17 (green at BASE)
# =====================================================================================================


def _makefile_at(rev: str) -> list[str]:
    proc = gr._git("show", f"{rev}:Makefile")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


def test_frozen_paths_over_closed_range() -> None:
    """A1.17 (green at BASE): A1.17's paths are byte-identical over ``BASE..TIP`` (two literals, never the working
    tree), each exists at BASE; under ``tests/`` only A1's five modules and ``test_guards_use_fixed_refs`` change; the
    Makefile changes only by lines naming ``browser-rtx``. Power: the same predicate sees T-0205 B's change to
    ``local_astra/pages/scenarios``."""
    gr.require_commits(BASE, TIP)
    assert gr.is_ancestor(BASE, TIP), f"{BASE[:7]} is not an ancestor of {TIP[:7]}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    changed = gr.changed(BASE, TIP, *FROZEN)
    assert changed == [], f"T-0209 A1.17: frozen paths differ in {BASE[:7]}..{TIP[:7]}: {changed}"
    tests_changed = [p for p in gr.changed(BASE, TIP, "tests") if p not in TESTS_OWNED]
    assert tests_changed == [], f"T-0209 A1.17: other test files edited: {tests_changed}"
    before, after = _makefile_at(BASE), _makefile_at(TIP)
    removed = [ln for ln in before if ln not in after]
    added = [ln for ln in after if ln not in before]
    assert all(ln.startswith(".PHONY:") for ln in removed), removed
    assert all("browser-rtx" in ln for ln in added), added
    gr.require_commits(POWER_BASE, POWER_TIP)
    assert gr.changed(POWER_BASE, POWER_TIP, "local_astra/pages/scenarios"), (
        "power: the predicate cannot see a change"
    )
