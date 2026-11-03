"""T-0138 group A, criteria A1, A3 (static half), A8 (static half), A10 and A12: the suite's page root read as files,
the default root's 404s, the ``browser-suite`` Make target, and the frozen paths (docs/tasks/T-0138.md, "Tests to
write", group A, ``tests/unit/test_t0138_suite_static.py``).

**What these rows read.** ``suite_tasks.SUITE_DIR`` (through :func:`suite_dir`, which turns the missing module or
directory into a named ``AssertionError``), the files under it, ``pages.server.resolve`` and two real loopback
``StartPageServer``s. No browser.

**The static egress rule** (A8) is :func:`egress_findings`, written from A8's list; ``test_no_egress_strings`` proves
it can fire (A8's six named mutants and the rest of A8's list, each seeded in memory into every real suite file) and
that it is not too wide (benign constructs every page may need: a JS comment, a ``data-suite`` attribute, a read of
``location.hash``, a root-relative link, ``example.com`` text on the S11 page). It reads text, so prose can trip it:
every failure names the file, the rule and the match.

**Choices this module makes where A3/A8 are silent** (reported as ambiguities): the JS bans (``Date``,
``Math.random``, timers) are applied to script code only (``.js`` files, inline ``<script>`` and ``on*=``
attributes) and the CSS bans (``transition``, ``animation``) to style code only (``.css``, ``<style>``, ``style=``),
so page prose such as a "Date" label cannot trip them; ``@font-face`` and ``<select`` anywhere. A prefilled text
input is a ``value=`` attribute on a text-like ``<input>``, a non-empty ``<textarea>``, or a script assigning a
non-empty literal to ``.value``/``.defaultValue``; the browser module also reads every text input's value at load.
``location.assign(``/``location.replace(`` count as "an assignment to location" (script navigation).

BASE_A = d7ef317 (main at branch time, "T-0138: plan ..."). ``test_frozen_paths_over_closed_range`` is green at BASE_A
and stays green; every other row is activated for group A (O-A rulings) and red until the implementation lands.

Regime: CPU, files and loopback sockets only, n = 1.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any

from tests import _git_range as gr
from tests.unit.test_computer_imports import MAKE_BROWSER
from tests.unit.test_start_pages import new_server, port_of, request

REPO = Path(__file__).resolve().parents[2]
PAGES = REPO / "local_astra" / "pages"
SUITE_PATH = PAGES / "suite"
STATIC = PAGES / "static"
SUITE_MODULE = "local_astra.pages.suite_tasks"
ALLOWED_SUFFIXES = frozenset({".html", ".css", ".js"})
#: The Goal table's start pages and A1's two assets, as files under the root.
REQUIRED_FILES = (
    "seats.html",
    "parts.html",
    "catalogue.html",
    "delivery.html",
    "store/index.html",
    "booking.html",
    "mugs.html",
    "checkout.html",
    "kettles.html",
    "order.html",
    "assets/suite.css",
    "assets/suite.js",
)
#: The only page that may hold ``example.com`` text (A8: the S11 page).
S11_PAGE = "booking.html"
MAKE_BROWSER_SUITE = "ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0138_suite_browser.py -q -m browser"

#: T-0138 group A's BASE: main at d7ef317 ("T-0138: plan ..."), the commit the branch was cut from. A literal, never
#: a moving ref.
BASE_A = "d7ef3173525e5c6461a81e3ed96e90421a58d130"
#: The implementation tip: 93cd364 (``T-0138-a-impl``, the O-A2 fix; was 370f895, before that BASE_A). Bumped by the test-author or the orchestrator to
#: the last implementation commit; at the merge it closes to that commit.
TIP_A = "93cd3647c20b6a328c0d2dc3f9b83560496ebe16"
#: A12's list plus the task's "Not owned" paths that exist at BASE_A, written out from the task file.
FROZEN = (
    "local_astra/pages/server.py",
    "local_astra/pages/__init__.py",
    "local_astra/pages/static",
    "local_astra/conductor",
    "local_astra/computer",
    "local_astra/toolcall",
    "local_astra/cli.py",
    "local_astra/config.py",
    "local_astra/contracts",
    "local_astra/perception",
    "hud",
    "tests/fakes",
    "tests/fixtures/browser",
    "tests/unit/_live_browser_support.py",
    "tests/unit/test_computer_imports.py",
    "scripts",
    "services",
    "docs/02-architecture.md",
    "docs/03-test-plan.md",
    "pyproject.toml",
    "uv.lock",
    ".claude",
)
#: The test files group A may change: this task's three, and the orchestrator's ``CLOSED_AS_OF`` bump.
TESTS_OWNED = frozenset(
    {
        "tests/unit/test_t0138_suite_oracle.py",
        "tests/unit/test_t0138_suite_static.py",
        "tests/integration/test_t0138_suite_browser.py",
        "tests/unit/test_guards_use_fixed_refs.py",
    }
)
#: The power commit: T-0151 group B's harness commit, which changed ``scripts/`` (24102c8, its parent cbaa426).
POWER_PARENT = "cbaa42635c3d5241b8e4fd6c8c00495367336692"
POWER = "24102c8bb818b7749ebcd7297b85e8bf27b6bc83"


# =====================================================================================================
# helpers
# =====================================================================================================


def suite() -> Any:
    try:
        return importlib.import_module(SUITE_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in (SUITE_MODULE, "local_astra.pages"):
            raise AssertionError(f"T-0138 A1: {SUITE_MODULE} does not exist yet") from exc
        raise


def suite_dir() -> Path:
    """``suite_tasks.SUITE_DIR``, checked to be the real path of ``local_astra/pages/suite/`` (A1)."""
    m = suite()
    assert hasattr(m, "SUITE_DIR"), f"T-0138 A1: {SUITE_MODULE} has no SUITE_DIR"
    d = Path(m.SUITE_DIR)
    want = Path(os.path.realpath(SUITE_PATH))
    assert d == want, f"T-0138 A1: SUITE_DIR is {d}, not the real path {want}"
    assert d.is_dir(), f"T-0138 A1: {d} is not a directory"
    return d


def suite_files() -> list[Path]:
    root = suite_dir()
    files = sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())
    assert files, f"T-0138 A1: {root} holds no file"
    return files


def rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def url_paths(root: Path) -> list[str]:
    """Every file's root-relative URL path, plus ``/dir/`` for each ``index.html``."""
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        r = rel(p, root)
        out.append("/" + r)
        if p.name == "index.html":
            parent = r[: -len("index.html")]
            out.append("/" + parent)
    return out


def resolve_fn() -> Any:
    from local_astra.pages import server

    return server.resolve


# ------------------------------------------------------------------------------- code and markup slices

SCRIPT_BLOCK = re.compile(r"(?is)<script\b[^>]*>(.*?)</script\s*>")
STYLE_BLOCK = re.compile(r"(?is)<style\b[^>]*>(.*?)</style\s*>")
TAG = re.compile(r"(?s)<[a-zA-Z][^>]*>")
ATTR = re.compile(r"""(?is)\s([a-z_:][-\w:.]*)\s*=\s*("[^"]*"|'[^']*'|[^\s"'>]+)""")


def attrs(tag: str) -> list[tuple[str, str]]:
    return [(m.group(1).lower(), m.group(2).strip("\"'")) for m in ATTR.finditer(tag)]


def script_code(name: str, text: str) -> str:
    if name.endswith(".js"):
        return text
    if not name.endswith(".html"):
        return ""
    parts = [m.group(1) for m in SCRIPT_BLOCK.finditer(text)]
    for tag in TAG.findall(text):
        parts += [v for k, v in attrs(tag) if k.startswith("on")]
    return "\n".join(parts)


def style_code(name: str, text: str) -> str:
    if name.endswith(".css"):
        return text
    if not name.endswith(".html"):
        return ""
    parts = [m.group(1) for m in STYLE_BLOCK.finditer(text)]
    for tag in TAG.findall(text):
        parts += [v for k, v in attrs(tag) if k == "style"]
    return "\n".join(parts)


# ------------------------------------------------------------------------------- A8's static rule

SCHEME = re.compile(r"(?i)://|\bjavascript:|\bdata:|\bblob:")
PROTO_RELATIVE = re.compile(r"""(?i)(?:=\s*["']?|["'(]|url\(\s*["']?)//""")
URL_ATTRS = frozenset({"href", "src", "action", "srcset", "poster", "formaction", "data", "cite"})
CSS_URL = re.compile(r"""(?i)url\(\s*["']?([^"')]*)""")
CSS_IMPORT = re.compile(r"""(?i)@import\s+(?:url\(\s*)?["']?([^"')\s;]+)""")
ABSOLUTE = re.compile(r"(?i)^\s*(?:[a-z][a-z0-9+.-]*:|//)")
API = re.compile(
    r"\b(?:fetch|XMLHttpRequest|sendBeacon|WebSocket|EventSource|serviceWorker)\b"
    r"|\bwindow\s*\.\s*open\b"
    r"|\bimport\s*\("
    r"|\blocation(?:\s*\.\s*(?:href|pathname|search|hash|host|hostname|port|protocol|origin))?\s*=(?!=)"
    r"|\blocation\s*\.\s*(?:assign|replace)\s*\("
)
FORM_NET = re.compile(r"(?is)<form\b[^>]*\b(?:action|method)\s*=")
EMBEDS = re.compile(r"(?i)<(?:iframe|object|embed)\b|<meta\b[^>]*\bhttp-equiv\b")
LINK_HINT = re.compile(
    r"""(?is)<link\b[^>]*\brel\s*=\s*["']?[^"'>]*\b(?:preconnect|dns-prefetch|prefetch|prerender|manifest)\b"""
)
HOSTNAME = re.compile(r"(?i)\b[a-z0-9-]+\.(?:com|org|net|io|dev|uk|example)\b")


def egress_findings(name: str, text: str) -> list[tuple[str, str]]:
    """A8's static rule over one file's text: ``(rule, match)`` for every hit. ``name`` is the root-relative file
    name (the ``example.com`` exception is the S11 page's text only)."""
    out: list[tuple[str, str]] = []
    out += [("scheme", m.group(0)) for m in SCHEME.finditer(text)]
    out += [("protocol-relative", m.group(0)) for m in PROTO_RELATIVE.finditer(text)]
    for tag in TAG.findall(text):
        for k, v in attrs(tag):
            if k in URL_ATTRS:
                values = [c.strip().split(" ")[0] for c in v.split(",")] if k == "srcset" else [v]
                out += [("absolute-attr", f"{k}={x}") for x in values if ABSOLUTE.match(x)]
    css = style_code(name, text) if name.endswith(".html") else text
    out += [
        ("absolute-css-url", m.group(1))
        for m in CSS_URL.finditer(css)
        if ABSOLUTE.match(m.group(1))
    ]
    out += [("import", m.group(1)) for m in CSS_IMPORT.finditer(css) if ABSOLUTE.match(m.group(1))]
    out += [("network-api", m.group(0)) for m in API.finditer(text)]
    out += [("form-network", m.group(0)) for m in FORM_NET.finditer(text)]
    out += [("embed", m.group(0)) for m in EMBEDS.finditer(text)]
    out += [("link-hint", m.group(0)) for m in LINK_HINT.finditer(text)]
    in_tags = "\n".join(TAG.findall(text)) if name.endswith(".html") else ""
    for m in HOSTNAME.finditer(text):
        token = m.group(0).lower()
        allowed = token == "example.com" and name == S11_PAGE
        if not allowed:
            out.append(("hostname", m.group(0)))
    if name == S11_PAGE and "example.com" in in_tags.lower():
        out.append(("hostname-in-attribute", "example.com"))
    return out


#: (mutant, rule that must fire): A8's six named mutants first, then the rest of A8's list (own).
MUTANTS = [
    ('<img src="https://198.51.100.7/x.png" alt="">', "scheme"),
    ('<link rel="stylesheet" href="//cdn/s.css">', "protocol-relative"),
    ('<script>fetch("/x")</script>', "network-api"),
    ('<form action="/x"></form>', "form-network"),
    ('<script>navigator.sendBeacon("/x", "1")</script>', "network-api"),
    ("<p>Order at kettlehub.com today</p>", "hostname"),
    ('<a href="javascript:void(0)">x</a>', "scheme"),
    ('<img src="data:image/png;base64,AAAA" alt="">', "scheme"),
    ('<script>const u = "blob:x";</script>', "scheme"),
    ('<a href="mailto:ada@kettles.invalid">x</a>', "absolute-attr"),
    ('<img srcset="/a.png 1x, //b/c.png 2x" alt="">', "absolute-attr"),
    ("<style>body { background: url(//b/x.png); }</style>", "protocol-relative"),
    ('<style>@import url("http2:/x.css");</style>', "import"),
    ("<script>new XMLHttpRequest()</script>", "network-api"),
    ('<script>new WebSocket("/w")</script>', "network-api"),
    ('<script>new EventSource("/e")</script>', "network-api"),
    ('<script>navigator.serviceWorker.register("/sw.js")</script>', "network-api"),
    ('<script>window.open("/x")</script>', "network-api"),
    ('<script>location = "/x";</script>', "network-api"),
    ('<script>window.location.href = "/x";</script>', "network-api"),
    ('<script>import("/x.js")</script>', "network-api"),
    ('<form method="post"></form>', "form-network"),
    ('<iframe src="/x"></iframe>', "embed"),
    ('<object data="/x"></object>', "embed"),
    ('<embed src="/x">', "embed"),
    ('<meta http-equiv="refresh" content="0">', "embed"),
    ('<link rel="prefetch" href="/x">', "link-hint"),
    ('<link rel="dns-prefetch" href="/x">', "link-hint"),
    ('<link rel="manifest" href="/m.json">', "link-hint"),
    ("<p>see shop.example for more</p>", "hostname"),
]
#: Benign constructs (own): the rule must not fire on any of them, on any page.
BENIGN = [
    "<script>// add to the cart\nconst n = 1; /* a comment */</script>",
    '<button data-suite="add-blue-kettle" type="button">Add to cart</button>',
    '<script>if (location.hash === "#x") { void 0; }</script>',
    '<a href="/store/office.html">Office</a>',
    '<link rel="stylesheet" href="/assets/suite.css">',
    '<script src="/assets/suite.js"></script>',
    "<p>Pickup at a locker near you.</p>",
]


def _seed(name: str, text: str, snippet: str) -> str:
    if name.endswith(".html") and "</body>" in text:
        return text.replace("</body>", snippet + "\n</body>", 1)
    return text + "\n" + snippet + "\n"


# =====================================================================================================
# A1
# =====================================================================================================


def test_root_contents() -> None:
    """A1: ``SUITE_DIR`` is the real path of ``local_astra/pages/suite/``, not under ``pages/static/``; it holds only
    ``.html``/``.css``/``.js`` regular files (no symlink), including every start page and both assets; and no
    ``pages/static/`` link, script or stylesheet target names a suite file."""
    root = suite_dir()
    static = Path(os.path.realpath(STATIC))
    assert not root.is_relative_to(static), f"T-0138 A1/H-T138-7: the suite is under {static}"
    files = suite_files()
    bad = [
        rel(p, root) for p in files if p.is_symlink() or p.suffix.lower() not in ALLOWED_SUFFIXES
    ]
    assert bad == [], f"T-0138 A1: files other than .html/.css/.js regular files: {bad}"
    names = {rel(p, root) for p in files}
    missing = [f for f in REQUIRED_FILES if f not in names]
    assert missing == [], f"T-0138 A1: missing {missing}"
    resolve = resolve_fn()
    into: list[str] = []
    for p in sorted(static.rglob("*")):
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        targets = [v for tag in TAG.findall(text) for k, v in attrs(tag) if k in URL_ATTRS]
        targets += CSS_URL.findall(text) + re.findall(
            r"""["'](/[^"'\s]*)["']""", script_code(p.name, text)
        )
        for t in targets:
            path = urllib.parse.urljoin("/" + rel(p, static), t)
            if (
                path.startswith("/")
                and resolve(static, path) is None
                and resolve(root, path) is not None
            ):
                into.append(f"{rel(p, static)} -> {t}")
    assert into == [], f"T-0138 A1: pages/static links into the suite: {into}"


def test_default_root_does_not_serve_suite() -> None:
    """A1 (H-T138-7): a real ``StartPageServer()`` on loopback gives 404 for every suite path (each file, each
    directory index, ``/assets/suite.js``), and ``StartPageServer(static_dir=SUITE_DIR)`` gives 200 with the file's
    bytes for each."""
    root = suite_dir()
    paths = url_paths(root)
    assert "/assets/suite.js" in paths and "/store/" in paths, paths
    default = new_server()
    default.start()
    try:
        port = port_of(default.origin)
        served = [(p, request(port, "GET", p)[0]) for p in paths]
    finally:
        default.stop()
    leaked = [(p, s) for p, s in served if s != 404]
    assert leaked == [], f"T-0138 A1: the default root serves suite paths: {leaked}"
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


# =====================================================================================================
# A8 (static) and A1 self-containment
# =====================================================================================================


def test_no_egress_strings() -> None:
    """A8's static rule over every suite file is clean (adversarial): no scheme, protocol-relative or absolute URL
    in a URL-bearing attribute, CSS ``url()`` or ``@import``; no network or navigation API; no networked form,
    frame, object, embed, ``http-equiv`` or resource hint; no hostname-shaped token except ``example.com`` text on
    the S11 page. Power: each of ``MUTANTS`` (A8's six named mutants, then the rest of A8's list) seeded in memory
    into every real file is caught by its own rule; width: ``BENIGN`` seeded into every file adds no finding, and
    ``example.com`` is allowed as S11 text but not in an S11 attribute or on another page."""
    root = suite_dir()
    files = [(rel(p, root), p.read_text(encoding="utf-8")) for p in suite_files()]
    dirty = {n: f for n, t in files if (f := egress_findings(n, t))}
    assert dirty == {}, f"T-0138 A8 (a text rule; prose can trip it): {dirty}"
    for name, text in files:
        for snippet, rule in MUTANTS:
            got = {r for r, _ in egress_findings(name, _seed(name, text, snippet))}
            assert rule in got, f"the scanner missed {snippet!r} in {name} (rule {rule}); got {got}"
        for snippet in BENIGN:
            assert egress_findings(name, _seed(name, text, snippet)) == [], (name, snippet)
    assert egress_findings(S11_PAGE, "<p>Questions? Write to bookings@example.com.</p>") == []
    assert {r for r, _ in egress_findings(S11_PAGE, '<a title="example.com">x</a>')} >= {
        "hostname-in-attribute"
    }
    assert {r for r, _ in egress_findings("order.html", "<p>bookings@example.com</p>")} == {
        "hostname"
    }


def _link_targets(name: str, text: str) -> list[str]:
    out = [
        v
        for tag in TAG.findall(text)
        for k, v in attrs(tag)
        if k in URL_ATTRS and not v.startswith("#")
    ]
    css = style_code(name, text) if name.endswith(".html") else text
    out += [u for u in CSS_URL.findall(css) if u and not u.startswith("#")]
    return out


def test_links_resolve() -> None:
    """A1/A8 "Link resolution": every ``href``/``src`` (and other URL attribute) and CSS ``url()`` target in the
    suite, resolved against its page's URL, resolves through ``pages.server.resolve(SUITE_DIR, ...)`` to an existing
    suite file; at least one target exists. A4 S10: no ``target=_blank`` under ``store/``."""
    root = suite_dir()
    resolve = resolve_fn()
    seen, broken = 0, []
    for p in suite_files():
        name = rel(p, root)
        text = p.read_text(encoding="utf-8")
        for t in _link_targets(name, text):
            seen += 1
            path = urllib.parse.urljoin("/" + name, t)
            hit = resolve(root, path) if path.startswith("/") else None
            if hit is None or not Path(hit).is_relative_to(root):
                broken.append(f"{name} -> {t}")
        if name.startswith("store/"):
            assert not re.search(r"""(?i)\btarget\s*=\s*["']?_blank""", text), name
    assert seen > 0, "premise: the suite pages reference at least one file"
    assert broken == [], f"T-0138 A1/A8: targets that do not resolve inside SUITE_DIR: {broken}"


# =====================================================================================================
# A3 (static half)
# =====================================================================================================

JS_BANS = re.compile(
    r"\bDate\b|\bMath\s*\.\s*random\b|\bsetTimeout\b|\bsetInterval\b|\brequestAnimationFrame\b"
    r"|\.(?:value|defaultValue)\s*=\s*([\"'`])(?!\1)"
    r"|setAttribute\(\s*[\"']value[\"']"
    r"|createElement\(\s*[\"']select"
)
CSS_BANS = re.compile(r"(?i)\btransition|\banimation")
ANYWHERE_BANS = re.compile(r"(?i)@font-face|<select\b")
TEXT_TYPES = frozenset({"", "text", "search", "email", "tel", "url", "number", "password"})
INPUT_TAG = re.compile(r"(?is)<input\b[^>]*>")
TEXTAREA = re.compile(r"(?is)<textarea\b[^>]*>(.*?)</textarea\s*>")


def determinism_findings(name: str, text: str) -> list[tuple[str, str]]:
    out = [("js", m.group(0)) for m in JS_BANS.finditer(script_code(name, text))]
    out += [("css", m.group(0)) for m in CSS_BANS.finditer(style_code(name, text))]
    out += [("anywhere", m.group(0)) for m in ANYWHERE_BANS.finditer(text)]
    for tag in INPUT_TAG.findall(text):
        a = dict(attrs(tag))
        if a.get("type", "").lower() in TEXT_TYPES and "value" in a:
            out.append(("prefilled", tag))
    out += [("prefilled", m.group(0)) for m in TEXTAREA.finditer(text) if m.group(1).strip()]
    return out


DET_MUTANTS = [
    "<script>const t = Date.now();</script>",
    "<script>new Date()</script>",
    "<script>Math.random()</script>",
    "<script>setTimeout(f, 10)</script>",
    "<script>setInterval(f, 10)</script>",
    "<script>requestAnimationFrame(f)</script>",
    "<style>button { transition: color 1s; }</style>",
    "<style>.x { animation: spin 1s; }</style>",
    '<div style="transition: none"></div>',
    "<style>@font-face { font-family: x; }</style>",
    "<select><option>a</option></select>",
    '<input data-suite="guests" type="text" value="2">',
    '<input name="guests" value="2">',
    "<textarea>window seat</textarea>",
    '<script>guests.value = "2";</script>',
    "<script>el.setAttribute('value', '2')</script>",
    '<button onclick="setTimeout(go, 1)">x</button>',
]
DET_BENIGN = [
    "<p>Date of delivery: see the list below.</p>",
    "<p>No animation here.</p>",
    '<input data-suite="note" type="text" placeholder="2">',
    '<input type="radio" name="time" value="19:00" checked>',
    '<script>box.value = "";</script>',
]


def test_determinism_bans() -> None:
    """A3 (static half): no ``Date``, ``Math.random``, ``setTimeout``, ``setInterval`` or ``requestAnimationFrame`` in
    script code; no ``transition`` or ``animation`` in style code; no ``@font-face`` or ``<select``; no prefilled
    text input (no ``value=`` on a text-like input, no non-empty ``<textarea>``, no script prefill); and S11's Note
    is not a ``<textarea>`` (implementation notes). Power: each ``DET_MUTANTS`` seeded into every file is caught;
    width: ``DET_BENIGN`` (a "Date" in prose, a placeholder, a radio's value) is not."""
    root = suite_dir()
    files = [(rel(p, root), p.read_text(encoding="utf-8")) for p in suite_files()]
    dirty = {n: f for n, t in files if (f := determinism_findings(n, t))}
    assert dirty == {}, f"T-0138 A3 (a text rule; prose in code can trip it): {dirty}"
    booking = dict(files).get(S11_PAGE, "")
    assert "<textarea" not in booking.lower(), "T-0138 A4 S11: Note must be a single-line input"
    for name, text in files:
        if not name.endswith(".html"):
            continue
        for snippet in DET_MUTANTS:
            assert determinism_findings(name, _seed(name, text, snippet)), (name, snippet)
        for snippet in DET_BENIGN:
            assert determinism_findings(name, _seed(name, text, snippet)) == [], (name, snippet)


# =====================================================================================================
# A10
# =====================================================================================================


def _make_lines(target: str) -> tuple[int, list[str], str]:
    proc = subprocess.run(
        ["make", "-n", "-s", target],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    lines = [ln.split("#", 1)[0].strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return proc.returncode, lines, proc.stderr


def test_make_browser_suite_target() -> None:
    """A10: ``make -n -s browser-suite`` prints exactly A10's command and the target is ``.PHONY``; ``make -n -s
    browser`` still prints ``MAKE_BROWSER`` (imported from ``test_computer_imports``)."""
    code, lines, err = _make_lines("browser-suite")
    assert code == 0, f"T-0138 A10: no browser-suite target: {err}"
    assert lines == [MAKE_BROWSER_SUITE], lines
    phony = [ln for ln in (REPO / "Makefile").read_text().splitlines() if ln.startswith(".PHONY:")]
    assert any("browser-suite" in ln.split() for ln in phony), phony
    code, lines, err = _make_lines("browser")
    assert code == 0, err
    assert lines == [MAKE_BROWSER], lines


# =====================================================================================================
# A12 (green at BASE_A)
# =====================================================================================================


def _makefile_at(rev: str) -> list[str]:
    proc = gr._git("show", f"{rev}:Makefile")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


def test_frozen_paths_over_closed_range() -> None:
    """A12 (green at BASE_A): A12's paths and the task's other not-owned paths are byte-identical over
    ``BASE_A..TIP_A`` (two literals, never the working tree), and each exists at BASE_A; under ``tests/`` only this
    task's three modules and ``test_guards_use_fixed_refs.py`` (the orchestrator's bump) change; the Makefile
    changes only by lines naming ``browser-suite`` (the target and its ``.PHONY`` entry). Power: the same predicate
    sees 24102c8's ``scripts/`` change."""
    gr.require_commits(BASE_A, TIP_A)
    assert gr.is_ancestor(BASE_A, TIP_A), f"{BASE_A[:7]} is not an ancestor of {TIP_A[:7]}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_A, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), (
            f"premise: {p} exists at BASE_A"
        )
    changed = gr.changed(BASE_A, TIP_A, *FROZEN)
    assert changed == [], f"T-0138 A12: frozen paths differ in {BASE_A[:7]}..{TIP_A[:7]}: {changed}"
    tests_changed = [p for p in gr.changed(BASE_A, TIP_A, "tests") if p not in TESTS_OWNED]
    assert tests_changed == [], f"T-0138 A12: existing test files edited: {tests_changed}"
    before, after = _makefile_at(BASE_A), _makefile_at(TIP_A)
    removed = [ln for ln in before if ln not in after]
    added = [ln for ln in after if ln not in before]
    assert all(ln.startswith(".PHONY:") for ln in removed), removed
    assert all("browser-suite" in ln for ln in added), added
    if removed:
        assert [sorted(ln.split()) for ln in added if ln.startswith(".PHONY:")] == [
            sorted([*removed[0].split(), "browser-suite"])
        ], (removed, added)
    gr.require_commits(POWER_PARENT, POWER)
    assert gr.changed(POWER_PARENT, POWER, "scripts"), "power: the predicate cannot see a change"
