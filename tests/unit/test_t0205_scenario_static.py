"""T-0205 group A, criteria A1, A3 (static half), A8 (static half), A9 and A12: the scenario page root read as files,
the default root's 404s, the ``browser-demo`` Make target and the frozen paths (docs/tasks/T-0205.md, "Tests to
write", ``tests/unit/test_t0205_scenario_static.py``).

**What these rows read.** ``scenario_tasks.SCENARIO_DIR`` (through :func:`scn_dir`, a named ``AssertionError`` while the
module or directory is absent), the files under it, ``pages.server.resolve`` and two real loopback
``StartPageServer``s. No browser.

**The static egress rule** (A8) is T-0138's ``egress_findings`` rule, re-written here with A8's one change:
``example.com`` may appear as *text* on D7's pages (``trees/``, group B) only, never in an attribute. The hostname rule
adds ``.fr`` to T-0138's TLD list (the Louvre's real domain; own). Power: T-0138's mutants plus ``location.assign``
seeded into every real file; width: benign constructs.

**Readings this module pins where the task is silent** (reported):

- Group A's required files: ``web.html``, ``assets/scenarios.css``, ``assets/scenarios.js``, the five start pages
  (``louvre/index.html``, ``ramen/index.html``, ``recipes/lasagna.html``, ``breville/index.html``,
  ``outdoor/stoves.html``), and the four cart/checkout pages A4 names (``breville/cart.html``,
  ``breville/checkout.html``, ``outdoor/cart.html``, ``outdoor/checkout.html``).
- ``web.html`` links to each of the five start pages (group A's "demo sites").
- The A3 footer is required on every page but ``web.html`` (it is the start-page stand-in, not a copy of a site),
  matched by ``Demo copy for testing; not affiliated with <name>.``; on ``louvre/`` and ``breville/`` pages the name
  holds "Louvre" / "Breville".
- "The default root gives 404 for every scenario path" excludes ``/web.html``, which the default root also serves (its
  own start page); there the two roots must serve different bytes.
- A3's determinism bans are T-0138's (script code / style code split), plus native dialogs (``alert``/``confirm``/
  ``prompt``) and native date/time inputs (A3 "every calendar drawn inside the page"; implementation notes).
- "No logo, image or markup copied from a real site" is not machine-checkable beyond the file-type rule (no image
  file can exist under the root); not tested further.
- A12's TIP is a literal: BASE_A until the implementation lands; the test-author or orchestrator bumps ``TIP_A`` to the
  last implementation commit, and at the merge it closes to that commit.

BASE_A = c97e5b30 (main at branch time). ``test_frozen_paths_over_closed_range`` is green at BASE_A; every other row
is red until the implementation lands.

**Group B (D6-D9; BASE_B = a4ba3cae).** Readings added (reported):

- Group B's required files: ``pizza/index.html``, ``pizza/cart.html``, ``pizza/checkout.html`` (A4 names the last;
  the cart by analogy with D4/D5), ``trees/index.html``, ``trees/quote.html``, ``trips/index.html``,
  ``flights/index.html`` and ``flights/united/checkout.html``; D8's destination pages are any other ``trips/*.html``
  (their names are not pinned: the row finds each destination's page by its stop names).
- ``web.html`` links to all nine start pages.
- Footers: ``pizza/`` names "Domino", ``trees/`` names "Austin Tree Pros".
- D8's pages show each stop's per-person price as ``$<n>`` (``$0`` or "free" for zero) and "closed" for a stop
  closed on Saturday, between the stop's name and the next stop name of that page; an open stop's span does not say
  "closed"; the hub shows each destination's drive as A4's ``h:mm`` after its name.
- A12 for B: the same frozen list, plus the Makefile, byte-identical over ``BASE_B..TIP_B`` (B owns no Makefile
  line); ``TIP_B`` is BASE_B until B's implementation lands, then bumped to its last implementation commit.

Regime: CPU, files and loopback sockets only, n = 1.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.unit.test_start_pages import new_server, port_of, request

REPO = Path(__file__).resolve().parents[2]
PAGES = REPO / "local_astra" / "pages"
SCN_PATH = PAGES / "scenarios"
STATIC = PAGES / "static"
SUITE = PAGES / "suite"
MODULE = "local_astra.pages.scenario_tasks"
ALLOWED_SUFFIXES = frozenset({".html", ".css", ".js"})
START_PAGES = {
    "D1": "/louvre/",
    "D2": "/ramen/",
    "D3": "/recipes/lasagna.html",
    "D4": "/breville/",
    "D5": "/outdoor/stoves.html",
}
REQUIRED_FILES = (
    "web.html",
    "assets/scenarios.css",
    "assets/scenarios.js",
    "louvre/index.html",
    "ramen/index.html",
    "recipes/lasagna.html",
    "breville/index.html",
    "breville/cart.html",
    "breville/checkout.html",
    "outdoor/stoves.html",
    "outdoor/cart.html",
    "outdoor/checkout.html",
)
#: A8: ``example.com`` may be D7's text only (group B's ``trees/`` pages).
EXAMPLE_COM_PREFIX = "trees/"
MAKE_BROWSER_DEMO = (
    "ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0205_scenarios_browser.py "
    "tests/integration/test_t0205_scenarios_session.py -q -m browser"
)
FOOTER = re.compile(r"Demo copy for testing; not affiliated with ([^.<]+)\.")

#: T-0205 group A's BASE: main at c97e5b30 ("T-0207 planned ..."), the commit the branch was cut from.
BASE_A = "c97e5b300813fc7155ff76bb3f2e86d978b49beb"
#: The implementation tip: b8d82b6e (T-0205 A, last implementation commit after the A-R rulings; was 7c335b69; b219118e, the Makefile edit, is its ancestor).
TIP_A = "b8d82b6e1f7a373109e7be0f09b1dfe51d8a1e5a"
#: A12's list, written out from the task file.
FROZEN = (
    "local_astra/pages/server.py",
    "local_astra/pages/__init__.py",
    "local_astra/pages/static",
    "local_astra/pages/suite",
    "local_astra/pages/suite_tasks.py",
    "local_astra/conductor",
    "local_astra/computer",
    "local_astra/toolcall",
    "local_astra/cli.py",
    "local_astra/config.py",
    "local_astra/contracts",
    "tests/fakes",
    "docs/02-architecture.md",
    "docs/03-test-plan.md",
)
#: The test files group A may change: its four, and the orchestrator's ``CLOSED_AS_OF`` bump.
TESTS_OWNED = frozenset(
    {
        "tests/unit/test_t0205_scenario_judge.py",
        "tests/unit/test_t0205_scenario_static.py",
        "tests/integration/test_t0205_scenarios_browser.py",
        "tests/integration/test_t0205_scenarios_session.py",
        "tests/unit/test_guards_use_fixed_refs.py",
    }
)
#: T-0205 group B's BASE: main at a4ba3cae ("T-0207 A2 merged; docs/02 spec delta"), the commit B branched from.
BASE_B = "a4ba3cae01da9bb4731124e5ebb068d87140b41a"
#: B's implementation tip: d4ad7597, B's last implementation commit (B merged on main at 93232de7); was BASE_B.
TIP_B = "d4ad7597f1ee7ec2d587d37bd8977792e2dbe338"
START_PAGES_B = {"D6": "/pizza/", "D7": "/trees/", "D8": "/trips/", "D9": "/flights/"}
REQUIRED_FILES_B = (
    "pizza/index.html",
    "pizza/cart.html",
    "pizza/checkout.html",
    "trees/index.html",
    "trees/quote.html",
    "trips/index.html",
    "flights/index.html",
    "flights/united/checkout.html",
)
#: A4 D8, typed from the task: destination -> (drive h:mm, {stop: per-person dollars, None = closed Saturday}).
D8_SPEC: dict[str, tuple[str, dict[str, int | None]]] = {
    "Half Moon Bay": (
        "0:45",
        {"Ocean Kayak Tour": 95, "Pillar Point Harbor Walk": 0, "Harbor Fish Shack": 30},
    ),
    "Sonoma": (
        "1:05",
        {"Mission": 8, "Valley Creek Winery": 45, "Glen Oak Cheese Shop": None, "Plaza Bistro": 35},
    ),
    "Point Reyes": (
        "1:15",
        {
            "Lighthouse Visitor Center": 0,
            "Tomales Bay Oyster Farm": 60,
            "Bear Valley Trail": 0,
            "Inverness Tea Room": None,
        },
    ),
    "Monterey": ("2:20", {"Aquarium": 60}),
}
#: The power commit: T-0204 B's ruling commit a3e61ffb changed ``local_astra/computer/executor.py``.
POWER_PARENT = "9d41217aaec9109758a015bbb78411ec6be81373"
POWER = "a3e61ffb8a7416fc5c85a587eb548dd7fb870651"


# =====================================================================================================
# helpers
# =====================================================================================================


def scn() -> Any:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in (MODULE, "local_astra.pages"):
            raise AssertionError(f"T-0205 A1: {MODULE} does not exist yet") from exc
        raise


def scn_dir() -> Path:
    """``scenario_tasks.SCENARIO_DIR``, checked to be the real path of ``local_astra/pages/scenarios/`` (A1)."""
    m = scn()
    assert hasattr(m, "SCENARIO_DIR"), f"T-0205 A1: {MODULE} has no SCENARIO_DIR"
    d = Path(m.SCENARIO_DIR)
    want = Path(os.path.realpath(SCN_PATH))
    assert d == want, f"T-0205 A1: SCENARIO_DIR is {d}, not the real path {want}"
    assert d.is_dir(), f"T-0205 A1: {d} is not a directory"
    return d


def scn_files() -> list[Path]:
    root = scn_dir()
    files = sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())
    assert files, f"T-0205 A1: {root} holds no file"
    return files


def rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def url_paths(root: Path) -> list[str]:
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        r = rel(p, root)
        out.append("/" + r)
        if p.name == "index.html":
            out.append("/" + r[: -len("index.html")])
    return out


def resolve_fn() -> Any:
    from local_astra.pages import server

    return server.resolve


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
    r"|\blocation\s*\.\s*(?:assign|replace|reload)\s*\("
    r"|\bhistory\s*\.\s*(?:back|forward|go|pushState|replaceState)\s*\("
)
FORM_NET = re.compile(r"(?is)<form\b[^>]*\b(?:action|method)\s*=")
EMBEDS = re.compile(r"(?i)<(?:iframe|object|embed)\b|<meta\b[^>]*\bhttp-equiv\b")
LINK_HINT = re.compile(
    r"""(?is)<link\b[^>]*\brel\s*=\s*["']?[^"'>]*\b(?:preconnect|dns-prefetch|prefetch|prerender|manifest)\b"""
)
HOSTNAME = re.compile(r"(?i)\b[a-z0-9-]+\.(?:com|org|net|io|dev|uk|fr|gov|example)\b")


def egress_findings(name: str, text: str) -> list[tuple[str, str]]:
    """A8's static rule over one file's text: ``(rule, match)`` for every hit. ``name`` is root-relative."""
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
    d7 = name.startswith(EXAMPLE_COM_PREFIX) and name.endswith(".html")
    for m in HOSTNAME.finditer(text):
        if not (m.group(0).lower() == "example.com" and d7):
            out.append(("hostname", m.group(0)))
    in_tags = "\n".join(TAG.findall(text)) if name.endswith(".html") else ""
    if d7 and "example.com" in in_tags.lower():
        out.append(("hostname-in-attribute", "example.com"))
    return out


#: (mutant, rule that must fire): T-0138's list plus ``location.assign`` (A8 "seeded mutants").
MUTANTS = [
    ('<img src="https://198.51.100.7/x.png" alt="">', "scheme"),
    ('<link rel="stylesheet" href="//cdn/s.css">', "protocol-relative"),
    ('<script>fetch("/x")</script>', "network-api"),
    ('<form action="/x"></form>', "form-network"),
    ('<script>navigator.sendBeacon("/x", "1")</script>', "network-api"),
    ("<p>Order at breville.com today</p>", "hostname"),
    ("<p>Tickets at louvre.fr</p>", "hostname"),
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
    ('<script>location.assign("/breville/cart.html")</script>', "network-api"),
    ('<script>window.location.replace("/x")</script>', "network-api"),
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
    ("<p>write to alex@example.com</p>", "hostname"),
    # A8's exception names ``example.com`` only: a near-miss domain trips everywhere, ``trees/`` included.
    ("<p>write to alex@example.org</p>", "hostname"),
]
#: A8's one change: this ``MUTANTS`` entry is D7 *text*, so on ``trees/*.html`` it must produce no finding at all.
EXAMPLE_COM_TEXT_MUTANT = "<p>write to alex@example.com</p>"
BENIGN = [
    "<script>// add to the cart\nconst n = 1; /* a comment */</script>",
    '<button data-scn="add-trailhead-duo-stove" type="button">Add to cart</button>',
    '<script>if (location.hash === "#ingredients") { void 0; }</script>',
    '<a href="/breville/cart.html">Cart</a>',
    '<a href="#ingredients">Jump to recipe</a>',
    '<link rel="stylesheet" href="/assets/scenarios.css">',
    '<script src="/assets/scenarios.js"></script>',
    "<p>Open 9:00-18:00 daily except Tuesday.</p>",
]


def _seed(name: str, text: str, snippet: str) -> str:
    if name.endswith(".html") and "</body>" in text:
        return text.replace("</body>", snippet + "\n</body>", 1)
    return text + "\n" + snippet + "\n"


# =====================================================================================================
# A1
# =====================================================================================================


def _targets(p: Path, root: Path) -> list[str]:
    text = p.read_text(encoding="utf-8", errors="replace")
    targets = [v for tag in TAG.findall(text) for k, v in attrs(tag) if k in URL_ATTRS]
    targets += CSS_URL.findall(text) + re.findall(
        r"""["'](/[^"'\s]*)["']""", script_code(p.name, text)
    )
    return [urllib.parse.urljoin("/" + rel(p, root), t) for t in targets]


def test_root_contents() -> None:
    """A1: ``SCENARIO_DIR`` is the real path of ``local_astra/pages/scenarios/``, under neither ``pages/static/`` nor
    ``pages/suite/``; it holds only ``.html``/``.css``/``.js`` regular files (no symlink), including group A's required
    files; and no ``pages/static/`` or ``pages/suite/`` link, script or stylesheet target names a scenario file that
    its own root lacks."""
    root = scn_dir()
    for other in (STATIC, SUITE):
        assert not root.is_relative_to(Path(os.path.realpath(other))), (root, other)
    files = scn_files()
    bad = [
        rel(p, root) for p in files if p.is_symlink() or p.suffix.lower() not in ALLOWED_SUFFIXES
    ]
    assert bad == [], f"T-0205 A1: files other than .html/.css/.js regular files: {bad}"
    names = {rel(p, root) for p in files}
    missing = [f for f in REQUIRED_FILES if f not in names]
    assert missing == [], f"T-0205 A1: missing {missing}"
    resolve = resolve_fn()
    into: list[str] = []
    for other in (STATIC, SUITE):
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
    assert into == [], f"T-0205 A1: pages/static or pages/suite link into the scenarios: {into}"


def test_web_html_lists_the_demo_sites() -> None:
    """A1: ``web.html`` links to each of group A's five start pages (root-relative, resolving inside the root), has
    one text input (the address box; the browser module checks that it only filters)."""
    root = scn_dir()
    page = root / "web.html"
    assert page.is_file(), "T-0205 A1: no web.html stand-in"
    text = page.read_text(encoding="utf-8")
    hrefs = {
        urllib.parse.urljoin("/web.html", v)
        for tag in TAG.findall(text)
        if tag.lower().startswith("<a")
        for k, v in attrs(tag)
        if k == "href"
    }
    resolve = resolve_fn()
    for sid, start in START_PAGES.items():
        want = resolve(root, start)
        assert want is not None, f"T-0205 A2: {sid}'s start page {start} does not resolve"
        assert any(h.split("#")[0] and resolve(root, h.split("#")[0]) == want for h in hrefs), (
            f"T-0205 A1: web.html has no link to {sid}'s {start}: {sorted(hrefs)}"
        )
    inputs = [t for t in re.findall(r"(?is)<input\b[^>]*>", text)]
    text_inputs = [
        t for t in inputs if dict(attrs(t)).get("type", "text").lower() in ("text", "search", "url")
    ]
    assert len(text_inputs) == 1, f"T-0205 A1: web.html needs one address box: {inputs}"


def test_root_contents_b() -> None:
    """A1/A4 (group B): the root holds group B's required files, and ``web.html`` links to each of the nine start
    pages (root-relative, resolving inside the root)."""
    root = scn_dir()
    names = {rel(p, root) for p in scn_files()}
    missing = [f for f in REQUIRED_FILES_B if f not in names]
    assert missing == [], f"T-0205 A1 (B): missing {missing}"
    text = (root / "web.html").read_text(encoding="utf-8")
    hrefs = {
        urllib.parse.urljoin("/web.html", v).split("#")[0]
        for tag in TAG.findall(text)
        if tag.lower().startswith("<a")
        for k, v in attrs(tag)
        if k == "href"
    }
    resolve = resolve_fn()
    for sid, start in {**START_PAGES, **START_PAGES_B}.items():
        want = resolve(root, start)
        assert want is not None, f"T-0205 A2: {sid}'s start page {start} does not resolve"
        assert any(h and resolve(root, h) == want for h in hrefs), (
            f"T-0205 A1: web.html has no link to {sid}'s {start}: {sorted(hrefs)}"
        )


def _plain(path: Path) -> str:
    """A page's text: scripts and styles dropped, tags removed, whitespace collapsed."""
    t = path.read_text(encoding="utf-8")
    t = SCRIPT_BLOCK.sub(" ", STYLE_BLOCK.sub(" ", t))
    return re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", t)).replace("&amp;", "&")


def _name_re(name: str) -> re.Pattern[str]:
    words = [re.escape(w) for w in name.casefold().split()]
    return re.compile(r"\b" + r"[\s-]+".join(words) + r"\b")


def _spans(text: str, names: list[str]) -> dict[str, str] | None:
    """For each name, the casefolded text from its first occurrence to the next occurrence of another name (or the
    end); ``None`` when some name is absent."""
    low = text.casefold()
    starts: dict[str, int] = {}
    for n in names:
        m = _name_re(n).search(low)
        if m is None:
            return None
        starts[n] = m.start()
    out = {}
    for n, s in starts.items():
        ends = [
            m.start() for o in names if o != n for m in _name_re(o).finditer(low) if m.start() > s
        ]
        out[n] = low[s : min(ends) if ends else len(low)]
    return out


def _stop_shown(span: str, price: int | None) -> bool:
    """A stop's casefolded span shows ``price`` (``None``: "closed"; 0: ``$0`` or "free")."""
    if price is None:
        return "closed" in span
    if price == 0:
        return bool(re.search(r"\$0(?![\d.])|\bfree\b", span)) and "closed" not in span
    return bool(re.search(rf"\${price}(?!\d)", span)) and "closed" not in span


def test_d8_page_rule_has_power() -> None:
    """The D8 page rule (own, green now): it accepts a hand-made Point Reyes page and refuses each one-fact mutant
    of it (a wrong price, a $60 shown as $600, a closed stop shown open, an open stop shown closed, a missing stop)."""
    stops = D8_SPEC["Point Reyes"][1]
    good = (
        "Point Reyes. Lighthouse Visitor Center Open Saturday $0 per person. Tomales Bay Oyster Farm Open "
        "Saturday $60 per person. Bear Valley Trail Open Saturday Free. Inverness Tea Room Closed Saturday."
    )

    def passes(text: str) -> bool:
        spans = _spans(text, list(stops))
        return spans is not None and all(_stop_shown(spans[s], p) for s, p in stops.items())

    assert passes(good)
    for bad in (
        good.replace("$60", "$45"),
        good.replace("$60", "$600"),
        good.replace("Tea Room Closed Saturday", "Tea Room Open Saturday $12"),
        good.replace("Bear Valley Trail Open Saturday Free", "Bear Valley Trail Closed Saturday"),
        good.replace("Bear Valley Trail", "Bear Creek Trail"),
    ):
        assert not passes(bad), bad


def test_d8_stops_match_pages() -> None:
    """A4 D8: ``scenario_tasks.D8_STOPS`` equals A4's table (typed here), and the pages show the same values: one
    ``trips/`` page per destination holds all its stop names, each stop's span shows its per-person price (``$<n>``;
    ``$0`` or "free" for zero) or "closed" for a stop closed on Saturday, and no open stop's span says "closed"; the hub
    ``trips/index.html`` shows each destination with its drive time (A4's ``h:mm``)."""
    m = scn()
    assert hasattr(m, "D8_STOPS"), f"T-0205 A4: {MODULE} has no D8_STOPS"
    got = {d: dict(s) for d, s in dict(m.D8_STOPS).items()}
    assert got == {d: stops for d, (_, stops) in D8_SPEC.items()}, got
    root = scn_dir()
    pages = sorted(p for p in (root / "trips").glob("*.html") if p.name != "index.html")
    texts = {p.name: _plain(p) for p in pages}
    problems: list[Any] = []
    for dest, (_, stops) in D8_SPEC.items():
        hits = [n for n, t in texts.items() if _spans(t, list(stops)) is not None]
        if len(hits) != 1:
            problems.append((dest, "pages holding every stop", hits))
            continue
        spans = _spans(texts[hits[0]], list(stops))
        assert spans is not None
        for stop, price in stops.items():
            span = spans[stop]
            if not _stop_shown(span, price):
                problems.append((dest, stop, price, span[:160]))
    hub = _plain(root / "trips" / "index.html")
    spans = _spans(hub, list(D8_SPEC))
    if spans is None:
        problems.append(("hub", "a destination is missing", hub[:300]))
    else:
        for dest, (drive, _) in D8_SPEC.items():
            if not re.search(rf"(?<![\d:]){re.escape(drive)}(?![\d])", spans[dest]):
                problems.append(("hub", dest, drive, spans[dest][:160]))
    assert problems == [], f"T-0205 A4 D8: the pages disagree with D8_STOPS: {problems}"


def test_default_root_does_not_serve_scenarios() -> None:
    """A1: a real ``StartPageServer()`` gives 404 for every scenario path except ``/web.html`` (which it serves as its
    own start page, with different bytes); ``StartPageServer(static_dir=SCENARIO_DIR)`` gives 200 and the file's bytes
    for each."""
    root = scn_dir()
    paths = url_paths(root)
    assert "/assets/scenarios.js" in paths and "/louvre/" in paths, paths
    default = new_server()
    default.start()
    try:
        port = port_of(default.origin)
        served = [(p, request(port, "GET", p)) for p in paths]
    finally:
        default.stop()
    leaked = [(p, r[0]) for p, r in served if p != "/web.html" and r[0] != 404]
    assert leaked == [], f"T-0205 A1: the default root serves scenario paths: {leaked}"
    web_default = dict(served)["/web.html"]
    assert web_default[1] != (root / "web.html").read_bytes(), (
        "T-0205 A1: the default /web.html is the scenario stand-in"
    )
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
    """A8's static rule over every scenario file is clean (adversarial). Power: each of ``MUTANTS`` seeded in memory
    into every real file is caught by its own rule; width: ``BENIGN`` seeded into every file adds no finding;
    ``example.com`` is allowed as D7 text but not in a D7 attribute or on another page."""
    root = scn_dir()
    files = [(rel(p, root), p.read_text(encoding="utf-8")) for p in scn_files()]
    dirty = {n: f for n, t in files if (f := egress_findings(n, t))}
    assert dirty == {}, f"T-0205 A8 (a text rule; prose can trip it): {dirty}"
    for name, text in files:
        d7_text = name.startswith(EXAMPLE_COM_PREFIX) and name.endswith(".html")
        for snippet, rule in MUTANTS:
            got = {r for r, _ in egress_findings(name, _seed(name, text, snippet))}
            if d7_text and snippet == EXAMPLE_COM_TEXT_MUTANT:
                assert got == set(), f"A8 allows {snippet!r} as D7 text in {name}; got {got}"
                continue
            assert rule in got, f"the scanner missed {snippet!r} in {name} (rule {rule}); got {got}"
        for snippet in BENIGN:
            assert egress_findings(name, _seed(name, text, snippet)) == [], (name, snippet)
    assert egress_findings("trees/quote.html", "<p>Email: alex@example.com</p>") == []
    assert {r for r, _ in egress_findings("trees/quote.html", '<a title="example.com">x</a>')} >= {
        "hostname-in-attribute"
    }
    assert {r for r, _ in egress_findings("breville/index.html", "<p>alex@example.com</p>")} == {
        "hostname"
    }


def test_links_resolve() -> None:
    """A1/A8: every URL attribute and CSS ``url()`` target in the root (fragments aside), resolved against its page's
    URL, resolves through ``pages.server.resolve(SCENARIO_DIR, ...)`` to a file inside the root; at least one target
    exists; no ``target=_blank``."""
    root = scn_dir()
    resolve = resolve_fn()
    seen, broken = 0, []
    for p in scn_files():
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
            path = urllib.parse.urljoin("/" + name, t).split("#")[0]
            hit = resolve(root, path) if path.startswith("/") else None
            if hit is None or not Path(hit).is_relative_to(root):
                broken.append(f"{name} -> {t}")
        assert not re.search(r"""(?i)\btarget\s*=\s*["']?_blank""", text), name
    assert seen > 0, "premise: the scenario pages reference at least one file"
    assert broken == [], f"T-0205 A1/A8: targets that do not resolve inside SCENARIO_DIR: {broken}"


# =====================================================================================================
# A3 (static half)
# =====================================================================================================

JS_BANS = re.compile(
    r"\bDate\b|\bMath\s*\.\s*random\b|\bsetTimeout\b|\bsetInterval\b|\brequestAnimationFrame\b"
    r"|\b(?:alert|confirm|prompt)\s*\("
    r"|\.(?:value|defaultValue)\s*=\s*([\"'`])(?!\1)"
    r"|setAttribute\(\s*[\"']value[\"']"
    r"|createElement\(\s*[\"']select"
)
CSS_BANS = re.compile(r"(?i)\btransition|\banimation")
ANYWHERE_BANS = re.compile(
    r"(?i)@font-face|<select\b"
    r"|<input\b[^>]*\btype\s*=\s*[\"']?(?:date|datetime-local|month|week|time)\b"
)
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
    "<script>alert('Subscribed')</script>",
    "<script>if (confirm('Leave?')) {}</script>",
    "<style>button { transition: color 1s; }</style>",
    "<style>.x { animation: spin 1s; }</style>",
    '<div style="transition: none"></div>',
    "<style>@font-face { font-family: x; }</style>",
    "<select><option>a</option></select>",
    '<input type="date" name="depart">',
    '<input data-scn="q" type="text" value="BES870">',
    '<input name="q" value="Barista">',
    "<textarea>my steam wand</textarea>",
    '<script>q.value = "BES870";</script>',
    "<script>el.setAttribute('value', '2')</script>",
    '<button onclick="setTimeout(go, 1)">x</button>',
]
DET_BENIGN = [
    "<p>Date of your visit: see the table below.</p>",
    "<p>No animation here.</p>",
    '<input data-scn="modal-email" type="email" placeholder="you@your.mail">',
    '<input type="checkbox" name="cap" value="1-2">',
    '<script>box.value = "";</script>',
    "<script>function confirmOrder() { return 1; }</script>",
]


def test_determinism_bans() -> None:
    """A3 (static half): no ``Date``, ``Math.random``, timers or native dialogs in script code; no ``transition`` or
    ``animation`` in style code; no ``@font-face``, ``<select`` or native date/time input; no prefilled text input.
    Power: each ``DET_MUTANTS`` seeded into every page is caught; width: ``DET_BENIGN`` is not."""
    root = scn_dir()
    files = [(rel(p, root), p.read_text(encoding="utf-8")) for p in scn_files()]
    dirty = {n: f for n, t in files if (f := determinism_findings(n, t))}
    assert dirty == {}, f"T-0205 A3 (a text rule; prose in code can trip it): {dirty}"
    for name, text in files:
        if not name.endswith(".html"):
            continue
        for snippet in DET_MUTANTS:
            assert determinism_findings(name, _seed(name, text, snippet)), (name, snippet)
        for snippet in DET_BENIGN:
            assert determinism_findings(name, _seed(name, text, snippet)) == [], (name, snippet)


def test_footer_on_every_page() -> None:
    """A3: every scenario page but ``web.html`` carries "Demo copy for testing; not affiliated with <name>."; the
    Louvre and Breville copies name their brand."""
    root = scn_dir()
    missing, wrong = [], []
    for p in scn_files():
        name = rel(p, root)
        if not name.endswith(".html") or name == "web.html":
            continue
        text = re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", p.read_text(encoding="utf-8")))
        m = FOOTER.search(text)
        if m is None:
            missing.append(name)
            continue
        brand = {"louvre/": "Louvre", "breville/": "Breville"}
        for prefix, word in brand.items():
            if name.startswith(prefix) and word not in m.group(1):
                wrong.append((name, m.group(0)))
    assert missing == [], f"T-0205 A3: pages without the footer: {missing}"
    assert wrong == [], f"T-0205 A3: footers that do not name the site: {wrong}"


def test_footer_names_b_brands() -> None:
    """A3 (group B): every ``pizza/`` page's footer names "Domino" and every ``trees/`` page's "Austin Tree Pros";
    each B directory exists and holds at least one page."""
    root = scn_dir()
    wrong, seen = [], {"pizza/": 0, "trees/": 0, "trips/": 0, "flights/": 0}
    brand = {"pizza/": "Domino", "trees/": "Austin Tree Pros"}
    for p in scn_files():
        name = rel(p, root)
        if not name.endswith(".html"):
            continue
        for prefix in seen:
            if name.startswith(prefix):
                seen[prefix] += 1
        text = re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", p.read_text(encoding="utf-8")))
        m = FOOTER.search(text)
        for prefix, word in brand.items():
            if name.startswith(prefix) and (m is None or word not in m.group(1)):
                wrong.append((name, m.group(0) if m else None))
    assert all(seen.values()), f"T-0205 A1 (B): directories without pages: {seen}"
    assert wrong == [], f"T-0205 A3: footers that do not name the site: {wrong}"


# =====================================================================================================
# A9
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


def test_make_browser_demo_target() -> None:
    """A9: ``make -n -s browser-demo`` prints exactly A9's command and the target is ``.PHONY``."""
    code, lines, err = _make_lines("browser-demo")
    assert code == 0, f"T-0205 A9: no browser-demo target: {err}"
    assert lines == [MAKE_BROWSER_DEMO], lines
    phony = [ln for ln in (REPO / "Makefile").read_text().splitlines() if ln.startswith(".PHONY:")]
    assert any("browser-demo" in ln.split() for ln in phony), phony


# =====================================================================================================
# A12 (green at BASE_A)
# =====================================================================================================


def _makefile_at(rev: str) -> list[str]:
    proc = gr._git("show", f"{rev}:Makefile")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


def test_frozen_paths_over_closed_range() -> None:
    """A12 (green at BASE_A): A12's paths are byte-identical over ``BASE_A..TIP_A`` (two literals, never the working
    tree), and each exists at BASE_A; under ``tests/`` only this task's four modules and ``test_guards_use_fixed_refs``
    change; the Makefile changes only by lines naming ``browser-demo`` (the target and its ``.PHONY`` entry), so every
    other target is unchanged. Power: the same predicate sees a3e61ffb's ``local_astra/computer`` change."""
    gr.require_commits(BASE_A, TIP_A)
    assert gr.is_ancestor(BASE_A, TIP_A), f"{BASE_A[:7]} is not an ancestor of {TIP_A[:7]}"
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_A, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), (
            f"premise: {p} exists at BASE_A"
        )
    changed = gr.changed(BASE_A, TIP_A, *FROZEN)
    assert changed == [], f"T-0205 A12: frozen paths differ in {BASE_A[:7]}..{TIP_A[:7]}: {changed}"
    tests_changed = [p for p in gr.changed(BASE_A, TIP_A, "tests") if p not in TESTS_OWNED]
    assert tests_changed == [], f"T-0205 A12: other test files edited: {tests_changed}"
    before, after = _makefile_at(BASE_A), _makefile_at(TIP_A)
    removed = [ln for ln in before if ln not in after]
    added = [ln for ln in after if ln not in before]
    assert all(ln.startswith(".PHONY:") for ln in removed), removed
    assert all("browser-demo" in ln for ln in added), added
    if removed:
        assert [sorted(ln.split()) for ln in added if ln.startswith(".PHONY:")] == [
            sorted([*removed[0].split(), "browser-demo"])
        ], (removed, added)
    gr.require_commits(POWER_PARENT, POWER)
    assert gr.changed(POWER_PARENT, POWER, "local_astra/computer"), (
        "power: the predicate cannot see a change"
    )


def test_frozen_paths_over_closed_range_b() -> None:
    """A12 for group B (green at BASE_B): A12's paths and the Makefile are byte-identical over ``BASE_B..TIP_B`` (two
    literals), and under ``tests/`` only this task's four modules and ``test_guards_use_fixed_refs`` change. Power:
    the same predicate sees group A's Makefile edit over ``BASE_A..TIP_A``."""
    gr.require_commits(BASE_B, TIP_B)
    assert gr.is_ancestor(BASE_B, TIP_B), f"{BASE_B[:7]} is not an ancestor of {TIP_B[:7]}"
    changed = gr.changed(BASE_B, TIP_B, *FROZEN, "Makefile")
    assert changed == [], (
        f"T-0205 A12 (B): frozen paths differ in {BASE_B[:7]}..{TIP_B[:7]}: {changed}"
    )
    tests_changed = [p for p in gr.changed(BASE_B, TIP_B, "tests") if p not in TESTS_OWNED]
    assert tests_changed == [], f"T-0205 A12 (B): other test files edited: {tests_changed}"
    gr.require_commits(BASE_A, TIP_A)
    assert gr.changed(BASE_A, TIP_A, "Makefile") == ["Makefile"], (
        "power: the predicate cannot see group A's Makefile edit"
    )


# =====================================================================================================
# B2 ruling B-R2 (D7): ``submissions`` joins the A-R4 cross-site union
# =====================================================================================================

NODE = shutil.which("node")
#: A stand-in DOM for running ``assets/scenarios.js`` under node: ``document.body`` carries the real page's ``<body>``
#: attributes; every other element is an inert stub; ``window`` is the global object and ``localStorage`` an in-memory
#: store seeded from the config. It prints ``window.__state`` as JSON. Nothing here knows how the page computes state.
_NODE_PRELUDE = r"""
const CFG = JSON.parse(process.argv[2]);
const store = new Map(Object.entries(CFG.storage));
globalThis.window = globalThis;
globalThis.localStorage = {
  get length() { return store.size; },
  key(i) { const k = Array.from(store.keys())[i]; return k === undefined ? null : k; },
  getItem(k) { return store.has(k) ? store.get(k) : null; },
  setItem(k, v) { store.set(k, String(v)); },
  removeItem(k) { store.delete(k); },
  clear() { store.clear(); },
};
function classes(names) {
  return {
    add() {}, remove() {}, toggle() { return false; },
    contains(c) { return names.indexOf(c) >= 0; },
  };
}
function stub() {
  const e = {
    value: "", textContent: "", innerHTML: "", innerText: "", checked: false, disabled: false, hidden: false,
    dataset: {}, style: {}, children: [], childNodes: [], classList: classes([]),
    getAttribute() { return null; }, setAttribute() {}, removeAttribute() {}, hasAttribute() { return false; },
    addEventListener() {}, removeEventListener() {}, appendChild(c) { return c; }, insertBefore(c) { return c; },
    removeChild(c) { return c; }, replaceChildren() {}, remove() {}, focus() {}, blur() {}, scrollIntoView() {},
    querySelectorAll() { return []; }, querySelector() { return null; }, closest() { return null; },
  };
  e.elements = new Proxy({}, { get: () => stub() });
  return e;
}
const attrs = CFG.body;
const body = Object.assign(stub(), {
  getAttribute(n) { return Object.prototype.hasOwnProperty.call(attrs, n) ? attrs[n] : null; },
  classList: classes((attrs["class"] || "").split(/\s+/).filter(Boolean)),
});
globalThis.document = {
  body, documentElement: stub(),
  getElementById() { return stub(); }, querySelectorAll() { return []; }, querySelector() { return null; },
  createElement() { return stub(); }, createTextNode() { return stub(); }, addEventListener() {},
};
"""
_NODE_EPILOGUE = "\nprocess.stdout.write(JSON.stringify(window.__state));\n"
_BODY = re.compile(r"<body\b([^>]*)>", re.IGNORECASE)
_ATTR = re.compile(r'([\w-]+)="([^"]*)"')
#: one D7 submission, as A5 names its fields
D7_ENTRY = {
    "service": "branch",
    "name": "Alex Rivera",
    "phone": "512-555-0142",
    "email": "alex@example.com",
    "address": "1200 Barton Springs Rd",
    "description": "",
}


def _page_state(rel: str, storage: dict[str, str]) -> dict[str, Any]:
    """``window.__state`` of the real page ``rel`` under the scenario root, after ``assets/scenarios.js`` runs with
    ``storage`` as its ``localStorage``."""
    if NODE is None:
        pytest.skip("node is a dev prerequisite (T-0008); the B-R2 page row could not run")
    root = scn_dir()
    html = (root / rel).read_text(encoding="utf-8")
    m = _BODY.search(html)
    assert m is not None, f"no <body> in {rel}"
    cfg = {"body": dict(_ATTR.findall(m.group(1))), "storage": storage}
    src = (
        _NODE_PRELUDE
        + (root / "assets" / "scenarios.js").read_text(encoding="utf-8")
        + _NODE_EPILOGUE
    )
    proc = subprocess.run(
        [NODE, "-", json.dumps(cfg)],
        input=src,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ},
    )
    assert proc.returncode == 0, (
        f"scenarios.js failed under the stand-in DOM on {rel}: {proc.stderr[-2000:]}"
    )
    st = json.loads(proc.stdout)
    assert isinstance(st, dict), st
    return st


#: pages off D7's site (web.html, D8's hub, D2's hub), and D7's own pages (the premise)
OFF_SITE = ("web.html", "trips/index.html", "ramen/index.html")
ON_SITE = ("trees/index.html", "trees/quote.html")


@pytest.mark.parametrize("rel", OFF_SITE + ON_SITE)
def test_submissions_union_on_every_page(rel: str) -> None:
    """B-R2 (D7; adversarial, the failure the ruling names: an off-site final page read as a submit press, because
    ``submissions`` was absent): every scenario page and ``web.html`` expose ``submissions`` as the union over every
    ``scn.*.submissions`` key, like ``orders`` (A-R4). With no submission stored it is ``[]`` and the judge's D7
    ``submit_pressed`` is ``False``; with one D7 submission stored it holds that entry and ``submit_pressed`` is
    ``True``, whichever page the run ends on. The trees pages are the premise (D7's own site already reads it)."""
    judge = scn().judge
    st = _page_state(rel, {})
    assert st.get("submissions") == [], (
        f"T-0205 B-R2: {rel} with nothing submitted exposes submissions={st.get('submissions', '<absent>')!r}"
    )
    assert judge("D7", st, "").submit_pressed is False, st
    st = _page_state(rel, {"scn.d7.submissions": json.dumps([D7_ENTRY])})
    assert st.get("submissions") == [D7_ENTRY], (
        f"T-0205 B-R2: {rel} does not show D7's stored submission: {st.get('submissions', '<absent>')!r}"
    )
    assert judge("D7", st, "").submit_pressed is True, st
