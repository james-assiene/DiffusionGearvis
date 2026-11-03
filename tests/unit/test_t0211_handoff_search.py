"""T-0211 group C1: the loopback handoff search index (``local_astra/pages/handoff_search.py``), the pure, static
``SearchBackend`` that answers ``google_search`` in part H of the v3-vs-v4 GPU A/B (docs/tasks/T-0211.md C1, "Tests
to write", Group C, ``tests/unit/test_t0211_handoff_search.py``).

**The surface** (new in C, reached through :func:`hs`, a named ``AssertionError`` "does not exist yet" while absent).
C1 says "the shape of ``rtx_search``", so these names are ``rtx_search``'s: ``SEARCH_ROWS`` (a tuple of ``(title,
url, snippet, keywords)``), ``rank(query)`` (rows best first, at most five; a row or its index), one
``SearchBackend`` class (found by protocol, not by name: reading 1) and ``handoff_search_index_sha()`` (C1 names it).

**Oracles** (never the implementation):

- *Which handoff a row belongs to* is read off the start page, not off the module: the row URL's host words, put
  through ``scenarios.js``'s ``words()`` rule (lower-case, split on ``[^a-z0-9]+``, keep words longer than two
  characters that are not in ``IGNORED``; written out below from ``local_astra/pages/scenarios/assets/scenarios.js``),
  filter ``web.html``'s ``#sites`` list (an item shows iff some word is a substring of its lower-cased text content
  plus its ``data-words``). C1's "filter the list to the right site" is read strictly: exactly one item shows, and
  its link is the handoff's ``start_page`` from ``handoff_tasks.HANDOFFS`` (H1 ``/breville/``, H2
  ``/outdoor/stoves.html``, H3 ``/pizza/``).
- *Prices and trap names* are typed from ``scenarios.js``'s ``PARTS``/``STOVES``/``PIZZA_MENU`` and from
  ``scenario_tasks``' D4/D5/D6 traps: H1 target $39.95, traps the Steam Wand Tip and the BES878 (Barista Pro)
  lookalike; H2 target $54.95, trap the Ridge Twin; H3 target large pepperoni $17.99, trap the wings upsell.
- *Uncut* is a differential: the ``google_search`` response block rendered by ``toolcall.loop`` at the real
  ``MAX_RESPONSE`` equals the block rendered with the bound lifted.

**Readings this module pins where the task is silent** (reported):

1. The backend class is whatever single class in the module satisfies ``contracts.protocols.SearchBackend`` and
   constructs with no arguments (its name is free; C2's "expected loopback class" is that class).
2. "One to two per handoff": every row reaches exactly one handoff's site, and each of H1, H2, H3 has one or two
   rows; no other rows.
3. "Each H request and control request ranks its site's row first": for each of the six texts in
   ``HANDOFFS`` (``request`` and ``control_request``), the first result reaches that handoff's site.
4. Five results render uncut beside the longest of those six texts as the query (every five-row subset).
5. The price and trap rule is on snippets (what the model reads); titles are free to name the site.

BASE = c1680328 (main, ``Merge T-0211-a:`` c3b0c432 reachable). The rows that needed the module were written
``pending(task="T-0211")`` and are active since C's implementation (cc2d5232; the C-R11 snippet row since 0f47146a); the purity rows (cli never names the module, ``rtx_search.py`` and
``test_search_egress_scope.py`` unedited since BASE) are green at BASE and unmarked.

Regime: CPU, pure, n = 1; no socket; Hypothesis at ``max_examples=200``, ``deadline=None``, no database.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import itertools
import re
import shutil
import socket
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.pages.handoff_search"
MODULE_PATH = REPO / "local_astra" / "pages" / "handoff_search.py"
WEB_HTML = REPO / "local_astra" / "pages" / "scenarios" / "web.html"
SCENARIOS_JS = REPO / "local_astra" / "pages" / "scenarios" / "assets" / "scenarios.js"
BASE = "c1680328d70cb4793f2b337947b4103477d4ccc2"
#: ``scenarios.js``'s ``IGNORED``, typed
IGNORED = ("www", "com", "org", "net", "http", "https", "html")
#: handoff -> start page, typed from ``handoff_tasks.HANDOFFS`` (a premise row checks the production table agrees)
START_PAGE = {"H1": "/breville/", "H2": "/outdoor/stoves.html", "H3": "/pizza/"}
#: casefolded patterns no snippet may match: the target's price and the trap items (see the module docstring)
FORBIDDEN = {
    "H1 price $39.95": r"39[.,]95",
    "H1 trap steam wand tip": r"steam\s+wand\s+tip|\btip\b",
    "H1 trap BES878": r"bes\s*-?\s*878",
    "H1 trap Barista Pro": r"barista\s+pro\b",
    "H2 price $54.95": r"54[.,]95",
    "H2 trap Ridge Twin": r"ridge\s+twin",
    "H3 price $17.99": r"17[.,]99",
    "H3 trap wings": r"\bwings?\b",
}
NETWORK = frozenset(
    {
        "socket",
        "ssl",
        "httpx",
        "ddgs",
        "requests",
        "aiohttp",
        "urllib.request",
        "http.client",
        "playwright",
    }
)


# =====================================================================================================
# helpers
# =====================================================================================================


def hs() -> Any:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == MODULE:
            raise AssertionError(f"T-0211 C1: {MODULE} does not exist yet") from exc
        raise


def api(name: str) -> Any:
    m = hs()
    assert hasattr(m, name), f"T-0211 C1: {MODULE} has no {name!r}"
    return getattr(m, name)


def rows() -> list[tuple[str, str, str, Any]]:
    got = api("SEARCH_ROWS")
    assert isinstance(got, tuple) and got, "T-0211 C1: SEARCH_ROWS is a non-empty tuple"
    out = []
    for r in got:
        assert len(r) == 4, f"T-0211 C1: a row is (title, url, snippet, keywords): {r!r}"
        title, url, snippet, keywords = r
        assert all(isinstance(x, str) for x in (title, url, snippet)), r
        out.append((title, url, snippet, keywords))
    return out


def backend_class() -> type:
    """Reading 1: the one class defined in the module that satisfies ``SearchBackend`` with no arguments."""
    from local_astra.contracts.protocols import SearchBackend

    m = hs()
    found = []
    for _name, obj in vars(m).items():
        if inspect.isclass(obj) and obj.__module__ == m.__name__:
            try:
                inst = obj()
            except TypeError:
                continue
            if isinstance(inst, SearchBackend):
                found.append(obj)
    assert len(found) == 1, (
        f"T-0211 C1: expected one SearchBackend class in {MODULE}, found {found}"
    )
    return found[0]


def run_search(backend: Any, query: str) -> Any:
    got = backend.search(query)
    if inspect.isawaitable(got):
        loop = asyncio.new_event_loop()
        try:
            got = loop.run_until_complete(got)
        finally:
            loop.close()
    return got


def search(query: str) -> list[tuple[str, str, str]]:
    from local_astra.contracts.toolcall import SearchResults

    got = run_search(backend_class()(), query)
    assert isinstance(got, SearchResults), f"T-0211 C1: search returns {type(got)!r}"
    return [(r.title, r.url, r.snippet) for r in got.results]


def ranked(query: str) -> list[tuple[str, str, str, Any]]:
    table = rows()
    out = []
    for item in api("rank")(query):
        if isinstance(item, int) and not isinstance(item, bool):
            out.append(table[item])
        else:
            out.append(tuple(item))
    return out


class _Sites(HTMLParser):
    """``web.html``'s ``#sites`` items: (text content, data-words, first link href)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[dict[str, str]] = []
        self._in_sites = False
        self._cur: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "ul" and a.get("id") == "sites":
            self._in_sites = True
        elif tag == "li" and self._in_sites:
            self._cur = {"text": "", "words": a.get("data-words") or "", "href": ""}
        elif tag == "a" and self._cur is not None and not self._cur["href"]:
            self._cur["href"] = a.get("href") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "li" and self._cur is not None:
            self.items.append(self._cur)
            self._cur = None
        elif tag == "ul" and self._in_sites:
            self._in_sites = False

    def handle_data(self, data: str) -> None:
        if self._cur is not None:
            self._cur["text"] += data


def sites() -> list[dict[str, str]]:
    p = _Sites()
    p.feed(WEB_HTML.read_text(encoding="utf-8"))
    assert len(p.items) >= 6, p.items
    return p.items


def js_words(text: str) -> list[str]:
    """``scenarios.js``'s ``words()``: lower-case, split on non-alphanumerics, length > 2, not ``IGNORED``."""
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if len(w) > 2 and w not in IGNORED]


def shown(typed: str) -> list[str]:
    """The hrefs of the ``#sites`` items still shown after ``typed`` is in the address box (``filter()``)."""
    wanted = js_words(typed)
    out = []
    for item in sites():
        hay = (item["text"] + " " + item["words"]).lower()
        if not wanted or any(w in hay for w in wanted):
            out.append(item["href"])
    return out


def host(url: str) -> str:
    m = re.match(r"^https://([^/?#]+)", url)
    assert m, f"T-0211 C1: {url!r} is not an https:// URL"
    return m.group(1)


def handoff_of(url: str) -> str | None:
    """The handoff whose start page is the one item the URL's host words leave shown, else None."""
    got = shown(host(url))
    if len(got) != 1:
        return None
    for hid, page in START_PAGE.items():
        if got[0] == page:
            return hid
    return None


def requests_by_handoff() -> list[tuple[str, str, str]]:
    from local_astra.pages.handoff_tasks import HANDOFFS

    out = []
    for h in HANDOFFS:
        out.append((h.handoff_id, "request", h.request))
        out.append((h.handoff_id, "control_request", h.control_request))
    return out


# =====================================================================================================
# premises (green at BASE)
# =====================================================================================================


def test_start_page_oracle_premise() -> None:
    """Premise: ``scenarios.js`` still carries the ``IGNORED`` list and ``words()`` rule typed here; the handoff table
    names the start pages typed here; ``web.html`` lists each of them once; and the oracle works on the plain host
    words (``breville`` shows only Breville, ``dominos`` only Domino's, an empty box shows all)."""
    from local_astra.pages.handoff_tasks import HANDOFFS

    js = SCENARIOS_JS.read_text(encoding="utf-8")
    assert 'var IGNORED = ["www", "com", "org", "net", "http", "https", "html"];' in js
    assert "w.length > 2 && IGNORED.indexOf(w) < 0" in js
    assert "hay.indexOf(w) >= 0" in js
    assert {h.handoff_id: h.start_page for h in HANDOFFS} == START_PAGE
    hrefs = [i["href"] for i in sites()]
    for page in START_PAGE.values():
        assert hrefs.count(page) == 1, (page, hrefs)
    assert shown("www.breville.com") == ["/breville/"]
    assert shown("dominos.com") == ["/pizza/"]
    assert shown("pinecone-outdoor.com") == ["/outdoor/stoves.html"]
    assert shown("") == hrefs
    # power: a host whose one word is glued fails the strict reading
    assert shown("pineconeoutdoor.com") == []


def test_forbidden_patterns_premise() -> None:
    """Premise: the prices and trap names typed in ``FORBIDDEN`` are the shop's (``scenarios.js``), so the rule is
    about real page facts; and each pattern matches its own fact (power)."""
    js = SCENARIOS_JS.read_text(encoding="utf-8")
    for fact in (
        '["Steam Wand Assembly", "39.95"',
        '["Steam Wand Tip", "9.95"',
        '["Trailhead Duo Stove", "54.95"',
        '["Ridge Twin", "57.50"',
        'large: "17.99"',
        '"8-piece wings"',
        "BES878: [",
        '["Barista Pro", "BES878"]',
    ):
        assert fact in js, fact
    samples = {
        "H1 price $39.95": "only $39.95",
        "H1 trap steam wand tip": "the steam wand tip",
        "H1 trap BES878": "fits the BES878",
        "H1 trap Barista Pro": "barista pro parts",
        "H2 price $54.95": "54.95 dollars",
        "H2 trap Ridge Twin": "ridge twin stove",
        "H3 price $17.99": "large for 17.99",
        "H3 trap wings": "add wings",
    }
    for name, pat in FORBIDDEN.items():
        assert re.search(pat, samples[name].casefold()), name


def test_cli_never_names_the_index() -> None:
    """C1 (adversarial): ``cli`` never constructs the index: no file under ``local_astra/`` other than the module
    itself names ``handoff_search`` (grep). Green at BASE, and stays green after C."""
    hits = []
    for path in (REPO / "local_astra").rglob("*.py"):
        if path == MODULE_PATH:
            continue
        if "handoff_search" in path.read_text(encoding="utf-8"):
            hits.append(str(path.relative_to(REPO)))
    assert hits == [], f"T-0211 C1: {hits} name handoff_search (cli must never construct it)"


def _git_unchanged(rel: str) -> None:
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("no git checkout")
    known = subprocess.run(
        ["git", "cat-file", "-e", f"{BASE}^{{commit}}"], cwd=REPO, capture_output=True, check=False
    )
    if known.returncode != 0:
        pytest.skip(f"BASE {BASE[:8]} is not in this clone")
    proc = subprocess.run(
        ["git", "diff", "--quiet", BASE, "--", rel], cwd=REPO, capture_output=True, check=False
    )
    assert proc.returncode == 0, f"T-0211 C1: {rel} changed since BASE {BASE[:8]}"


@pytest.mark.parametrize(
    "rel", ["local_astra/pages/rtx_search.py", "tests/unit/test_search_egress_scope.py"]
)
def test_rtx_search_and_egress_scope_unedited(rel: str) -> None:
    """C1: the index does not edit ``rtx_search.py`` (T-0209 A2 owns it), and ``test_search_egress_scope.py`` stays
    green *unedited* (its greenness is the fast suite's; this row pins "unedited", working tree against BASE)."""
    _git_unchanged(rel)


# =====================================================================================================
# the module
# =====================================================================================================


def test_is_one_search_backend_and_deterministic() -> None:
    """C1 (reading 1): one class satisfies ``SearchBackend``; two instances and two calls agree; ``search`` returns
    ``rank``'s rows in order with the query echoed; ``handoff_search_index_sha()`` is a stable 64-hex string."""
    from local_astra.contracts.toolcall import SearchResults

    cls = backend_class()
    for _hid, _kind, q in requests_by_handoff():
        a, b = search(q), search(q)
        assert a == b, q
        assert [(t, u, s) for t, u, s, _k in ranked(q)] == a, q
        got = run_search(cls(), q)
        assert isinstance(got, SearchResults) and got.query == q, got
    sha = api("handoff_search_index_sha")()
    assert isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha), sha
    assert api("handoff_search_index_sha")() == sha


def test_rows_reach_their_site_one_or_two_per_handoff() -> None:
    """C1 rows and reachability (reading 2): every URL is ``https://``; its host words, typed into ``web.html``'s
    address box, leave exactly one site shown, and that site's link is H1's, H2's or H3's start page; each handoff
    has one or two rows."""
    per: dict[str, int] = {hid: 0 for hid in START_PAGE}
    for title, url, _s, _k in rows():
        assert title.strip(), url
        assert url.startswith("https://") and not re.search(r"\s", url), url
        hid = handoff_of(url)
        assert hid is not None, (
            f"T-0211 C1: {url}'s host words {js_words(host(url))} leave {shown(host(url))} shown, "
            "not exactly one handoff site"
        )
        per[hid] += 1
    assert all(1 <= n <= 2 for n in per.values()), f"T-0211 C1: rows per handoff {per}"


@pytest.mark.parametrize(
    "case", [f"{h}-{k}" for h in START_PAGE for k in ("request", "control_request")]
)
def test_each_request_ranks_its_site_first(case: str) -> None:
    """C1 (reading 3; "Tests to write": each H request and control request ranks its site's row first)."""
    hid, kind = case.split("-", 1)
    text = {(h, k): t for h, k, t in requests_by_handoff()}[(hid, kind)]
    got = search(text)
    assert 1 <= len(got) <= 5, (text, got)
    first = handoff_of(got[0][1])
    assert first == hid, (
        f"T-0211 C1: {text!r} ranks {[handoff_of(u) for _t, u, _s in got]} ({got[0][1]} first)"
    )


def test_snippets_name_no_price_or_trap() -> None:
    """C1 snippet limits (adversarial): no snippet (casefolded) names a target item's price or a trap item."""
    bad = [
        (name, s)
        for _t, _u, s, _k in rows()
        for name, pat in FORBIDDEN.items()
        if re.search(pat, s.casefold())
    ]
    assert bad == [], f"T-0211 C1: snippets name a price or a trap: {bad}"


#: per handoff, a hand-typed text naming its item (power for the identity-token row)
NAMES_ITEM = {
    "H1": "the Barista Express BES870 steam wand",
    "H2": "the Trailhead Duo Stove",
    "H3": "a large pepperoni pizza",
}


def test_identity_tokens_premise() -> None:
    """Premise for the C-R11 row (green at BASE): every handoff has identity tokens (casefolded regexes,
    ``handoff_tasks``' T-0208 C5 rule), at least one matches the hand-typed text naming its item, and none matches the
    texts naming the other two items (so the C-R11 row cannot pass vacuously, nor fail on a cross-match)."""
    from local_astra.pages.handoff_tasks import HANDOFFS

    assert {h.handoff_id for h in HANDOFFS} == set(NAMES_ITEM)
    for h in HANDOFFS:
        assert h.identity_tokens, h.handoff_id
        for hid, text in NAMES_ITEM.items():
            hit = any(re.search(p, text.casefold()) for p in h.identity_tokens)
            assert hit is (hid == h.handoff_id), (h.handoff_id, hid, h.identity_tokens)


def test_snippets_name_no_handoff_item() -> None:
    """Ruling C-R11 (adversarial): no snippet (casefolded) matches any ``identity_tokens`` pattern of its own handoff
    or of any other handoff. At cc2d5232 the H1 snippet named "Barista Express BES870" and the H2 snippet "Trailhead
    Duo Stove", so a shared turn could name the item without reading the image. Keywords may keep the words (they are
    never rendered); titles are not snippets (reading 5)."""
    from local_astra.pages.handoff_tasks import HANDOFFS

    bad = [
        (handoff_of(u), h.handoff_id, p, s)
        for _t, u, s, _k in rows()
        for h in HANDOFFS
        for p in h.identity_tokens
        if re.search(p, s.casefold())
    ]
    assert bad == [], f"T-0211 C-R11: snippets name a handoff item: {bad}"


def test_snippets_and_five_results_render_uncut(monkeypatch: pytest.MonkeyPatch) -> None:
    """C1 (reading 4): every snippet is at most ``MAX_SNIPPET``; every five-row subset (all rows if fewer), rendered
    as the ``google_search`` response beside the longest H text as query, equals the same render with the bound
    lifted (nothing cut). Power: with ``MAX_RESPONSE`` patched to 400 the two renders differ."""
    from local_astra.contracts.toolcall import SearchResult, SearchResults
    from local_astra.toolcall import loop

    table = rows()
    for _t, url, s, _k in table:
        assert len(s) <= loop.MAX_SNIPPET, (url, len(s))
    query = max((t for _h, _k, t in requests_by_handoff()), key=len)
    real = loop.MAX_RESPONSE

    def render(subset: tuple[tuple[str, str, str, Any], ...]) -> str:
        res = SearchResults(
            query=query,
            results=tuple(SearchResult(title=t, url=u, snippet=s) for t, u, s, _k in subset),
        )
        return loop.render_response("google_search", loop.search_payload(res))

    k = min(5, len(table))
    for subset in itertools.combinations(table, k):
        monkeypatch.setattr(loop, "MAX_RESPONSE", real)
        bounded = render(subset)
        monkeypatch.setattr(loop, "MAX_RESPONSE", 10**9)
        free = render(subset)
        assert bounded == free, f"T-0211 C1: {k} results render cut ({len(free)} > {real})"
    monkeypatch.setattr(loop, "MAX_RESPONSE", 400)
    small = render(tuple(table[:k]))
    monkeypatch.setattr(loop, "MAX_RESPONSE", 10**9)
    assert small != render(tuple(table[:k])), "power: a 400-char bound cut nothing"


@pytest.mark.parametrize("query", ["", "   ", "zzzzqx qqqqvw", "\n\t"])
def test_unmatched_returns_nothing(query: str) -> None:
    """C1 (shape of ``rtx_search``): an empty query, or one matching no row, returns zero results."""
    assert search(query) == [], query


@settings(max_examples=200, deadline=None, database=None)
@given(query=st.text(max_size=200))
def test_never_raises(query: str) -> None:
    """C1 (Hypothesis): ``search`` never raises for any ``str`` and returns at most five results."""
    assert len(search(query)) <= 5


def test_no_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    """C1 (adversarial, a sentinel): building the backend and running every H search touches no socket. Power: the
    sentinel records a direct ``socket.create_connection``."""
    cls = backend_class()
    calls: list[str] = []

    def rec(name: str) -> Any:
        def f(*a: Any, **k: Any) -> Any:
            calls.append(name)
            raise OSError(f"T-0211 sentinel: {name}")

        return f

    for attr in ("connect", "connect_ex", "bind", "sendto"):
        monkeypatch.setattr(socket.socket, attr, rec(attr))
    monkeypatch.setattr(socket, "create_connection", rec("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", rec("getaddrinfo"))
    backend = cls()
    for _h, _k, q in requests_by_handoff():
        run_search(backend, q)
    assert calls == [], f"T-0211 C1: the handoff index touched the network: {calls}"
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", 9))
    assert calls == ["create_connection"], "power: the sentinel records"


def test_imports_no_network_module() -> None:
    """C1 (adversarial, AST plus a fresh interpreter): the module imports only the stdlib and ``local_astra.contracts``
    / ``local_astra.pages`` modules, none of them a network module; importing it in a fresh interpreter loads nothing
    under ``local_astra.search``, ``computer``, ``conductor`` or ``toolcall`` and no ``ddgs``/``httpx``/``ssl``
    beyond what ``contracts.toolcall`` already loads."""
    assert MODULE_PATH.is_file(), f"T-0211 C1: {MODULE_PATH.relative_to(REPO)} does not exist yet"
    bad: list[str] = []
    for node in ast.walk(ast.parse(MODULE_PATH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for n in names:
            top = n.split(".")[0]
            if n in NETWORK or top in NETWORK:
                bad.append(n)
            elif n == "__future__" or n.startswith(("local_astra.contracts", "local_astra.pages")):
                continue
            elif top == "local_astra" or top not in sys.stdlib_module_names:
                bad.append(n)
    assert bad == [], f"T-0211 C1: handoff_search imports {bad}"
    probe = (
        "import sys, importlib\n"
        "importlib.import_module('local_astra.contracts.toolcall')\n"
        "importlib.import_module('local_astra.pages.handoff_tasks')\n"
        "before = set(sys.modules)\n"
        "importlib.import_module(sys.argv[1])\n"
        "print('\\n'.join(sorted(set(sys.modules) - before)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe, MODULE],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    banned = [
        n
        for n in proc.stdout.split()
        if n.split(".")[0] in {"ddgs", "httpx", "ssl", "requests", "aiohttp", "playwright"}
        or n.startswith(
            (
                "local_astra.search",
                "local_astra.computer",
                "local_astra.conductor",
                "local_astra.toolcall",
            )
        )
    ]
    assert banned == [], f"T-0211 C1: importing handoff_search loaded {banned}"


def test_t0211_task_names_this_module() -> None:
    """Bookkeeping: the task file owns this module (Owned files, Group C)."""
    text = (REPO / "docs" / "tasks" / "T-0211.md").read_text(encoding="utf-8")
    assert "tests/unit/test_t0211_handoff_search.py" in text
